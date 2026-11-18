import glob
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------- Device discovery and helpers ----------

def discover_cameras():
    """List Linux V4L2 devices without opening them."""
    cameras = []

    def number(path):
        suffix = path.rsplit("video", 1)[-1]
        return int(suffix) if suffix.isdigit() else 9999

    for path in sorted(glob.glob("/dev/video*"), key=number):
        suffix = path.rsplit("video", 1)[-1]
        if not suffix.isdigit():
            continue

        try:
            name = Path(
                f"/sys/class/video4linux/video{suffix}/name"
            ).read_text(encoding="utf-8").strip()
        except OSError:
            name = "Video device"

        cameras.append((path, f"{name} — {path}"))

    return cameras


def discover_alsa_inputs():
    """
    Get FFmpeg ALSA capture-device identifiers from `arecord -L`.

    Device identifiers such as `default`, `hw:CARD=...,DEV=...`, or `plughw:...`
    are passed directly to FFmpeg. Availability depends on the installed ALSA
    configuration and hardware.
    """
    arecord = shutil.which("arecord")
    if not arecord:
        return []

    try:
        result = subprocess.run(
            [arecord, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    devices = []
    current_name = None
    current_description = []

    def add_current():
        if current_name:
            description = " ".join(current_description).strip()
            label = f"{current_name} — {description}" if description else current_name
            devices.append((current_name, label))

    for line in result.stdout.splitlines():
        if not line.strip():
            add_current()
            current_name = None
            current_description = []
        elif line[:1].isspace():
            current_description.append(line.strip())
        else:
            add_current()
            current_name = line.strip()
            current_description = []

    add_current()

    # `arecord -L` can list aliases and plugin devices. Keep the order but
    # remove duplicate identifiers.
    seen = set()
    unique = []
    for identifier, label in devices:
        if identifier not in seen:
            unique.append((identifier, label))
            seen.add(identifier)

    return unique


def safe_name(text):
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return result or "item"


def supported_mic_rate(device_index):
    """Choose a mono sample rate supported by the primary microphone."""
    info = sd.query_devices(device_index)
    rates = []

    default_rate = info.get("default_samplerate")
    if default_rate:
        rates.append(int(default_rate))
    rates.extend([48000, 44100, 32000, 24000, 16000])

    for rate in dict.fromkeys(rates):
        try:
            sd.check_input_settings(
                device=device_index,
                channels=1,
                samplerate=rate,
            )
            return rate
        except Exception:
            continue

    raise RuntimeError(
        f"No supported mono sample rate found for microphone "
        f"{info.get('name', device_index)}."
    )


def resample_to_16k(audio, source_rate):
    """Resample float audio chunks for Whisper."""
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate == 16000 or audio.size == 0:
        return audio

    output_length = max(1, round(len(audio) * 16000 / source_rate))
    old_positions = np.arange(len(audio), dtype=np.float32)
    new_positions = np.linspace(
        0, len(audio) - 1, output_length, dtype=np.float32
    )
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


# ---------- Signals ----------

class AppSignals(QObject):
    camera_frame = Signal(str, QImage)
    camera_error = Signal(str, str)
    transcript = Signal(str)
    status = Signal(str)


# ---------- Camera preview (not used for recording) ----------

class CameraPreview:
    """OpenCV preview worker. Recording itself is performed by FFmpeg."""

    def __init__(self, source, display_name, width, height, fps, signals):
        self.source = source
        self.display_name = display_name
        self.width = width
        self.height = height
        self.fps = fps
        self.signals = signals
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        suffix = self.source.rsplit("video", 1)[-1]
        if not suffix.isdigit():
            self.signals.camera_error.emit(
                self.source, f"Invalid camera path: {self.source}"
            )
            return

        cap = cv2.VideoCapture(int(suffix), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            self.signals.camera_error.emit(
                self.source, f"Could not open {self.display_name}"
            )
            cap.release()
            return

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.signals.camera_frame.emit(self.source, image)
        finally:
            cap.release()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


# ---------- FFmpeg camera recorder ----------

class FFmpegCameraRecorder:
    """Record one camera and its optionally assigned ALSA audio into an MP4."""

    def __init__(
        self,
        source,
        display_name,
        output_path,
        width,
        height,
        fps,
        alsa_device=None,
    ):
        self.source = source
        self.display_name = display_name
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.alsa_device = alsa_device
        self.process = None
        self.error = None
        self.stderr_thread = None
        self.stderr_tail = queue.Queue()

    def start(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg executable was not found on PATH.")

        suffix = self.source.rsplit("video", 1)[-1]
        if not suffix.isdigit():
            raise RuntimeError(f"Invalid camera path: {self.source}")

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-thread_queue_size", "512",
            "-f", "v4l2",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.source,
        ]

        if self.alsa_device:
            command += [
                "-thread_queue_size", "512",
                "-f", "alsa",
                "-i", self.alsa_device,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "160k",
                "-shortest",
            ]
        else:
            command += [
                "-map", "0:v:0",
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
            ]

        command += [
            "-movflags", "+faststart",
            str(self.output_path),
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stderr_thread = threading.Thread(
            target=self._collect_stderr,
            daemon=True,
        )
        self.stderr_thread.start()

        # Check promptly for common startup failures.
        time.sleep(0.4)
        if self.process.poll() is not None:
            self.error = self._read_stderr_tail()
            raise RuntimeError(
                f"FFmpeg could not start recording {self.display_name}.\n"
                f"{self.error}"
            )

    def _collect_stderr(self):
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            self.stderr_tail.put(line.rstrip())
            while self.stderr_tail.qsize() > 30:
                try:
                    self.stderr_tail.get_nowait()
                except queue.Empty:
                    break

    def _read_stderr_tail(self):
        lines = []
        while not self.stderr_tail.empty():
            try:
                lines.append(self.stderr_tail.get_nowait())
            except queue.Empty:
                break
        return "\n".join(lines) or "No FFmpeg diagnostic output."

    def stop(self):
        if not self.process:
            return

        if self.process.poll() is None:
            # SIGINT lets FFmpeg finalize the MP4 container.
            self.process.send_signal(2)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)

        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=2)

        if self.process.returncode not in (0, 255, -2, None):
            self.error = self._read_stderr_tail()


# ---------- Primary microphone recorder ----------

class MicrophoneRecorder:
    """Record the selected actual microphone and queue audio for transcription."""

    def __init__(self, device_index, wav_path, transcription_queue):
        self.device_index = device_index
        self.wav_path = str(wav_path)
        self.transcription_queue = transcription_queue
        self.sample_rate = supported_mic_rate(device_index)
        self.chunk_samples = self.sample_rate * 5

        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.stream = None
        self.error = None
        self.writer_thread = threading.Thread(
            target=self._write_audio,
            daemon=True,
        )

    def start(self):
        self.writer_thread.start()
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                callback=self._callback,
                blocksize=1024,
            )
            self.stream.start()
        except Exception as exc:
            self.error = str(exc)
            self.stop_event.set()

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.transcription_queue.put(f"[Microphone status: {status}]")
        self.audio_queue.put(indata[:, 0].copy())

    def _write_audio(self):
        pending = []
        try:
            with sf.SoundFile(
                self.wav_path,
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as wav:
                while not self.stop_event.is_set() or not self.audio_queue.empty():
                    try:
                        samples = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    wav.write(samples)
                    pending.extend(samples.tolist())

                    while len(pending) >= self.chunk_samples:
                        chunk = np.asarray(
                            pending[:self.chunk_samples],
                            dtype=np.float32,
                        )
                        pending = pending[self.chunk_samples:]
                        self.transcription_queue.put(
                            resample_to_16k(chunk, self.sample_rate)
                        )

                if pending:
                    self.transcription_queue.put(
                        resample_to_16k(
                            np.asarray(pending, dtype=np.float32),
                            self.sample_rate,
                        )
                    )
        except Exception as exc:
            self.error = str(exc)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.stop_event.set()
        if self.writer_thread.is_alive():
            self.writer_thread.join(timeout=10)


# ---------- Transcriber ----------

class Transcriber:
    def __init__(self, audio_queue, model_name, language, signals):
        self.audio_queue = audio_queue
        self.model_name = model_name
        self.language = language or None
        self.signals = signals
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            self.signals.status.emit(
                f"Loading transcription model: {self.model_name}"
            )
            model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
            )
            self.signals.status.emit("Transcription model ready.")

            while not self.stop_event.is_set() or not self.audio_queue.empty():
                try:
                    audio = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if isinstance(audio, str):
                    self.signals.transcript.emit(audio)
                    continue

                segments, _ = model.transcribe(
                    audio,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=True,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    self.signals.transcript.emit(text)

        except Exception as exc:
            self.signals.transcript.emit(f"[Transcription error: {exc}]")

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=8)


# ---------- Main window ----------

class StudyRecorderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multicamera Study Recorder")
        self.resize(1200, 800)
        self.setMinimumSize(720, 500)

        self.signals = AppSignals()
        self.signals.camera_frame.connect(self.update_camera_frame)
        self.signals.camera_error.connect(self.show_camera_error)
        self.signals.transcript.connect(self.add_transcript)
        self.signals.status.connect(self.show_status)

        self.recording = False
        self.previewing = False
        self.preview_workers = []
        self.camera_recorders = []
        self.mic_recorder = None
        self.transcriber = None
        self.transcript_queue = None
        self.transcript_lines = []
        self.session_dir = None
        self.camera_labels = {}
        self.camera_checkboxes = []
        self.camera_audio_combos = []

        self.build_ui()
        self.refresh_devices()
        self.update_buttons()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Controls in a scrollable panel.
        controls_content = QWidget()
        controls_layout = QVBoxLayout(controls_content)

        session_group = QGroupBox("Session")
        session_form = QFormLayout(session_group)
        self.participant_edit = QLineEdit("P001")
        self.location_edit = QLineEdit("Room 1")
        self.output_edit = QLineEdit("recordings")
        session_form.addRow("Participant:", self.participant_edit)
        session_form.addRow("Location:", self.location_edit)
        session_form.addRow("Output folder:", self.output_edit)
        controls_layout.addWidget(session_group)

        camera_group = QGroupBox("Cameras and their audio")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_list_widget = QWidget()
        self.camera_list_layout = QVBoxLayout(self.camera_list_widget)
        self.camera_list_layout.setContentsMargins(0, 0, 0, 0)

        camera_scroll = QScrollArea()
        camera_scroll.setWidgetResizable(True)
        camera_scroll.setMinimumHeight(140)
        camera_scroll.setMaximumHeight(330)
        camera_scroll.setWidget(self.camera_list_widget)
        camera_layout.addWidget(camera_scroll)

        self.refresh_button = QPushButton("Refresh device lists")
        self.refresh_button.clicked.connect(self.refresh_devices)
        camera_layout.addWidget(self.refresh_button)

        camera_form = QFormLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 7680)
        self.width_spin.setValue(1920)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 4320)
        self.height_spin.setValue(1080)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        camera_form.addRow("Width:", self.width_spin)
        camera_form.addRow("Height:", self.height_spin)
        camera_form.addRow("FPS:", self.fps_spin)
        camera_layout.addLayout(camera_form)
        controls_layout.addWidget(camera_group)

        mic_group = QGroupBox("Actual microphone for transcription")
        mic_form = QFormLayout(mic_group)
        self.mic_combo = QComboBox()
        mic_form.addRow("Microphone:", self.mic_combo)
        controls_layout.addWidget(mic_group)

        transcription_group = QGroupBox("Transcription")
        transcription_form = QFormLayout(transcription_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("small")
        self.language_edit = QLineEdit("en")
        self.language_edit.setPlaceholderText("Blank for auto-detect")
        transcription_form.addRow("Model:", self.model_combo)
        transcription_form.addRow("Language:", self.language_edit)
        controls_layout.addWidget(transcription_group)

        self.preview_button = QPushButton("Preview cameras")
        self.preview_button.clicked.connect(self.start_preview)
        controls_layout.addWidget(self.preview_button)

        self.stop_preview_button = QPushButton("Stop preview")
        self.stop_preview_button.clicked.connect(self.stop_preview)
        controls_layout.addWidget(self.stop_preview_button)

        self.start_button = QPushButton("Start recording")
        self.start_button.clicked.connect(self.start_recording)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop recording")
        self.stop_button.clicked.connect(self.stop_recording)
        controls_layout.addWidget(self.stop_button)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(280)
        controls_scroll.setMaximumWidth(470)
        controls_scroll.setWidget(controls_content)
        splitter.addWidget(controls_scroll)

        # Resizable preview/transcription area.
        views_content = QWidget()
        views_layout = QVBoxLayout(views_content)

        self.camera_group = QGroupBox("Camera previews")
        self.camera_grid_widget = QWidget()
        self.camera_grid = QGridLayout(self.camera_grid_widget)
        self.camera_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        camera_layout = QVBoxLayout(self.camera_group)
        camera_layout.addWidget(self.camera_grid_widget)
        views_layout.addWidget(self.camera_group, stretch=3)

        transcript_group = QGroupBox("Live transcription — primary microphone")
        transcript_layout = QVBoxLayout(transcript_group)
        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_view)
        views_layout.addWidget(transcript_group, stretch=2)

        views_scroll = QScrollArea()
        views_scroll.setWidgetResizable(True)
        views_scroll.setWidget(views_content)
        splitter.addWidget(views_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 810])

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def refresh_devices(self):
        if self.recording or self.previewing:
            return

        old_camera_selection = {
            checkbox.property("path")
            for checkbox in self.camera_checkboxes
            if checkbox.isChecked()
        }
        old_audio_selection = {}
        for checkbox, combo in self.camera_audio_combos:
            old_audio_selection[checkbox.property("path")] = combo.currentData()

        self.clear_layout(self.camera_list_layout)
        self.camera_checkboxes = []
        self.camera_audio_combos = []

        cameras = discover_cameras()
        audio_devices = discover_alsa_inputs()

        if not cameras:
            self.camera_list_layout.addWidget(
                QLabel("No /dev/video* cameras found.")
            )

        for path, name in cameras:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 6)

            checkbox = QCheckBox(name)
            checkbox.setProperty("path", path)
            checkbox.setChecked(path in old_camera_selection)
            row_layout.addWidget(checkbox)

            audio_row = QHBoxLayout()
            audio_row.addWidget(QLabel("Audio:"))

            combo = QComboBox()
            combo.addItem("No camera audio", None)
            for identifier, label in audio_devices:
                combo.addItem(label, identifier)

            previous_identifier = old_audio_selection.get(path)
            if previous_identifier:
                index = combo.findData(previous_identifier)
                if index >= 0:
                    combo.setCurrentIndex(index)

            audio_row.addWidget(combo, stretch=1)
            row_layout.addLayout(audio_row)

            self.camera_list_layout.addWidget(row)
            self.camera_checkboxes.append(checkbox)
            self.camera_audio_combos.append((checkbox, combo))

        try:
            sd_devices = sd.query_devices()
            mic_devices = [
                (index, device["name"])
                for index, device in enumerate(sd_devices)
                if device["max_input_channels"] > 0
            ]
        except Exception as exc:
            mic_devices = []
            self.status_label.setText(f"Could not list microphones: {exc}")

        previous_mic = self.mic_combo.currentData()
        self.mic_combo.clear()

        for index, name in mic_devices:
            self.mic_combo.addItem(f"{name} (device {index})", index)

        if previous_mic is not None:
            position = self.mic_combo.findData(previous_mic)
            if position >= 0:
                self.mic_combo.setCurrentIndex(position)

        if not mic_devices:
            self.mic_combo.addItem("No microphone inputs found", None)

        self.update_buttons()

    def selected_cameras(self):
        selected = []
        for checkbox, combo in self.camera_audio_combos:
            if checkbox.isChecked():
                selected.append(
                    (
                        checkbox.property("path"),
                        checkbox.text(),
                        combo.currentData(),
                        combo.currentText(),
                    )
                )
        return selected

    def update_buttons(self):
        selected = self.selected_cameras()
        has_cameras = bool(selected)
        has_mic = self.mic_combo.currentData() is not None

        self.preview_button.setEnabled(
            not self.recording and not self.previewing and has_cameras
        )
        self.stop_preview_button.setEnabled(
            not self.recording and self.previewing
        )
        self.start_button.setEnabled(
            not self.recording and has_cameras and has_mic
        )
        self.stop_button.setEnabled(self.recording)

        enabled = not self.recording and not self.previewing
        self.refresh_button.setEnabled(enabled)
        for checkbox, combo in self.camera_audio_combos:
            checkbox.setEnabled(enabled)
            combo.setEnabled(enabled)

    def create_camera_labels(self, selected):
        self.clear_layout(self.camera_grid)
        self.camera_labels = {}

        for position, (source, name, _audio_id, _audio_label) in enumerate(selected):
            box = QGroupBox(name)
            box_layout = QVBoxLayout(box)

            label = QLabel("Waiting for camera…")
            label.setMinimumSize(160, 110)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: #202020; color: white;")
            box_layout.addWidget(label)

            row, column = divmod(position, 2)
            self.camera_grid.addWidget(box, row, column)
            self.camera_labels[source] = label

    def update_camera_frame(self, source, image):
        label = self.camera_labels.get(source)
        if label is None:
            return

        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def show_camera_error(self, source, message):
        label = self.camera_labels.get(source)
        if label:
            label.setText(message)
        self.status_label.setText(message)

    def start_preview(self):
        selected = self.selected_cameras()
        if not selected:
            QMessageBox.warning(self, "No camera", "Select at least one camera.")
            return

        self.create_camera_labels(selected)
        self.preview_workers = [
            CameraPreview(
                source,
                name,
                self.width_spin.value(),
                self.height_spin.value(),
                self.fps_spin.value(),
                self.signals,
            )
            for source, name, _audio_id, _audio_label in selected
        ]

        for worker in self.preview_workers:
            worker.start()

        self.previewing = True
        self.status_label.setText("Preview active — not recording.")
        self.update_buttons()

    def stop_preview(self):
        for worker in self.preview_workers:
            worker.stop()

        self.preview_workers = []
        self.previewing = False
        self.status_label.setText("Preview stopped.")
        self.update_buttons()

    def start_recording(self):
        selected = self.selected_cameras()
        mic_index = self.mic_combo.currentData()

        if not selected:
            QMessageBox.warning(self, "No camera", "Select at least one camera.")
            return
        if mic_index is None:
            QMessageBox.warning(
                self, "No microphone", "Select a primary transcription microphone."
            )
            return
        if not shutil.which("ffmpeg"):
            QMessageBox.critical(
                self, "FFmpeg not found",
                "Install FFmpeg and make sure it is available on PATH.",
            )
            return

        try:
            if self.previewing:
                self.stop_preview()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = (
                Path(self.output_edit.text().strip() or "recordings")
                / f"{safe_name(self.participant_edit.text())}_"
                  f"{safe_name(self.location_edit.text())}_{timestamp}"
            )
            session_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = session_dir
            self.transcript_lines = []
            self.transcript_view.clear()
            self.transcript_queue = queue.Queue()

            self.create_camera_labels(selected)
            self.camera_recorders = []

            for number, (source, name, audio_id, audio_label) in enumerate(
                selected, start=1
            ):
                output_path = (
                    session_dir / f"camera_{number}_{safe_name(name)}.mp4"
                )
                recorder = FFmpegCameraRecorder(
                    source=source,
                    display_name=name,
                    output_path=output_path,
                    width=self.width_spin.value(),
                    height=self.height_spin.value(),
                    fps=self.fps_spin.value(),
                    alsa_device=audio_id,
                )
                recorder.start()
                self.camera_recorders.append(recorder)

            self.mic_recorder = MicrophoneRecorder(
                mic_index,
                session_dir / "microphone.wav",
                self.transcript_queue,
            )
            self.mic_recorder.start()
            if self.mic_recorder.error:
                raise RuntimeError(
                    f"Could not start the primary microphone: "
                    f"{self.mic_recorder.error}"
                )

            self.transcriber = Transcriber(
                self.transcript_queue,
                self.model_combo.currentText(),
                self.language_edit.text().strip(),
                self.signals,
            )
            self.transcriber.start()

            details = [
                f"Participant: {self.participant_edit.text()}",
                f"Location: {self.location_edit.text()}",
                f"Session start: {datetime.now().astimezone().isoformat()}",
                f"Requested video size: "
                f"{self.width_spin.value()}x{self.height_spin.value()}",
                f"Requested FPS: {self.fps_spin.value()}",
                f"Primary microphone index: {mic_index}",
                f"Primary microphone sample rate: "
                f"{self.mic_recorder.sample_rate}",
                "Camera inputs:",
            ]
            for source, name, audio_id, audio_label in selected:
                details.append(
                    f"  {name}: video={source}; "
                    f"camera_audio={audio_id or 'none'} ({audio_label})"
                )

            (session_dir / "session_info.txt").write_text(
                "\n".join(details),
                encoding="utf-8",
            )

            self.recording = True
            self.status_label.setText(f"Recording to: {session_dir}")
            self.update_buttons()

        except Exception as exc:
            self.stop_recording_workers()
            self.recording = False
            self.update_buttons()
            QMessageBox.critical(self, "Could not start recording", str(exc))
            self.status_label.setText(f"Could not start recording: {exc}")

    def stop_recording_workers(self):
        errors = []

        for recorder in self.camera_recorders:
            recorder.stop()
            if recorder.error:
                errors.append(
                    f"{recorder.display_name}: {recorder.error}"
                )
        self.camera_recorders = []

        if self.mic_recorder:
            self.mic_recorder.stop()
            if self.mic_recorder.error:
                errors.append(f"Microphone: {self.mic_recorder.error}")
            self.mic_recorder = None

        if self.transcriber:
            self.transcriber.stop()
            while not self.transcriber.audio_queue.empty():
                # Audio still waiting for transcription is intentionally drained
                # by the transcriber during its stop routine.
                break
            self.transcriber = None

        if self.session_dir:
            (self.session_dir / "transcript.txt").write_text(
                "\n".join(self.transcript_lines),
                encoding="utf-8",
            )

        return errors

    def stop_recording(self):
        if not self.recording:
            return

        self.status_label.setText("Stopping FFmpeg and saving files…")
        QApplication.processEvents()

        errors = self.stop_recording_workers()
        self.recording = False
        self.update_buttons()

        if errors:
            QMessageBox.warning(
                self,
                "Recording stopped with warnings",
                "\n".join(errors),
            )

        self.status_label.setText(
            f"Recording stopped. Files saved in: {self.session_dir}"
        )

    def add_transcript(self, text):
        self.transcript_lines.append(text)
        self.transcript_view.append(text)

    def show_status(self, text):
        self.status_label.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for label in self.camera_labels.values():
            pixmap = label.pixmap()
            if pixmap and not pixmap.isNull():
                label.setPixmap(
                    pixmap.scaled(
                        label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def closeEvent(self, event):
        if self.previewing:
            self.stop_preview()

        if self.recording:
            answer = QMessageBox.question(
                self,
                "Recording in progress",
                "Stop recording and close the app?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.stop_recording()

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = StudyRecorderWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()