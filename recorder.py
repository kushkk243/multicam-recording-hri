import glob
import platform
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_autorefresh import st_autorefresh


# ---------- Helpers ----------

def discover_cameras():
    """List Linux V4L2 devices without opening them."""
    if platform.system() != "Linux":
        return []

    def number(path):
        suffix = path.rsplit("video", 1)[-1]
        return int(suffix) if suffix.isdigit() else 9999

    cameras = []
    for path in sorted(glob.glob("/dev/video*"), key=number):
        device_number = number(path)
        try:
            name = Path(
                f"/sys/class/video4linux/video{device_number}/name"
            ).read_text(encoding="utf-8").strip()
        except OSError:
            name = "Video device"
        cameras.append((path, f"{name} — {path}"))
    return cameras


def camera_index(path):
    return int(path.rsplit("video", 1)[1])


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_") or "item"


def supported_input_rate(device_index):
    device = sd.query_devices(device_index)
    candidates = []
    if device.get("default_samplerate"):
        candidates.append(int(device["default_samplerate"]))
    candidates.extend([48000, 44100, 32000, 24000, 16000])

    for rate in dict.fromkeys(candidates):
        try:
            sd.check_input_settings(
                device=device_index,
                channels=1,
                samplerate=rate,
            )
            return rate
        except Exception:
            pass

    raise RuntimeError(
        f"No supported mono sample rate found for {device['name']}."
    )


def resample_to_16k(audio, source_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate == 16000 or audio.size == 0:
        return audio
    output_length = max(1, round(len(audio) * 16000 / source_rate))
    old_positions = np.arange(len(audio), dtype=np.float32)
    new_positions = np.linspace(
        0, len(audio) - 1, output_length, dtype=np.float32
    )
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


# ---------- Camera ----------

class CameraFeed:
    """Camera preview; records video if output_path is provided."""

    def __init__(self, source, name, width, height, fps, output_path=None):
        self.source = source
        self.display_name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = str(output_path) if output_path else None
        self.frame = None
        self.error = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            index = camera_index(self.source)
        except (IndexError, ValueError):
            self.error = f"Invalid camera path: {self.source}"
            return

        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            self.error = f"Could not open {self.display_name}"
            cap.release()
            return

        writer = None
        if self.output_path:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                actual_fps,
                (actual_width, actual_height),
            )
            if not writer.isOpened():
                self.error = f"Could not create {self.output_path}"
                cap.release()
                return

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if writer:
                    writer.write(frame)
                with self.lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
            if writer:
                writer.release()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10)


# ---------- Audio ----------

class AudioRecorder:
    """Records an input device; optionally queues audio for transcription."""

    def __init__(self, device_index, wav_path, transcription_queue=None):
        self.device_index = device_index
        self.wav_path = str(wav_path)
        self.transcription_queue = transcription_queue
        self.sample_rate = supported_input_rate(device_index)
        self.chunk_samples = self.sample_rate * 5
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.stream = None
        self.error = None
        self.writer_thread = threading.Thread(target=self._write, daemon=True)

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
        if status and self.transcription_queue is not None:
            self.transcription_queue.put(f"[Audio status: {status}]")
        self.audio_queue.put(indata[:, 0].copy())

    def _write(self):
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
                    if self.transcription_queue is not None:
                        pending.extend(samples.tolist())
                        while len(pending) >= self.chunk_samples:
                            chunk = np.asarray(
                                pending[:self.chunk_samples], dtype=np.float32
                            )
                            pending = pending[self.chunk_samples:]
                            self.transcription_queue.put(
                                resample_to_16k(chunk, self.sample_rate)
                            )

                if pending and self.transcription_queue is not None:
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


# ---------- Transcription ----------

class Transcriber:
    def __init__(self, audio_queue, model_name, language):
        self.audio_queue = audio_queue
        self.text_queue = queue.Queue()
        self.model_name = model_name
        self.language = language or None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
            )
            while not self.stop_event.is_set() or not self.audio_queue.empty():
                try:
                    audio = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if isinstance(audio, str):
                    self.text_queue.put(audio)
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
                    self.text_queue.put(text)
        except Exception as exc:
            self.text_queue.put(f"[Transcription error: {exc}]")

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


# ---------- Session cleanup ----------

def stop_workers():
    """Stop all active workers and save the transcript."""
    for worker in st.session_state.cameras:
        worker.stop()

    if st.session_state.audio:
        st.session_state.audio.stop()

    for worker in st.session_state.extra_audio:
        worker.stop()

    transcriber = st.session_state.transcriber
    if transcriber:
        transcriber.stop()
        while not transcriber.text_queue.empty():
            st.session_state.transcript_lines.append(
                transcriber.text_queue.get_nowait()
            )

    session_dir = st.session_state.session_dir
    if session_dir:
        (session_dir / "transcript.txt").write_text(
            "\n".join(st.session_state.transcript_lines),
            encoding="utf-8",
        )

    st.session_state.cameras = []
    st.session_state.audio = None
    st.session_state.extra_audio = []
    st.session_state.transcriber = None
    st.session_state.transcript_queue = None
    st.session_state.recording = False


# ---------- Streamlit app ----------

st.set_page_config(page_title="Study Recorder", layout="wide")
st.title("Multicamera Study Recorder")

defaults = {
    "recording": False,
    "previews": [],
    "cameras": [],
    "audio": None,
    "extra_audio": [],
    "transcriber": None,
    "transcript_lines": [],
    "transcript_queue": None,
    "session_dir": None,
    "camera_options": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.camera_options is None:
    st.session_state.camera_options = discover_cameras()

with st.sidebar:
    st.header("Session setup")
    participant = st.text_input("Participant ID", "P001")
    location = st.text_input("Location", "Room 1")
    output_root = st.text_input("Output folder", "recordings")

    st.subheader("Cameras")
    selected_cameras = st.multiselect(
        "Select cameras",
        options=st.session_state.camera_options,
        format_func=lambda item: item[1],
        key="camera_selector",
        disabled=st.session_state.recording,
    )

    refresh_cameras = st.button(
        "Refresh camera list",
        disabled=st.session_state.recording or bool(st.session_state.previews),
    )

    width = st.number_input("Requested camera width", 320, 7680, 1920, step=160)
    height = st.number_input("Requested camera height", 240, 4320, 1080, step=120)
    fps = st.number_input("Requested camera FPS", 1, 120, 60)

    audio_inputs = [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]

    st.subheader("Primary microphone")
    mic = st.selectbox(
        "Mic used for transcription",
        audio_inputs,
        format_func=lambda item: f"{item[0]} — {item[1]}",
    ) if audio_inputs else None

    st.subheader("Additional audio")
    extra_audio_devices = st.multiselect(
        "Other audio inputs to record separately",
        audio_inputs,
        format_func=lambda item: f"{item[0]} — {item[1]}",
    )

    st.subheader("Transcription")
    model_name = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
    )
    language = st.text_input("Language code (blank for automatic detection)", "en")

    # Show only the appropriate recording control for the current state.
    preview_button = st.button(
        "Preview cameras",
        disabled=st.session_state.recording or not selected_cameras,
    )
    stop_preview_button = st.button(
        "Stop preview",
        disabled=st.session_state.recording or not st.session_state.previews,
    )

    if st.session_state.recording:
        stop_button = st.button("Stop recording", key="stop_recording")
        start_button = False
    else:
        start_button = st.button(
            "Start recording",
            key="start_recording",
            disabled=not selected_cameras or mic is None,
        )
        stop_button = False


# Refresh only on explicit request; discovery never opens camera devices.
if refresh_cameras:
    st.session_state.camera_options = discover_cameras()
    if "camera_selector" in st.session_state:
        del st.session_state["camera_selector"]
    st.rerun()


# Preview controls
if preview_button:
    for feed in st.session_state.previews:
        feed.stop()

    st.session_state.previews = [
        CameraFeed(source, name, int(width), int(height), int(fps))
        for source, name in selected_cameras
    ]
    for feed in st.session_state.previews:
        feed.start()
    st.rerun()

if stop_preview_button:
    for feed in st.session_state.previews:
        feed.stop()
    st.session_state.previews = []
    st.rerun()


# Start recording
if start_button:
    started_cameras = []
    started_audio = None
    started_extra_audio = []
    started_transcriber = None

    try:
        for feed in st.session_state.previews:
            feed.stop()
        st.session_state.previews = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = (
            Path(output_root)
            / f"{safe_name(participant)}_{safe_name(location)}_{timestamp}"
        )
        session_dir.mkdir(parents=True, exist_ok=True)

        transcript_queue = queue.Queue()
        camera_workers = []

        for number, (source, name) in enumerate(selected_cameras, start=1):
            worker = CameraFeed(
                source,
                name,
                int(width),
                int(height),
                int(fps),
                session_dir / f"camera_{number}_{safe_name(name)}.mp4",
            )
            worker.start()
            camera_workers.append(worker)
            started_cameras.append(worker)

        time.sleep(0.3)
        errors = [worker.error for worker in camera_workers if worker.error]
        if errors:
            raise RuntimeError("; ".join(errors))

        started_audio = AudioRecorder(
            mic[0],
            session_dir / "microphone.wav",
            transcription_queue=transcript_queue,
        )
        started_audio.start()
        if started_audio.error:
            raise RuntimeError(
                f"Could not start primary microphone: {started_audio.error}"
            )

        for number, (device_index, device_name) in enumerate(
            extra_audio_devices, start=1
        ):
            worker = AudioRecorder(
                device_index,
                session_dir
                / f"additional_audio_{number}_{safe_name(device_name)}.wav",
            )
            worker.start()
            if worker.error:
                raise RuntimeError(
                    f"Could not start audio input {device_name}: {worker.error}"
                )
            started_extra_audio.append(worker)

        started_transcriber = Transcriber(
            transcript_queue,
            model_name,
            language.strip(),
        )
        started_transcriber.start()

        (session_dir / "session_info.txt").write_text(
            "\n".join([
                f"Participant: {participant}",
                f"Location: {location}",
                f"Start time: {datetime.now().astimezone().isoformat()}",
                f"Cameras: {selected_cameras}",
                f"Requested resolution: {width}x{height}",
                f"Requested FPS: {fps}",
                f"Primary microphone: {mic}",
                f"Primary microphone sample rate: {started_audio.sample_rate}",
                f"Additional audio devices: {extra_audio_devices}",
            ]),
            encoding="utf-8",
        )

        st.session_state.cameras = camera_workers
        st.session_state.audio = started_audio
        st.session_state.extra_audio = started_extra_audio
        st.session_state.transcriber = started_transcriber
        st.session_state.transcript_queue = transcript_queue
        st.session_state.transcript_lines = []
        st.session_state.session_dir = session_dir
        st.session_state.recording = True
        st.rerun()

    except Exception as exc:
        for worker in started_cameras:
            worker.stop()
        if started_audio:
            started_audio.stop()
        for worker in started_extra_audio:
            worker.stop()
        if started_transcriber:
            started_transcriber.stop()
        st.error(str(exc))


# Stop recording: finish cleanup, change state, then rerun the app.
if stop_button:
    stop_workers()
    st.success(f"Files saved in: {st.session_state.session_dir}")
    st.rerun()


# Camera multiview
left, right = st.columns([3, 1])

with left:
    st.subheader("Camera views")
    feeds = (
        st.session_state.cameras
        if st.session_state.recording
        else st.session_state.previews
    )

    if feeds:
        for start in range(0, len(feeds), 3):
            row = feeds[start:start + 3]
            columns = st.columns(len(row))
            for column, feed in zip(columns, row):
                with column:
                    st.markdown(f"**{feed.display_name}**")
                    frame = feed.get_frame()
                    if frame is not None:
                        st.image(frame, width="stretch")
                    elif feed.error:
                        st.error(feed.error)
                    else:
                        st.info("Waiting for camera frame…")
    else:
        st.info("Select cameras, then click **Preview cameras**.")

with right:
    st.subheader("Live transcription")
    if st.session_state.recording and st.session_state.transcriber:
        text_queue = st.session_state.transcriber.text_queue
        while not text_queue.empty():
            st.session_state.transcript_lines.append(text_queue.get_nowait())

    st.text_area(
        "Recognized speech",
        "\n\n".join(st.session_state.transcript_lines[-40:]),
        height=500,
        disabled=True,
    )

if st.session_state.recording:
    st_autorefresh(interval=500, key="recording_refresh")
    st.caption(f"Recording to: {st.session_state.session_dir}")
elif st.session_state.previews:
    st_autorefresh(interval=500, key="preview_refresh")
    st.caption("Preview only — no files are being recorded.")