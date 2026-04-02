"""
tools/ai_commands.py
GPT-4o natural language → Crazyflie flight step list.
Also handles Whisper voice transcription.
All network I/O runs in daemon threads; main.py stays at 100 Hz.
"""

import io
import json
import os
import queue
import threading
import traceback

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI

# ---------------------------------------------------------------------------
# Speed limits (mirror from main.py so GPT knows the caps)
# ---------------------------------------------------------------------------
MAX_FORWARD_SPEED  = 1.2    # m/s
MAX_STRAFE_SPEED   = 0.8    # m/s
MAX_VERTICAL_SPEED = 0.35   # m/s
MAX_YAW_RATE       = 80.0   # deg/s
SAMPLE_RATE        = 16000  # Hz — Whisper works well at 16 kHz

# ---------------------------------------------------------------------------
# Public result queue — main.py reads from this each frame.
# Each item is either:
#   ("steps",  [step, ...])    – command parsed successfully
#   ("status", "some text")    – progress / error string for UI
# ---------------------------------------------------------------------------
result_queue: queue.Queue = queue.Queue()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — add it to your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""
You are a flight command parser for a Crazyflie 2.x nano-drone.
Convert a natural language command into a JSON array of flight steps.

SPEED LIMITS (never exceed):
  forward/back : {MAX_FORWARD_SPEED} m/s
  left/right   : {MAX_STRAFE_SPEED} m/s
  up/down      : {MAX_VERTICAL_SPEED} m/s
  yaw rate     : {MAX_YAW_RATE} deg/s

ALLOWED STEP TYPES — output ONLY a valid JSON array, no markdown fences:
[
  {{"action": "takeoff"}},
  {{"action": "land"}},
  {{"action": "move", "vx": <float>, "vy": <float>, "vz": <float>, "yaw": <float>, "duration": <float>}},
  {{"action": "rotate", "degrees": <float>}},
  {{"action": "wait",   "duration": <float>}}
]

AXIS CONVENTIONS (body frame):
  vx > 0  = forward      vx < 0  = backward
  vy > 0  = left         vy < 0  = right
  vz > 0  = up           vz < 0  = down
  yaw > 0 = turn right   (degrees > 0 = clockwise from above)

RULES:
- Never exceed the speed limits above.
- Use durations of 0.5–3 s unless the user specifies a distance or time.
- If a distance is given, estimate duration = distance / appropriate_speed.
- If the drone is already airborne and the user says "go forward", do NOT prepend takeoff.
- If the drone is on the ground and the user says "fly forward", DO prepend takeoff.
- For "spin" / "360" use {{"action": "rotate", "degrees": 360}}.
- Never output anything except the JSON array.
""".strip()


def _build_user_message(command: str, airborne: bool, height_m: float) -> str:
    state = "AIRBORNE" if airborne else "ON THE GROUND"
    return (
        f"Drone state: {state}, current height: {height_m:.2f} m.\n"
        f"Command: {command}"
    )


def _validate_steps(steps: list) -> list:
    """Clamp speeds so a hallucinated value never sends an unsafe setpoint."""
    valid = {"takeoff", "land", "move", "rotate", "wait"}
    out = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if action not in valid:
            continue
        if action == "move":
            step["vx"]       = float(np.clip(step.get("vx", 0),  -MAX_FORWARD_SPEED,  MAX_FORWARD_SPEED))
            step["vy"]       = float(np.clip(step.get("vy", 0),  -MAX_STRAFE_SPEED,   MAX_STRAFE_SPEED))
            step["vz"]       = float(np.clip(step.get("vz", 0),  -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED))
            step["yaw"]      = float(np.clip(step.get("yaw", 0), -MAX_YAW_RATE,       MAX_YAW_RATE))
            step["duration"] = max(0.05, float(step.get("duration", 1.0)))
        elif action == "rotate":
            step["degrees"]  = float(step.get("degrees", 90))
        elif action == "wait":
            step["duration"] = max(0.0, float(step.get("duration", 1.0)))
        out.append(step)
    return out


# ---------------------------------------------------------------------------
# GPT-4o call
# ---------------------------------------------------------------------------
def _call_gpt(command: str, airborne: bool, height_m: float) -> list:
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(command, airborne, height_m)},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    steps = json.loads(raw)
    if not isinstance(steps, list):
        raise ValueError(f"GPT returned non-list: {raw}")
    return _validate_steps(steps)


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------
def _transcribe(audio_bytes: bytes) -> str:
    client = _get_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",
    )
    return transcript.text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit_text_command(command: str, airborne: bool, height_m: float) -> None:
    """Fire-and-forget: calls GPT-4o in a daemon thread, posts to result_queue."""
    def _worker():
        try:
            result_queue.put(("status", "Thinking..."))
            steps = _call_gpt(command, airborne, height_m)
            result_queue.put(("steps", steps))
        except Exception as exc:
            traceback.print_exc()
            result_queue.put(("status", f"Error: {exc}"))

    threading.Thread(target=_worker, daemon=True).start()


def submit_voice_command(audio_pcm: np.ndarray, airborne: bool, height_m: float) -> None:
    """Encode PCM → WAV → Whisper → GPT-4o in a daemon thread."""
    def _worker():
        try:
            result_queue.put(("status", "Transcribing..."))
            buf = io.BytesIO()
            wavfile.write(buf, SAMPLE_RATE, (audio_pcm * 32767).astype(np.int16))
            transcript = _transcribe(buf.getvalue())
            if not transcript:
                result_queue.put(("status", "Voice: nothing heard."))
                return
            result_queue.put(("status", f'Heard: "{transcript}"  Thinking...'))
            steps = _call_gpt(transcript, airborne, height_m)
            result_queue.put(("steps", steps))
        except Exception as exc:
            traceback.print_exc()
            result_queue.put(("status", f"Error: {exc}"))

    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Voice recording helpers
# ---------------------------------------------------------------------------

_recording_buffer: list[np.ndarray] = []
_recording_active  = False
_recording_lock    = threading.Lock()
_stream: sd.InputStream | None = None


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    with _recording_lock:
        if _recording_active:
            _recording_buffer.append(indata[:, 0].copy())  # mono


def open_audio_stream() -> None:
    """Open the mic input stream once at startup."""
    global _stream
    try:
        _stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_audio_callback,
            blocksize=1024,
        )
        _stream.start()
    except Exception as exc:
        print(f"[ai_commands] Warning: could not open audio stream: {exc}")
        print("[ai_commands] Voice input will be unavailable.")


def start_recording() -> None:
    """Begin accumulating mic samples (call on V keydown)."""
    global _recording_active, _recording_buffer
    with _recording_lock:
        _recording_buffer = []
        _recording_active = True


def stop_recording_and_submit(airborne: bool, height_m: float) -> None:
    """Stop accumulating, grab buffer, submit to Whisper + GPT-4o (call on V keyup)."""
    global _recording_active
    with _recording_lock:
        _recording_active = False
        if not _recording_buffer:
            result_queue.put(("status", "Voice: no audio captured."))
            return
        audio = np.concatenate(_recording_buffer)
    submit_voice_command(audio, airborne, height_m)
