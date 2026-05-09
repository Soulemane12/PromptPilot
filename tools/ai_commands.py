import base64
import io
import json
import os
import queue
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import websocket
from openai import OpenAI

MAX_FORWARD_SPEED  = 1.0
MAX_STRAFE_SPEED   = 0.7
MAX_VERTICAL_SPEED = 0.35
MAX_YAW_RATE       = 80.0
MAX_HISTORY        = 8

result_queue: queue.Queue = queue.Queue()

_client: OpenAI | None = None
_history: list = []
_history_lock = threading.Lock()
_action_log: deque = deque(maxlen=20)
_audio_play_queue: queue.Queue = queue.Queue()
_ws = None
_ws_lock = threading.Lock()
_pending_fn_calls: list = []
_rt_running = False
_mic_active  = False
_drone_state = {"airborne": False, "height": 0.0, "battery_pct": 0, "crashed": False}
_latest_frame_fn = lambda: None


def register_frame_getter(fn) -> None:
    global _latest_frame_fn
    _latest_frame_fn = fn


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — add it to your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


_TOOLS = [
    {
        "type": "function",
        "name": "takeoff",
        "description": "Take off from the ground and hover at ~1 metre.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "land",
        "description": "Land the drone on the ground.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "move",
        "description": (
            "Move the drone in body-frame velocity for a given duration. "
            "vx>0=forward, vy>0=left, vz>0=up, yaw>0=turn right (deg/s). "
            f"Speed limits: vx±{MAX_FORWARD_SPEED} m/s, vy±{MAX_STRAFE_SPEED} m/s, "
            f"vz±{MAX_VERTICAL_SPEED} m/s, yaw±{MAX_YAW_RATE} deg/s."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "vx":       {"type": "number", "description": "Forward speed m/s"},
                "vy":       {"type": "number", "description": "Left speed m/s"},
                "vz":       {"type": "number", "description": "Up speed m/s"},
                "yaw":      {"type": "number", "description": "Yaw rate deg/s"},
                "duration": {"type": "number", "description": "Duration in seconds"},
            },
            "required": ["vx", "vy", "vz", "yaw", "duration"],
        },
    },
    {
        "type": "function",
        "name": "rotate",
        "description": "Rotate the drone in place. degrees>0=clockwise, <0=counter-clockwise.",
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {"type": "number", "description": "Degrees to rotate"},
            },
            "required": ["degrees"],
        },
    },
    {
        "type": "function",
        "name": "wait",
        "description": "Hover in place for a given duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Duration in seconds"},
            },
            "required": ["duration"],
        },
    },
    {
        "type": "function",
        "name": "stop",
        "description": "Immediately cancel all pending flight commands and hover in place. Use when told to stop, cancel, abort, or hold.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "describe_scene",
        "description": (
            "Capture and analyze what the drone's camera currently sees. "
            "Use when the user asks what is ahead, if it's safe to proceed, "
            "whether there are obstacles, or what's visible."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "follow_subject",
        "description": (
            "Analyze the camera feed to locate a person or object and generate "
            "movement commands to approach or track them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What to follow, e.g. 'person', 'red object'",
                },
            },
            "required": ["subject"],
        },
    },
    {
        "type": "function",
        "name": "circle",
        "description": (
            "Fly a horizontal circle. The drone moves forward while yawing continuously. "
            "Use this when the user asks to 'fly in a circle', 'do a loop', or 'orbit'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "radius": {
                    "type": "number",
                    "description": "Circle radius in metres (0.3–1.0, default 0.5)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["clockwise", "counterclockwise"],
                    "description": "Direction of rotation (default clockwise)",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "drift",
        "description": (
            "Drift dance — oscillate strafe + yaw side-to-side, like sliding "
            "left and right repeatedly while turning. Use when asked to "
            "'drift', 'sway', 'dance', or 'wiggle side to side'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "Total drift duration in seconds (default 6, range 2–15)",
                },
            },
        },
    },
]

_SYSTEM_PROMPT = f"""
You are the AI pilot of a Crazyflie 2.x nano-drone. You have a friendly, confident personality.
Respond conversationally — short, natural replies. Then call the appropriate flight tool(s).

SPEED LIMITS (never exceed):
  forward/back : {MAX_FORWARD_SPEED} m/s
  left/right   : {MAX_STRAFE_SPEED} m/s
  up/down      : {MAX_VERTICAL_SPEED} m/s
  yaw rate     : {MAX_YAW_RATE} deg/s

RULES:
- Never exceed speed limits.
- Default durations 0.5–3 s unless the user specifies distance or time.
- If distance given: estimate duration = distance / appropriate_speed.
- Already airborne + "go forward" → do NOT call takeoff first.
- On ground + "fly forward" → call takeoff first.
- You have conversation history — use it for follow-ups ("do that again", "go back").
- If the command is chitchat with no flight intent, just reply naturally without calling any tool.
- Keep replies casual and brief.
- You have a forward camera. Use describe_scene when asked what you see, if it's safe, or about obstacles.
- Use follow_subject when asked to follow or track a person or object.
- Use stop when told to stop, cancel, abort, freeze, or hold position.
- Use circle when asked to fly in a circle, do a loop, orbit, or go around. Never simulate a circle with move commands.
- Use drift when asked to drift, sway, dance, or wiggle side-to-side. Never simulate it with move commands.
""".strip()


def _get_memory_context() -> str:
    if not _action_log:
        return ""
    lines = "\n".join(f"  - {ts}: {desc}" for ts, desc in _action_log)
    return f"\nRecent flight history:\n{lines}"


_VISION_ACTIONS  = {"describe_scene", "follow_subject"}
_NON_FLIGHT      = _VISION_ACTIONS | {"stop"}

_tracking_active = False
_tracking_lock   = threading.Lock()


def _encode_frame(frame_bgr) -> str:
    _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(jpeg.tobytes()).decode()


def _vision_query(prompt: str, frame_bgr) -> str:
    client = _get_client()
    b64 = _encode_frame(frame_bgr)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are the AI pilot of a Crazyflie nano-drone analyzing its forward camera feed.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                ],
            },
        ],
        max_tokens=250,
    )
    return response.choices[0].message.content or "I can't make out anything clearly."


def _vision_locate(subject: str, frame_bgr):
    """
    Returns (found, x_offset, distance_cat) where:
      x_offset: -1.0 (far left) to +1.0 (far right), 0 = centered
      distance_cat: 'close' | 'medium' | 'far' | 'none'
    """
    client = _get_client()
    b64 = _encode_frame(frame_bgr)
    prompt = (
        f"Locate the {subject} in this drone camera image. "
        "Reply with ONLY a JSON object, no other text. Example: "
        '{"found": true, "x_offset": 0.3, "distance": "medium"} '
        "x_offset is -1.0 (far left) to 1.0 (far right), 0 is centered. "
        'distance is one of: "close", "medium", "far". '
        '{"found": false} if not visible.'
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                ]},
            ],
            max_tokens=60,
            temperature=0,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        if not data.get("found", False):
            return False, 0.0, "none"
        return True, float(data.get("x_offset", 0.0)), data.get("distance", "medium")
    except Exception as e:
        print(f"[Vision] locate parse error: {e}")
        return False, 0.0, "none"


def stop_tracking() -> None:
    global _tracking_active
    with _tracking_lock:
        _tracking_active = False


def _tracking_loop(subject: str) -> None:
    global _tracking_active
    print(f"[Vision] Tracking '{subject}' — say 'stop' or press V to cancel")
    result_queue.put(("status", f"Tracking {subject}..."))

    CLOSE_THRESHOLD   = 0.5   # m/s forward when far
    MEDIUM_FORWARD    = 0.3
    MAX_YAW           = 40.0  # deg/s yaw to center
    STEP_DURATION     = 0.6   # seconds per move command
    NOT_FOUND_LIMIT   = 4     # consecutive misses before giving up

    not_found_streak = 0

    while True:
        with _tracking_lock:
            if not _tracking_active:
                break

        frame = _latest_frame_fn()
        if frame is None:
            time.sleep(0.5)
            continue

        found, x_offset, dist = _vision_locate(subject, frame)
        print(f"[Vision] found={found}  x={x_offset:.2f}  dist={dist}")

        if not found:
            not_found_streak += 1
            if not_found_streak >= NOT_FOUND_LIMIT:
                print(f"[Vision] Lost {subject} after {NOT_FOUND_LIMIT} misses — stopping")
                result_queue.put(("status", f"Lost {subject}"))
                break
            time.sleep(0.3)
            continue

        not_found_streak = 0

        # Yaw to center the subject horizontally
        yaw = float(np.clip(-x_offset * MAX_YAW, -MAX_YAW, MAX_YAW))

        # Forward speed based on distance
        if dist == "far":
            vx = CLOSE_THRESHOLD
        elif dist == "medium":
            vx = MEDIUM_FORWARD
        else:
            vx = 0.0  # close — just center, don't advance

        step = {"action": "move", "vx": vx, "vy": 0.0, "vz": 0.0,
                "yaw": yaw, "duration": STEP_DURATION}
        result_queue.put(("steps", [step]))

        # Wait for the move to execute before re-evaluating
        time.sleep(STEP_DURATION + 0.15)

    with _tracking_lock:
        _tracking_active = False
    result_queue.put(("status", ""))
    print("[Vision] Tracking ended")


def _handle_vision_calls(ws, calls: list) -> None:
    global _tracking_active

    for call_id, step in calls:
        frame = _latest_frame_fn()
        if frame is None:
            output = "Camera is not connected — I can't see anything right now."
        elif step["action"] == "describe_scene":
            output = _vision_query(
                "Describe what you see from the drone's forward camera. "
                "Be concise. Note any people, obstacles, furniture, or open space.",
                frame,
            )
        elif step["action"] == "follow_subject":
            subject = step.get("subject", "person")
            with _tracking_lock:
                if _tracking_active:
                    output = f"Already tracking. Say 'stop' first."
                else:
                    _tracking_active = True
                    threading.Thread(
                        target=_tracking_loop, args=(subject,), daemon=True
                    ).start()
                    output = f"On it — tracking the {subject} now."
        else:
            output = "Unknown vision action."

        try:
            ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }))
            print(f"[Vision] {step['action']}: {output[:120]}")
        except Exception as e:
            print(f"[Vision] send error: {e}")

    try:
        ws.send(json.dumps({"type": "response.create"}))
    except Exception:
        pass


def _validate_steps(steps: list) -> list:
    valid = {"takeoff", "land", "move", "rotate", "wait", "circle", "drift"}
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
        elif action == "circle":
            step["radius"]    = float(np.clip(step.get("radius", 0.5), 0.3, 1.0))
            step["direction"] = step.get("direction", "clockwise")
        elif action == "drift":
            step["duration"] = float(np.clip(step.get("duration", 6.0), 2.0, 15.0))
        out.append(step)
    return out


def _call_gpt(command: str, airborne: bool, height_m: float) -> tuple[str, list]:
    client = _get_client()
    state = "AIRBORNE" if airborne else "ON THE GROUND"
    user_msg = f"Drone state: {state}, current height: {height_m:.2f} m.\nCommand: {command}"

    system = _SYSTEM_PROMPT + _get_memory_context()

    with _history_lock:
        messages = (
            [{"role": "system", "content": system}]
            + _history[-MAX_HISTORY:]
            + [{"role": "user", "content": user_msg}]
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=_TOOLS,
        tool_choice="auto",
        temperature=0.4,
        max_tokens=600,
    )

    msg = response.choices[0].message
    reply = msg.content or ""
    steps = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            step = {"action": tc.function.name, **json.loads(tc.function.arguments)}
            steps.append(step)

    vision_steps  = [s for s in steps if s["action"] in _VISION_ACTIONS]
    flight_steps  = [s for s in steps if s["action"] not in _NON_FLIGHT]
    if any(s["action"] == "stop" for s in steps):
        result_queue.put(("stop", None))

    if vision_steps:
        frame = _latest_frame_fn()
        for vs in vision_steps:
            if vs["action"] == "describe_scene":
                reply = _vision_query(
                    "Describe what the drone's camera sees. Be concise.",
                    frame,
                ) if frame is not None else "Camera is not connected."
            elif vs["action"] == "follow_subject":
                subject = vs.get("subject", "person")
                reply = _vision_query(
                    f"Locate the {subject} and describe their position and how to approach them.",
                    frame,
                ) if frame is not None else "Camera is not connected."

    with _history_lock:
        _history.append({"role": "user",      "content": user_msg})
        _history.append({"role": "assistant",  "content": reply or json.dumps(flight_steps)})
        while len(_history) > MAX_HISTORY:
            _history.pop(0)

    return reply, _validate_steps(flight_steps)


def add_to_action_log(action_str: str) -> None:
    ts = time.strftime("%H:%M:%S")
    _action_log.append((ts, action_str))


def update_drone_state(airborne: bool, height: float, battery_pct: int = 0, crashed: bool = False) -> None:
    _drone_state["airborne"]    = airborne
    _drone_state["height"]      = height
    _drone_state["battery_pct"] = battery_pct
    _drone_state["crashed"]     = crashed


def _get_current_state_str() -> str:
    state   = "AIRBORNE" if _drone_state["airborne"] else "ON THE GROUND"
    h       = _drone_state["height"]
    batt    = _drone_state["battery_pct"]
    crashed = _drone_state["crashed"]
    s = f"[Drone state: {state}, height={h:.2f}m, battery={batt}%"
    if crashed:
        s += ", CRASHED/LOCKED"
    if _action_log:
        recent = ", ".join(desc for _, desc in list(_action_log)[-3:])
        s += f", recent actions: {recent}"
    s += "]"
    return s


def notify_event(event_str: str) -> None:
    add_to_action_log(event_str)
    with _ws_lock:
        ws = _ws
    if ws is not None:
        try:
            ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"[SYSTEM EVENT] {event_str}"}],
                },
            }))
        except Exception as e:
            print(f"[RT] notify_event error: {e}")


def set_mic_active(active: bool) -> None:
    global _mic_active
    with _ws_lock:
        ws = _ws
    if active:
        _mic_active = True
        if ws:
            try:
                ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except Exception:
                pass
    else:
        _mic_active = False
        if ws:
            try:
                ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": _get_current_state_str()}],
                    },
                }))
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                ws.send(json.dumps({"type": "response.create"}))
            except Exception:
                pass


def clear_history() -> None:
    with _history_lock:
        _history.clear()
    _action_log.clear()


def speak_async(text: str) -> None:
    def _worker():
        try:
            resp = _get_client().audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm",
            )
            _audio_play_queue.put(resp.content)
        except Exception as e:
            print(f"[TTS] {e}")
    threading.Thread(target=_worker, daemon=True).start()


def submit_text_command(command: str, airborne: bool, height_m: float) -> None:
    with _ws_lock:
        ws_live = _ws

    if ws_live is not None:
        state   = "AIRBORNE" if airborne else "ON THE GROUND"
        content = f"Drone state: {state}, current height: {height_m:.2f} m.\nCommand: {command}"
        try:
            ws_live.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                },
            }))
            ws_live.send(json.dumps({"type": "response.create"}))
            result_queue.put(("heard", command))
        except Exception as e:
            result_queue.put(("status", f"RT send error: {e}"))
        return

    def _worker():
        try:
            result_queue.put(("status", "Thinking..."))
            reply, steps = _call_gpt(command, airborne, height_m)
            if reply:
                result_queue.put(("reply", reply))
            if steps:
                result_queue.put(("steps", steps))
            elif not reply:
                result_queue.put(("status", "No steps returned."))
        except Exception as exc:
            traceback.print_exc()
            result_queue.put(("status", f"Error: {exc}"))
    threading.Thread(target=_worker, daemon=True).start()


def _send_session_config(ws) -> None:
    ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "instructions": _SYSTEM_PROMPT + _get_memory_context(),
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1", "language": "en"},
            "turn_detection": None,
            "tools": _TOOLS,
            "tool_choice": "auto",
        },
    }))


def _on_open(ws) -> None:
    global _ws
    with _ws_lock:
        _ws = ws
    print("[RT] Connected to Realtime API")
    result_queue.put(("status", "Realtime connected — just speak!"))


def _on_message(ws, raw: str) -> None:
    try:
        ev = json.loads(raw)
        t  = ev.get("type", "")

        if t == "session.created":
            _send_session_config(ws)

        elif t == "conversation.item.input_audio_transcription.completed":
            tr = ev.get("transcript", "").strip()
            if tr and tr.isascii():
                result_queue.put(("heard", tr))
                print(f"[RT] Heard: {tr}")

        elif t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            tr = ev.get("transcript", "").strip()
            if tr:
                result_queue.put(("reply", tr))
                print(f"[RT] Drone: {tr}")

        elif t == "response.function_call_arguments.done":
            name = ev.get("name", "")
            args = json.loads(ev.get("arguments", "{}"))
            step = {"action": name, **args}
            _pending_fn_calls.append((ev.get("call_id", ""), step))

        elif t == "response.done":
            if _pending_fn_calls:
                stop_calls   = [(cid, s) for cid, s in _pending_fn_calls if s["action"] == "stop"]
                flight_calls = [(cid, s) for cid, s in _pending_fn_calls if s["action"] not in _NON_FLIGHT]
                vision_calls = [(cid, s) for cid, s in _pending_fn_calls if s["action"] in _VISION_ACTIONS]

                if stop_calls:
                    result_queue.put(("stop", None))
                    for call_id, _ in stop_calls:
                        ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": call_id, "output": "stopped"},
                        }))
                    if not vision_calls:
                        ws.send(json.dumps({"type": "response.create"}))

                if flight_calls:
                    steps = _validate_steps([s for _, s in flight_calls])
                    if steps:
                        result_queue.put(("steps", steps))
                        print(f"[RT] Dispatching {len(steps)} flight step(s)")
                    for call_id, _ in flight_calls:
                        ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": "executed",
                            },
                        }))
                    if not vision_calls:
                        ws.send(json.dumps({"type": "response.create"}))

                if vision_calls:
                    threading.Thread(
                        target=_handle_vision_calls,
                        args=(ws, vision_calls),
                        daemon=True,
                    ).start()

                _pending_fn_calls.clear()

        elif t in ("response.audio.delta", "response.output_audio.delta"):
            data = ev.get("delta", "")
            if data:
                _audio_play_queue.put(base64.b64decode(data))

        elif t == "error":
            msg = ev.get("error", {}).get("message", str(ev))
            print(f"[RT] API error: {msg}")
            result_queue.put(("status", f"RT Error: {msg}"))

        else:
            _NOISY = {
                "rate_limits.updated", "response.created", "response.audio_transcript.delta",
                "response.content_part.added", "response.content_part.done",
                "response.output_item.added", "response.output_item.done",
                "conversation.item.created", "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
                "input_audio_buffer.cleared", "response.audio_transcript.done",
                "response.function_call_arguments.delta", "response.audio.done",
                "conversation.item.input_audio_transcription.delta",
                "response.output_audio_transcript.delta",
            }
            if t not in _NOISY:
                print(f"[RT] unhandled: {t}")

    except Exception as e:
        print(f"[RT] _on_message exception: {e}")


def _rt_ws_thread() -> None:
    global _ws
    while _rt_running:
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            app = websocket.WebSocketApp(
                "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
                header={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta":   "realtime=v1",
                },
                on_open=_on_open,
                on_message=_on_message,
                on_error=lambda ws, e: print(f"[RT] WS error: {e}"),
                on_close=lambda ws, c, m: _on_ws_close(),
            )
            app.run_forever()
        except Exception as e:
            print(f"[RT] Connection failed: {e}")
        if _rt_running:
            print("[RT] Reconnecting in 3 s...")
            time.sleep(3)


def _on_ws_close() -> None:
    global _ws
    with _ws_lock:
        _ws = None
    print("[RT] Disconnected")
    result_queue.put(("status", "Realtime disconnected — reconnecting..."))


def _rt_mic_thread() -> None:
    def cb(indata, frames, t, status):
        if not _mic_active:
            return
        with _ws_lock:
            if _ws is not None:
                try:
                    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                    b64 = base64.b64encode(pcm).decode()
                    _ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                except Exception:
                    pass

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            callback=cb, blocksize=1600):
            while _rt_running:
                time.sleep(0.1)
    except Exception as e:
        print(f"[RT] Mic error: {e}")


def _rt_playback_thread() -> None:
    try:
        with sd.OutputStream(samplerate=24000, channels=1, dtype="int16") as stream:
            while _rt_running:
                try:
                    pcm = _audio_play_queue.get(timeout=0.1)
                    arr = np.frombuffer(pcm, dtype=np.int16)
                    stream.write(arr.reshape(-1, 1))
                except queue.Empty:
                    pass
    except Exception as e:
        print(f"[RT] Playback error: {e}")


def start_realtime_session() -> None:
    global _rt_running
    _rt_running = True
    threading.Thread(target=_rt_ws_thread,      daemon=True).start()
    threading.Thread(target=_rt_mic_thread,      daemon=True).start()
    threading.Thread(target=_rt_playback_thread, daemon=True).start()
    print("[RT] Starting Realtime session (connecting...)")
