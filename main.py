import time
import threading
import socket
import struct
import math
import os
import sys
import queue
from contextlib import contextmanager
import pygame
import numpy as np
import cv2
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from dotenv import load_dotenv

load_dotenv()

from tools.ai_commands import (
    result_queue as ai_result_queue,
    submit_text_command,
    start_realtime_session,
    register_frame_getter,
    add_to_action_log,
    speak_async,
    update_drone_state,
    notify_event,
    set_mic_active,
    stop_tracking,
    clear_history,
)

DEFAULT_ADDR = "E7E7E7E7E7"
DEFAULT_URI = f"radio://0/80/2M/{DEFAULT_ADDR}"
DEADZONE = 0.12
HOVER_HEIGHT = 1.3
DECK_IP = "192.168.4.1"
DECK_PORT = 5000
MAX_FORWARD_SPEED = 1.0
MAX_STRAFE_SPEED = 0.7
MAX_VERTICAL_SPEED = 0.35
MAX_YAW_RATE = 80.0
MAX_WORLD_HEIGHT = 1.6
MIN_WORLD_HEIGHT = 0.2
MAX_HOVER_ZDISTANCE = 2.2
MIN_HOVER_ZDISTANCE = 0.15
ALTITUDE_RESPONSE = 0.18

def apply_deadzone(value):
    if abs(value) < DEADZONE:
        return 0.0
    if value > 0:
        return (value - DEADZONE) / (1.0 - DEADZONE)
    return (value + DEADZONE) / (1.0 - DEADZONE)

def scan_uri():
    return scan_candidate_uris()[0]

def normalize_radio_uri(uri):
    uri = (uri or "").strip()
    if not uri.startswith("radio://"):
        return None
    if uri.count("/") == 4:
        return f"{uri}/{DEFAULT_ADDR}"
    if uri.count("/") >= 5:
        return uri
    return None

def uri_priority(uri):
    return (
        0 if uri == DEFAULT_URI else 1,
        0 if "/80/" in uri else 1,
        0 if "/2M/" in uri else 1,
        uri,
    )

def scan_candidate_uris(timeout_s=5.0):
    candidates = []

    env_uri = normalize_radio_uri(os.environ.get("CF_URI", ""))
    if env_uri:
        candidates.append(env_uri)
    candidates.append(DEFAULT_URI)

    deadline = time.time() + timeout_s
    discovered = set()
    while time.time() < deadline:
        found = cflib.crtp.scan_interfaces()
        for item in found:
            raw_uri = item[0] if isinstance(item, (tuple, list)) else item
            uri = normalize_radio_uri(raw_uri)
            if uri:
                discovered.add(uri)
        if discovered:
            break
        time.sleep(0.2)

    candidates.extend(sorted(discovered, key=uri_priority))

    unique_candidates = []
    seen = set()
    for uri in candidates:
        if uri and uri not in seen:
            unique_candidates.append(uri)
            seen.add(uri)

    if not unique_candidates:
        raise RuntimeError("No Crazyflie found. Power it on and plug in Crazyradio.")
    return unique_candidates

@contextmanager
def connect_crazyflie(rw_cache="./cache"):
    candidate_uris = scan_candidate_uris(timeout_s=8.0)
    print("Radio candidates:", ", ".join(candidate_uris))

    scf = None
    last_error = None
    for uri in candidate_uris:
        print(f"Trying link: {uri}")
        attempt = SyncCrazyflie(uri, cf=Crazyflie(rw_cache=rw_cache))
        try:
            attempt.open_link()
            scf = attempt
            print(f"Drone connected at: {uri}")
            break
        except Exception as exc:
            last_error = exc
            print(f"Link failed for {uri}: {exc}")
            try:
                attempt.close_link()
            except Exception:
                pass
            time.sleep(0.6)

    if scf is None:
        raise RuntimeError(f"Unable to connect to Crazyflie. Last error: {last_error}")

    try:
        yield scf
    finally:
        try:
            scf.close_link()
        except Exception:
            pass

latest_frame = None
camera_connected = False


def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Camera disconnected")
        data.extend(chunk)
    return data

def camera_thread():
    global latest_frame, camera_connected
    attempt = 0
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            attempt += 1
            sock.settimeout(5)
            sock.connect((DECK_IP, DECK_PORT))
            sock.settimeout(60)
            camera_connected = True
            attempt = 0
            print("Camera connected! Waiting for frames...")
            while True:
                packetInfoRaw = rx_bytes(sock, 4)
                [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
                imgHeader = rx_bytes(sock, length - 2)
                [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', imgHeader)
                if magic == 0xBC:
                    imgStream = bytearray()
                    while len(imgStream) < size:
                        packetInfoRaw = rx_bytes(sock, 4)
                        [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                        chunk = rx_bytes(sock, length - 2)
                        imgStream.extend(chunk)
                    if fmt == 0:
                        bayer = np.frombuffer(imgStream, dtype=np.uint8)
                        bayer.shape = (244, 324)
                        frame = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
                    else:
                        nparr = np.frombuffer(imgStream, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        latest_frame = cv2.resize(frame, (860, 300), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            camera_connected = False
            print(f"Camera reconnecting... ({e}) [attempt {attempt}]")
        finally:
            try:
                sock.close()
            except Exception:
                pass
        time.sleep(3)

cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()

pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((860, 720))
pygame.display.set_caption("PromptPilot — click here first!")
font = pygame.font.SysFont("monospace", 16)

def estimate_time_remaining():
    now = time.time()

    if airborne:
        window = [(t, p) for t, p in battery_history if now - t <= 45]
    else:
        window = [(t, p) for t, p in battery_history if now - t <= 120]

    if len(window) < 2:
        return None, None

    airborne_samples = 0
    landed_samples = 0
    total_drain = 0
    total_time = 0

    for i in range(1, len(window)):
        dt = window[i][0] - window[i-1][0]
        dp = window[i-1][1] - window[i][1]

        if dt > 0 and dp >= 0:
            sample_was_airborne = i >= len(window) // 2 if airborne else False

            if sample_was_airborne:
                airborne_samples += 1
                weight = 1.5
            else:
                landed_samples += 1
                weight = 1.0

            total_drain += dp * weight
            total_time += dt * weight

    if total_time <= 0 or total_drain <= 0:
        return None, None

    drain_per_min = total_drain / (total_time / 60)

    if airborne:
        drain_per_min *= 1.2
    else:
        drain_per_min *= 0.3

    mins_left = battery_pct / drain_per_min if drain_per_min > 0 else None
    return round(drain_per_min, 1), round(mins_left, 1) if mins_left else None

def draw_ui():
    screen.fill((20, 20, 20))

    if latest_frame is not None:
        display_frame = latest_frame.copy()
        lab = cv2.cvtColor(display_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(l)
        display_frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        display_frame = cv2.filter2D(display_frame, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
    else:
        status = "Connecting to camera..." if not camera_connected else "No frame yet"
        pygame.draw.rect(screen, (40, 40, 40), (0, 0, 860, 300))
        msg = font.render(status, True, (150, 150, 150))
        screen.blit(msg, (330, 140))

    pct = max(0, min(100, battery_pct))
    bar_color = (0, 220, 80) if pct > 40 else (255, 180, 0) if pct > 20 else (220, 50, 50)
    pygame.draw.rect(screen, (60, 60, 60), (15, 306, 830, 16), border_radius=4)
    pygame.draw.rect(screen, bar_color, (15, 306, int(830 * pct / 100), 16), border_radius=4)
    disp_volt = smoothed_ground_volt if (airborne and smoothed_ground_volt > 0) else battery_volt
    batt_text = f"Battery: {pct}%  ({disp_volt:.2f}V)"
    screen.blit(font.render(batt_text, True, (255, 255, 255)), (250, 307))

    drain, mins_left = estimate_time_remaining()
    if mins_left is not None:
        time_color = (0, 220, 80) if mins_left > 3 else (255, 180, 0) if mins_left > 1.5 else (220, 50, 50)
        time_text = f"~{mins_left} min left  (-{drain}%/min)"
    else:
        time_color = (150, 150, 150)
        time_text = "Calculating flight time..."

    if altitude_telemetry_ready():
        alt_text  = f"z={state_z_m:.2f}m  range={range_z_m:.2f}m"
        alt_color = (120, 220, 255)
    else:
        alt_text  = "waiting for telemetry..."
        alt_color = (255, 180, 0)

    screen.blit(font.render(time_text, True, time_color), (15, 326))
    screen.blit(font.render(alt_text,  True, alt_color),  (470, 326))

    is_armed   = bool(supervisor_info & (1 << 1))
    can_fly    = bool(supervisor_info & (1 << 3))
    is_flying  = bool(supervisor_info & (1 << 4))
    is_tumbled = bool(supervisor_info & (1 << 5))
    is_locked  = bool(supervisor_info & (1 << 6))

    if is_tumbled or is_locked:
        badge_text  = "CRASHED / LOCKED — Press R to recover"
        badge_color = (220, 50, 50)
    elif is_flying:
        badge_text  = "● FLYING"
        badge_color = (0, 220, 80)
    elif can_fly:
        badge_text  = "● READY  (armed, pre-flight OK)"
        badge_color = (0, 200, 255)
    elif is_armed:
        badge_text  = "● ARMED  (waiting for pre-flight)"
        badge_color = (255, 180, 0)
    else:
        badge_text  = "○ DISARMED"
        badge_color = (130, 130, 130)

    screen.blit(font.render(badge_text, True, badge_color), (15, 346))

    pygame.draw.line(screen, (55, 55, 55), (15, 364), (845, 364))

    if is_ps:
        ctrl_hdr    = "PLAYSTATION CONTROLS"
        takeoff_btn = "Triangle / Y  Take off"
        land_btn    = "Cross / A     Land"
        ud_btn      = "R2 / L2       Up / Down"
        recover_btn = "Options       Recover"
        talk_btn    = "Square (hold) Voice"
    else:
        ctrl_hdr    = "XBOX CONTROLS"
        takeoff_btn = "Y button      Take off"
        land_btn    = "A button      Land"
        ud_btn      = "RT / LT       Up / Down"
        recover_btn = "Menu          Recover"
        talk_btn    = "X (hold)      Voice"

    left_col = [
        (ctrl_hdr,    (0, 200, 255)),
        ("L.Stick   Move fwd/back/L/R", (200, 200, 200)),
        ("R.Stick X  Yaw",              (200, 200, 200)),
        (ud_btn,                        (200, 200, 200)),
        (takeoff_btn,                   (200, 200, 200)),
        (land_btn,                      (200, 200, 200)),
        (recover_btn,                   (255, 110, 110)),
        (talk_btn,                      (100, 220, 255)),
    ]
    right_col = [
        ("KEYBOARD",           (0, 200, 255)),
        ("W / S   Fwd / Back", (200, 200, 200)),
        ("A / D   Left / Right",(200, 200, 200)),
        ("Q / E   Yaw",        (200, 200, 200)),
        ("Up/Dn   Up / Down",  (200, 200, 200)),
        ("T       Take off",   (200, 200, 200)),
        ("L       Land",       (200, 200, 200)),
        ("R       Recover",    (255, 110, 110)),
        ("V hold  Voice",      (100, 220, 255)),
        ("/ Enter AI command", (100, 220, 255)),
        ("ESC     Quit",       (200, 200, 200)),
    ]

    CX, KX, CY, SPC = 15, 450, 372, 19
    for i, (text, color) in enumerate(left_col):
        screen.blit(font.render(text, True, color), (CX, CY + i * SPC))
    for i, (text, color) in enumerate(right_col):
        screen.blit(font.render(text, True, color), (KX, CY + i * SPC))

    pygame.draw.line(screen, (55, 55, 55), (15, 582), (845, 582))

    y_ai = 590
    if ai_heard:
        screen.blit(font.render(f'You: "{ai_heard[:82]}"', True, (160, 160, 160)), (15, y_ai))
        y_ai += 20
    if ai_reply:
        words = ai_reply.split()
        line1, line2 = [], []
        for w in words:
            if len(" ".join(line1 + [w])) <= 84:
                line1.append(w)
            else:
                line2.append(w)
        screen.blit(font.render("Drone: " + " ".join(line1), True, (100, 220, 255)), (15, y_ai))
        y_ai += 20
        if line2:
            screen.blit(font.render("       " + " ".join(line2), True, (100, 220, 255)), (15, y_ai))
            y_ai += 20
    if ai_status:
        status_color = (255, 80, 80) if ai_status.startswith("Error") else (180, 180, 180)
        screen.blit(font.render(ai_status, True, status_color), (15, y_ai))

    box_color = (70, 130, 200) if text_input_active else (45, 45, 45)
    pygame.draw.rect(screen, box_color, (15, 692, 830, 22), border_radius=3)
    if text_input_active:
        display_text = "/" + text_input_buffer + "|"
    elif ai_voice_recording:
        display_text = "● Listening..."
    else:
        display_text = "Press / to type  |  Hold V / Square to speak"
    screen.blit(font.render(display_text, True, (255, 255, 255)), (18, 695))

    pygame.draw.rect(screen, (60, 40, 40), (790, 276, 60, 20), border_radius=3)
    pygame.draw.rect(screen, (120, 60, 60), (790, 276, 60, 20), border_radius=3, width=1)
    screen.blit(font.render("RESTART", True, (220, 120, 120)), (795, 278))

    pygame.display.flip()

RESTART_BTN_RECT = pygame.Rect(790, 276, 60, 20)

controller = None
is_ps = False
if pygame.joystick.get_count() > 0:
    controller = pygame.joystick.Joystick(0)
    controller.init()
    name = controller.get_name().lower()
    is_ps = "playstation" in name or "dualshock" in name or "dualsense" in name or "ps4" in name or "ps5" in name
    print(f"Controller: {controller.get_name()} ({'PlayStation' if is_ps else 'Xbox'})")
else:
    print("No controller detected. Keyboard-only mode.")

start_realtime_session()
register_frame_getter(lambda: latest_frame)

cflib.crtp.init_drivers(enable_debug_driver=False)
print(f"Preferred radio URI: {DEFAULT_URI}")

left_x = left_y = right_x = 0.0
left_trigger = right_trigger = -1.0
running = True
_restart_requested = False
airborne = False
height = HOVER_HEIGHT
smoothed_height = HOVER_HEIGHT
keys_held = set()
battery_pct = 0
battery_volt = 0.0
battery_history = []
voltage_samples = []
smoothed_voltage = 0.0
ground_voltage_samples: list = []
smoothed_ground_volt: float = 0.0
heading_rad = 0.0
state_z_m = 0.0
state_x_m: float = 0.0
state_y_m: float = 0.0
range_z_m = None
supervisor_info: int = 0
hold_x: float | None = None
hold_y: float | None = None
hold_yaw_deg: float = 0.0

ai_steps: list = []
ai_step_index: int = 0
ai_step_start: float = 0.0
ai_status: str = ""
ai_reply: str = ""
ai_heard: str = ""
ai_voice_recording: bool = False
last_battery_warn: float = 0.0
_prev_crashed: bool = False
_prev_camera_connected: bool = False
_recovering: bool = False

text_input_active: bool = False
text_input_buffer: str = ""

def clamp(value, low, high):
    return max(low, min(high, value))

def altitude_telemetry_ready():
    return range_z_m is not None

def hover_distance_for_world_height(target_world_z):
    if not altitude_telemetry_ready():
        return clamp(target_world_z, MIN_HOVER_ZDISTANCE, MAX_HOVER_ZDISTANCE)

    compensated = range_z_m + (target_world_z - state_z_m)
    return clamp(compensated, MIN_HOVER_ZDISTANCE, MAX_HOVER_ZDISTANCE)

def send_world_hover_setpoint(cf, vx, vy, yawrate, target_world_z):
    hover_distance = hover_distance_for_world_height(target_world_z)
    cf.commander.send_hover_setpoint(vx, vy, yawrate, hover_distance)
    return hover_distance

def current_takeoff_target():
    if altitude_telemetry_ready():
        return clamp(state_z_m + HOVER_HEIGHT, MIN_WORLD_HEIGHT, MAX_WORLD_HEIGHT)
    return HOVER_HEIGHT

def body_to_world_velocity(vx_body, vy_body, yaw_rad):
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    vx_world = (vx_body * cos_yaw) - (vy_body * sin_yaw)
    vy_world = (vx_body * sin_yaw) + (vy_body * cos_yaw)
    return vx_world, vy_world

def do_takeoff(cf):
    global left_x, left_y, right_x, left_trigger, right_trigger

    # Reset the Kalman estimator and let it converge while stationary on the
    # ground BEFORE motors spool up. Without this, IMU / optical-flow bias
    # accumulated since the last reset gets "corrected" the moment thrust
    # starts — and that correction comes out as horizontal drift (almost
    # always forward, because takeoff thrust biases the pitch axis).
    print("Calibrating estimator...")
    try:
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
    except Exception as e:
        print(f"  (estimator reset skipped: {e})")
    # Hold still ~1.5 s while Kalman converges.
    for _ in range(30):
        pygame.event.pump()
        draw_ui()
        time.sleep(0.05)

    print("Taking off...")
    base_world_z = state_z_m if altitude_telemetry_ready() else 0.0
    target_world_z = current_takeoff_target()
    # Pre-arm hover commands at ground level so the firmware switches into
    # commander mode while still on the ground, with vx=vy=0 already locked.
    for _ in range(5):
        send_world_hover_setpoint(cf, 0, 0, 0, base_world_z)
        time.sleep(0.05)
    h = base_world_z
    CLIMB_STEPS = 40
    for _ in range(CLIMB_STEPS):
        h += (target_world_z - base_world_z) / CLIMB_STEPS
        send_world_hover_setpoint(cf, 0, 0, 0, h)
        pygame.event.pump()
        draw_ui()
        time.sleep(0.04)

    # Discard joystick / key events that piled up while we were blocking,
    # so stale JOYAXISMOTION values from during takeoff cannot overwrite
    # the stick globals on the next main-loop frame.
    pygame.event.clear()
    keys_held.clear()

    # Re-snapshot live stick state from the hardware so the next velocity
    # command reflects the stick *now*, not whatever happened during the
    # ~3s blocking takeoff.
    left_x = left_y = right_x = 0.0
    left_trigger = right_trigger = -1.0
    if controller is not None:
        try:
            left_x        = apply_deadzone(controller.get_axis(0))
            left_y        = apply_deadzone(controller.get_axis(1))
            right_x       = apply_deadzone(controller.get_axis(2))
            left_trigger  = controller.get_axis(4)
            right_trigger = controller.get_axis(5)
        except Exception:
            pass

    print("Airborne!")
    return target_world_z

def do_land(cf, current_height):
    print("Landing...")
    h = current_height
    steps = max(1, int(h / 0.015))
    for _ in range(steps):
        h = max(0.0, h - 0.015)
        send_world_hover_setpoint(cf, 0, 0, 0, h)
        pygame.event.pump()
        draw_ui()
        time.sleep(0.02)
    cf.commander.send_stop_setpoint()
    print("Landed.")


def _is_manual_input_active() -> bool:
    keys = pygame.key.get_pressed()
    manual_keys = (
        keys[pygame.K_w] or keys[pygame.K_s] or
        keys[pygame.K_a] or keys[pygame.K_d] or
        keys[pygame.K_UP] or keys[pygame.K_DOWN] or
        keys[pygame.K_q] or keys[pygame.K_e]
    )
    manual_stick = (
        abs(left_x) > DEADZONE or abs(left_y) > DEADZONE or
        abs(right_x) > DEADZONE or
        abs((right_trigger + 1.0) / 2.0) > DEADZONE or
        abs((left_trigger + 1.0) / 2.0) > DEADZONE
    )
    return bool(manual_keys or manual_stick)


def _advance_step(total: int) -> None:
    global ai_step_index, ai_step_start, ai_status
    ai_step_index += 1
    ai_step_start = time.time()
    ai_status = f"Step {ai_step_index + 1}/{total}" if ai_step_index < total else ""


def tick_ai_executor(cf) -> bool:
    global ai_steps, ai_step_index, ai_step_start, ai_status, ai_reply, ai_heard, airborne, height, smoothed_height

    while True:
        try:
            kind, payload = ai_result_queue.get_nowait()
        except queue.Empty:
            break
        if kind == "status":
            ai_status = payload
        elif kind == "reply":
            ai_reply = payload
            ai_status = ""
            print(f"[AI] Drone: {payload}")
        elif kind == "heard":
            ai_heard = payload
            ai_status = ""
            print(f"[AI] Heard: {payload}")
        elif kind == "steps":
            was_empty = len(ai_steps) == 0 or ai_step_index >= len(ai_steps)
            ai_steps.extend(payload)
            if was_empty:
                ai_step_index = 0
                ai_step_start = time.time()
            total = len(ai_steps)
            ai_status = f"Step {ai_step_index + 1}/{total}" if total else ""
            print(f"[AI] Queued {len(payload)} step(s) — total {total}:")
            for idx, s in enumerate(payload):
                print(f"  +{idx+1}. {s}")
        elif kind == "stop":
            print("[AI] Stop command — cancelling sequence")
            stop_tracking()
            speak_async("Stopping.")
            ai_steps = []
            ai_step_index = 0
            ai_step_start = 0.0
            ai_status = ""

    if not ai_steps:
        return False

    if _is_manual_input_active():
        print("[AI] Sequence cancelled — manual override")
        stop_tracking()
        speak_async("Manual override")
        ai_steps = []
        ai_step_index = 0
        ai_status = ""
        ai_reply = "Got it, taking manual control."
        return False

    if ai_step_index >= len(ai_steps):
        print("[AI] Sequence complete")
        ai_steps = []
        ai_status = "Done."
        return False

    step    = ai_steps[ai_step_index]
    now     = time.time()
    elapsed = now - ai_step_start
    action  = step["action"]
    total   = len(ai_steps)

    if not step.get("_printed"):
        step["_printed"] = True
        idx = ai_step_index + 1
        if action == "move":
            vx_b = step.get("vx", 0.0)
            vy_b = step.get("vy", 0.0)
            vz_b = step.get("vz", 0.0)
            dur  = step.get("duration", 1.0)
            print(f"[AI] Step {idx}/{total}: move  vx={vx_b:.2f}  vy={vy_b:.2f}  vz={vz_b:.2f}  yaw={step.get('yaw',0):.1f}  dur={dur:.1f}s")
            if   abs(vx_b) >= abs(vy_b) and abs(vx_b) >= abs(vz_b):
                direction = "forward" if vx_b > 0 else "backward"
            elif abs(vy_b) >= abs(vz_b):
                direction = "left"    if vy_b > 0 else "right"
            else:
                direction = "up"      if vz_b > 0 else "down"
            speak_async(f"Moving {direction}")
            add_to_action_log(f"move vx={vx_b:.1f} vy={vy_b:.1f} vz={vz_b:.1f} {dur:.1f}s")
        elif action == "rotate":
            deg = step.get("degrees", 90)
            print(f"[AI] Step {idx}/{total}: rotate {deg:.0f}°")
            speak_async(f"Rotating {'right' if deg >= 0 else 'left'}")
            add_to_action_log(f"rotate {deg:.0f} degrees")
        elif action == "wait":
            dur = step.get("duration", 1.0)
            print(f"[AI] Step {idx}/{total}: wait {dur:.1f}s")
            add_to_action_log(f"wait {dur:.1f}s")
        elif action == "takeoff":
            print(f"[AI] Step {idx}/{total}: takeoff")
            speak_async("Taking off")
            add_to_action_log("takeoff")
        elif action == "land":
            print(f"[AI] Step {idx}/{total}: land")
            speak_async("Landing")
            add_to_action_log("land")
        elif action == "circle":
            r   = step.get("radius", 0.5)
            cw  = step.get("direction", "clockwise")
            print(f"[AI] Step {idx}/{total}: circle  radius={r:.2f}m  {cw}")
            speak_async("Flying in a circle")
            add_to_action_log(f"circle r={r:.2f}m {cw}")
        elif action == "drift":
            dur = step.get("duration", 6.0)
            print(f"[AI] Step {idx}/{total}: drift  duration={dur:.1f}s")
            speak_async("Drifting")
            add_to_action_log(f"drift {dur:.1f}s")

    if action == "takeoff":
        if not airborne:
            height = do_takeoff(cf)
            airborne = True
            smoothed_height = height
            cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
        _advance_step(total)

    elif action == "land":
        if airborne:
            do_land(cf, height)
            airborne = False
        _advance_step(total)

    elif action == "wait":
        if airborne:
            send_world_hover_setpoint(cf, 0, 0, 0, height)
        if elapsed >= step.get("duration", 1.0):
            _advance_step(total)

    elif action == "move":
        vx_body = step.get("vx", 0.0)
        vy_body = step.get("vy", 0.0)
        vz      = step.get("vz", 0.0)
        yaw     = step.get("yaw", 0.0)
        if airborne:
            vx_world, vy_world = body_to_world_velocity(vx_body, vy_body, heading_rad)
            cf.commander.send_velocity_world_setpoint(vx_world, vy_world, vz, yaw)
        if elapsed >= step.get("duration", 1.0):
            _advance_step(total)

    elif action == "rotate":
        # Chunked rotation: yaw a small arc, then hover-settle, then yaw again.
        # The pauses let the Flow-deck position-hold null out any drift the
        # yaw motion induced before it can integrate into a noticeable circle.
        degrees_target = abs(step.get("degrees", 90))
        direction = 1.0 if step.get("degrees", 90) >= 0 else -1.0

        CHUNK_DEG = 45.0    # yaw this much per chunk
        YAW_RATE  = 45.0    # deg/s — moderate, flow stays clean
        PAUSE_S   = 0.30    # hover-settle between chunks

        if "degrees_done" not in step:
            step["degrees_done"]   = 0.0
            step["chunk_done"]     = 0.0
            step["phase"]          = "yaw"
            step["_last_tick"]     = now
            step["pause_start"]    = 0.0

        if step["phase"] == "yaw":
            dt = now - step["_last_tick"]
            step["_last_tick"] = now
            yawrate = direction * YAW_RATE
            step["degrees_done"] += abs(yawrate) * dt
            step["chunk_done"]   += abs(yawrate) * dt
            if airborne:
                send_world_hover_setpoint(cf, 0, 0, yawrate, height)
            if step["degrees_done"] >= degrees_target:
                step["phase"] = "final_settle"
                step["pause_start"] = now
                if airborne:
                    send_world_hover_setpoint(cf, 0, 0, 0, height)
            elif step["chunk_done"] >= CHUNK_DEG:
                step["phase"] = "pause"
                step["pause_start"] = now
                step["chunk_done"]  = 0.0
                if airborne:
                    send_world_hover_setpoint(cf, 0, 0, 0, height)

        elif step["phase"] == "pause":
            if airborne:
                send_world_hover_setpoint(cf, 0, 0, 0, height)
            if now - step["pause_start"] >= PAUSE_S:
                step["phase"]      = "yaw"
                step["_last_tick"] = now

        elif step["phase"] == "final_settle":
            if airborne:
                send_world_hover_setpoint(cf, 0, 0, 0, height)
            if now - step["pause_start"] >= 0.6:
                _advance_step(total)

    elif action == "circle":
        radius    = float(step.get("radius", 0.5))
        cw        = step.get("direction", "clockwise")
        yaw_dir   = 1.0 if cw == "clockwise" else -1.0
        YAW_RATE  = 40.0                           # deg/s
        # forward speed so arc radius = radius metres
        vx_body   = radius * math.radians(YAW_RATE)
        duration  = 360.0 / YAW_RATE              # one full circle (9s)
        if airborne:
            vx_world, vy_world = body_to_world_velocity(vx_body, 0.0, heading_rad)
            cf.commander.send_velocity_world_setpoint(vx_world, vy_world, 0.0, yaw_dir * YAW_RATE)
        if elapsed >= duration:
            _advance_step(total)

    elif action == "drift":
        # Oscillate strafe + yaw side-to-side: simulates holding A+Q one
        # moment then D+E the next. Each half-cycle is HALF_S seconds.
        duration = float(step.get("duration", 6.0))
        HALF_S   = 1.0
        STRAFE_V = MAX_STRAFE_SPEED * 0.7          # ~70% strafe
        YAW_V    = MAX_YAW_RATE     * 0.5          # ~50% yaw
        # Phase 0..1: drifting left (yaw left + strafe left)
        # Phase 1..2: drifting right (yaw right + strafe right)
        phase = (elapsed % (2 * HALF_S))
        if phase < HALF_S:
            vy_body = +STRAFE_V       # left in body frame
            yawrate = -YAW_V          # yaw left
        else:
            vy_body = -STRAFE_V       # right
            yawrate = +YAW_V          # yaw right
        if airborne:
            vx_world, vy_world = body_to_world_velocity(0.0, vy_body, heading_rad)
            cf.commander.send_velocity_world_setpoint(vx_world, vy_world, 0.0, yawrate)
        if elapsed >= duration:
            _advance_step(total)

    return True


with connect_crazyflie(rw_cache="./cache") as scf:
    cf = scf.cf

    # Discharge curve for a 1S LiPo (Bitcraze BC-LIPO-350mAh) under
    # typical Crazyflie hover load (~1–2 A). Voltages are the *resting*
    # readings on the ground — battery_callback already feeds the
    # smoothed-ground-voltage into voltage_to_pct, not the sagged in-flight
    # value. Knee at ~3.70V where the cell drops off a cliff.
    LIPO_CURVE = [
        (4.20, 100),
        (4.15, 95),
        (4.10, 90),
        (4.05, 83),
        (4.00, 76),
        (3.95, 68),
        (3.90, 60),
        (3.85, 52),
        (3.80, 44),
        (3.75, 36),
        (3.70, 28),    # knee
        (3.65, 20),
        (3.60, 14),
        (3.55, 9),
        (3.50, 5),
        (3.40, 2),
        (3.30, 0),     # firmware low-battery cutoff
    ]

    def voltage_to_pct(v):
        if v >= LIPO_CURVE[0][0]: return 100
        if v <= LIPO_CURVE[-1][0]: return 0
        for i in range(len(LIPO_CURVE) - 1):
            v_hi, p_hi = LIPO_CURVE[i]
            v_lo, p_lo = LIPO_CURVE[i + 1]
            if v_lo <= v <= v_hi:
                t = (v - v_lo) / (v_hi - v_lo)
                return int(p_lo + t * (p_hi - p_lo))
        return 0

    def update_voltage_smoothing(raw_voltage):
        global voltage_samples, smoothed_voltage, ground_voltage_samples, smoothed_ground_volt

        voltage_samples.append(raw_voltage)
        if len(voltage_samples) > 50:
            voltage_samples.pop(0)
        smoothed_voltage = sum(voltage_samples) / len(voltage_samples)

        if not airborne:
            ground_voltage_samples.append(raw_voltage)
            if len(ground_voltage_samples) > 30:
                ground_voltage_samples.pop(0)
            if ground_voltage_samples:
                smoothed_ground_volt = sum(ground_voltage_samples) / len(ground_voltage_samples)

    log_conf = LogConfig(name="Flight", period_in_ms=100)
    log_conf.add_variable("pm.vbat", "float")
    log_conf.add_variable("stateEstimate.yaw", "float")
    log_conf.add_variable("stateEstimate.z", "float")
    log_conf.add_variable("stateEstimate.x", "float")
    log_conf.add_variable("stateEstimate.y", "float")
    log_conf.add_variable("range.zrange", "uint16_t")

    sup_conf = LogConfig(name="Supervisor", period_in_ms=100)
    sup_conf.add_variable("supervisor.info", "uint16_t")

    def supervisor_callback(_, data, __):
        global supervisor_info
        supervisor_info = int(data["supervisor.info"])

    def battery_callback(_, data, __):
        global battery_pct, battery_volt, heading_rad, state_z_m, range_z_m, state_x_m, state_y_m
        raw_voltage = data["pm.vbat"]
        battery_volt = raw_voltage

        update_voltage_smoothing(raw_voltage)
        ref = smoothed_ground_volt if smoothed_ground_volt > 0 else smoothed_voltage
        battery_pct = voltage_to_pct(ref)
        battery_history.append((time.time(), battery_pct))

        heading_rad = math.radians(data["stateEstimate.yaw"])
        state_z_m = float(data["stateEstimate.z"])
        state_x_m = float(data["stateEstimate.x"])
        state_y_m = float(data["stateEstimate.y"])
        raw_range = int(data["range.zrange"])
        range_z_m = raw_range / 1000.0 if 50 <= raw_range <= 3000 else None

    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(battery_callback)
    log_conf.start()

    try:
        cf.log.add_config(sup_conf)
        sup_conf.data_received_cb.add_callback(supervisor_callback)
        sup_conf.start()
    except Exception as e:
        print(f"Supervisor log unavailable: {e}")

    cf.commander.send_stop_setpoint()
    time.sleep(0.2)
    print("Ready. Click the window, then press T or Y to take off.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if text_input_active:
                    if event.key == pygame.K_RETURN:
                        cmd = text_input_buffer.strip()
                        text_input_active = False
                        text_input_buffer = ""
                        pygame.key.stop_text_input()
                        if cmd:
                            ai_heard = cmd
                            ai_reply = ""
                            ai_status = "Thinking..."
                            submit_text_command(cmd, airborne, height)
                    elif event.key == pygame.K_BACKSPACE:
                        text_input_buffer = text_input_buffer[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        text_input_active = False
                        text_input_buffer = ""
                        pygame.key.stop_text_input()
                else:
                    keys_held.add(event.key)
                    if event.key == pygame.K_SLASH:
                        text_input_active = True
                        text_input_buffer = ""
                        pygame.key.start_text_input()
                    elif event.key == pygame.K_v and not ai_voice_recording:
                        ai_voice_recording = True
                        stop_tracking()
                        set_mic_active(True)
                        if ai_steps:
                            print("[AI] Interrupted by PTT")
                        ai_steps = []
                        ai_step_index = 0
                        ai_step_start = 0.0
                        ai_status = ""
                        while not ai_result_queue.empty():
                            try: ai_result_queue.get_nowait()
                            except: break
                    elif event.key == pygame.K_r and not _recovering:
                        _recovering = True
                        def _do_recover():
                            global airborne, height, smoothed_height, ai_steps, ai_step_index
                            global ai_reply, ai_heard, ai_status, _recovering
                            print("Recovering...")
                            for _ in range(50):
                                cf.commander.send_setpoint(0, 0, 0, 0)
                                time.sleep(0.01)
                            cf.commander.send_stop_setpoint()
                            time.sleep(0.2)
                            cf.platform.send_crash_recovery_request()
                            time.sleep(0.5)
                            cf.platform.send_arming_request(True)
                            time.sleep(0.3)
                            airborne = False
                            height = current_takeoff_target()
                            smoothed_height = height
                            ai_steps = []
                            ai_step_index = 0
                            ai_reply = "Recovered. Ready when you are."
                            ai_heard = ""
                            ai_status = ""
                            clear_history()
                            notify_event("Drone has been recovered by pilot. Ready to fly again.")
                            speak_async("Recovered. Ready when you are.")
                            print("Recovered. Press T to take off again.")
                            _recovering = False
                        threading.Thread(target=_do_recover, daemon=True).start()
                    elif event.key == pygame.K_t and not airborne:
                        height = do_takeoff(cf)
                        airborne = True
                        smoothed_height = height
                        cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                    elif event.key == pygame.K_l and airborne:
                        do_land(cf, height)
                        airborne = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F5:
                        _restart_requested = True
                        running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and RESTART_BTN_RECT.collidepoint(event.pos):
                    _restart_requested = True
                    running = False

            elif event.type == pygame.TEXTINPUT and text_input_active:
                text_input_buffer += event.text

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_v and ai_voice_recording:
                    ai_voice_recording = False
                    set_mic_active(False)
                keys_held.discard(event.key)

            elif event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    left_x = apply_deadzone(event.value)
                elif event.axis == 1:
                    left_y = apply_deadzone(event.value)
                elif event.axis == 2:
                    right_x = apply_deadzone(event.value)
                elif event.axis == 4:
                    left_trigger = event.value
                elif event.axis == 5:
                    right_trigger = event.value

            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3 and not airborne:
                    height = do_takeoff(cf)
                    airborne = True
                    smoothed_height = height
                    cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                elif event.button == 0 and airborne:
                    do_land(cf, height)
                    airborne = False
                elif event.button == 7 and not _recovering:
                    def _do_recover_ctrl():
                        global airborne, height, smoothed_height, ai_steps, ai_step_index
                        global ai_reply, ai_heard, ai_status, _recovering
                        print("Recovering (controller)...")
                        for _ in range(50):
                            cf.commander.send_setpoint(0, 0, 0, 0)
                            time.sleep(0.01)
                        cf.commander.send_stop_setpoint()
                        time.sleep(0.2)
                        cf.platform.send_crash_recovery_request()
                        time.sleep(0.5)
                        cf.platform.send_arming_request(True)
                        time.sleep(0.3)
                        airborne = False
                        height = current_takeoff_target()
                        smoothed_height = height
                        ai_steps = []
                        ai_step_index = 0
                        ai_reply = "Recovered. Ready when you are."
                        ai_heard = ""
                        ai_status = ""
                        clear_history()
                        notify_event("Drone has been recovered by pilot. Ready to fly again.")
                        speak_async("Recovered. Ready when you are.")
                        _recovering = False
                    _recovering = True
                    threading.Thread(target=_do_recover_ctrl, daemon=True).start()
                elif event.button == 2 and not ai_voice_recording:
                    ai_voice_recording = True
                    stop_tracking()
                    set_mic_active(True)
                    if ai_steps:
                        print("[AI] Interrupted by controller PTT")
                    ai_steps = []
                    ai_step_index = 0
                    ai_step_start = 0.0
                    ai_status = ""
                    while not ai_result_queue.empty():
                        try: ai_result_queue.get_nowait()
                        except: break

            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 2 and ai_voice_recording:
                    ai_voice_recording = False
                    set_mic_active(False)

        ai_commanding = tick_ai_executor(cf)

        if not ai_commanding:
            if airborne:
                keys = pygame.key.get_pressed()

                vx_body = -left_y * MAX_FORWARD_SPEED
                vy_body = -left_x * MAX_STRAFE_SPEED
                yawrate = -right_x * MAX_YAW_RATE

                rt = (right_trigger + 1.0) / 2.0
                lt = (left_trigger + 1.0) / 2.0
                vz = (rt - lt) * MAX_VERTICAL_SPEED

                if keys[pygame.K_UP]:
                    vz += MAX_VERTICAL_SPEED
                if keys[pygame.K_DOWN]:
                    vz -= MAX_VERTICAL_SPEED

                vx_body += ((1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)) * MAX_FORWARD_SPEED
                vy_body += ((1.0 if keys[pygame.K_a] else 0.0) - (1.0 if keys[pygame.K_d] else 0.0)) * MAX_STRAFE_SPEED
                yawrate += ((1.0 if keys[pygame.K_q] else 0.0) - (1.0 if keys[pygame.K_e] else 0.0)) * (MAX_YAW_RATE * 0.75)

                vx_body = clamp(vx_body, -MAX_FORWARD_SPEED, MAX_FORWARD_SPEED)
                vy_body = clamp(vy_body, -MAX_STRAFE_SPEED, MAX_STRAFE_SPEED)
                vz = clamp(vz, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
                yawrate = clamp(yawrate, -MAX_YAW_RATE, MAX_YAW_RATE)

                # Snap tiny stick / sensor noise to exact zero so the drone
                # doesn't crawl forward from controller bias when no input.
                IDLE_LIN = 0.05   # m/s
                IDLE_YAW = 1.5    # deg/s
                if abs(vx_body) < IDLE_LIN: vx_body = 0.0
                if abs(vy_body) < IDLE_LIN: vy_body = 0.0
                if abs(vz)      < IDLE_LIN: vz      = 0.0
                if abs(yawrate) < IDLE_YAW: yawrate = 0.0

                if altitude_telemetry_ready():
                    height = clamp(state_z_m, MIN_WORLD_HEIGHT, MAX_WORLD_HEIGHT)
                    smoothed_height = height
                vx_world, vy_world = body_to_world_velocity(vx_body, vy_body, heading_rad)
                cf.commander.send_velocity_world_setpoint(vx_world, vy_world, vz, yawrate)
            else:
                cf.commander.send_stop_setpoint()

        is_tumbled = bool(supervisor_info & (1 << 5))
        is_locked  = bool(supervisor_info & (1 << 6))
        crashed    = is_tumbled or is_locked

        if crashed and not _prev_crashed:
            add_to_action_log(f"CRASH at h={height:.2f}m batt={battery_pct}%")
            notify_event(
                f"Drone just crashed (tumbled/locked). Height was {height:.2f}m, battery {battery_pct}%. "
                f"Waiting for pilot to press R to recover."
            )
            speak_async("Oh no, I crashed. Press R to recover me.")
        _prev_crashed = crashed

        if camera_connected and not _prev_camera_connected:
            notify_event("Camera reconnected — visual feed restored.")
        elif not camera_connected and _prev_camera_connected:
            notify_event("Camera disconnected — no visual feed available.")
        _prev_camera_connected = camera_connected

        update_drone_state(airborne, height, battery_pct, crashed)

        if battery_pct > 0 and battery_pct < 20 and time.time() - last_battery_warn > 30:
            speak_async("Warning, battery low")
            last_battery_warn = time.time()

        draw_ui()
        time.sleep(0.01)

    if airborne:
        do_land(cf, height)
    print("Done.")

if _restart_requested:
    print("Restarting...")
    pygame.quit()
    os.execv(sys.executable, [sys.executable] + sys.argv)
