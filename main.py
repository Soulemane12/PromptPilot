import time
import threading
import socket
import struct
import math
import os
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
    start_recording,
    stop_recording_and_submit,
    open_audio_stream,
)

DEFAULT_ADDR = "E7E7E7E7E7"
DEFAULT_URI = f"radio://0/80/2M/{DEFAULT_ADDR}"
DEADZONE = 0.12
HOVER_HEIGHT = 1.0
DECK_IP = "192.168.4.1"
DECK_PORT = 5000
MAX_FORWARD_SPEED = 1.2
MAX_STRAFE_SPEED = 0.8
MAX_VERTICAL_SPEED = 0.35
MAX_YAW_RATE = 80.0
MAX_WORLD_HEIGHT = 1.2
MIN_WORLD_HEIGHT = 0.2
MAX_HOVER_ZDISTANCE = 2.0
MIN_HOVER_ZDISTANCE = 0.15
ALTITUDE_RESPONSE = 0.15

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

# --- Camera feed thread ---
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
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((DECK_IP, DECK_PORT))
            sock.settimeout(60)
            camera_connected = True
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
                        latest_frame = cv2.resize(frame, (420, 320), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            camera_connected = False
            print(f"Camera reconnecting... ({e})")
            time.sleep(2)

cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()

# --- Pygame setup ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((420, 900))
pygame.display.set_caption("PromptPilot — click here first!")
font = pygame.font.SysFont("monospace", 15)

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

    # Camera feed
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
        pygame.draw.rect(screen, (40, 40, 40), (0, 0, 420, 320))
        msg = font.render(status, True, (150, 150, 150))
        screen.blit(msg, (120, 150))

    # Battery bar
    pct = max(0, min(100, battery_pct))
    bar_color = (0, 220, 80) if pct > 40 else (255, 180, 0) if pct > 20 else (220, 50, 50)
    pygame.draw.rect(screen, (60, 60, 60), (15, 328, 390, 18), border_radius=4)
    pygame.draw.rect(screen, bar_color, (15, 328, int(390 * pct / 100), 18), border_radius=4)
    if airborne and load_compensation_factor > 0:
        comp_info = f" +{load_compensation_factor:.2f}V comp"
        label = font.render(f"Battery: {pct}%  ({battery_volt:.2f}V{comp_info})", True, (255, 255, 255))
    else:
        label = font.render(f"Battery: {pct}%  ({battery_volt:.2f}V)", True, (255, 255, 255))
    screen.blit(label, (120, 329))

    drain, mins_left = estimate_time_remaining()
    if mins_left is not None:
        time_color = (0, 220, 80) if mins_left > 3 else (255, 180, 0) if mins_left > 1.5 else (220, 50, 50)
        time_label = font.render(f"~{mins_left} min left  (-{drain}%/min)", True, time_color)
    else:
        time_label = font.render("Calculating flight time...", True, (150, 150, 150))
    screen.blit(time_label, (15, 350))

    if altitude_telemetry_ready():
        altitude_label = font.render(f"Flight mode: manual altitude  z={state_z_m:.2f}m  range={range_z_m:.2f}m", True, (120, 220, 255))
    else:
        altitude_label = font.render("Flight mode: manual altitude  waiting for telemetry", True, (255, 180, 0))
    screen.blit(altitude_label, (15, 370))

    ctrl_label = "PLAYSTATION CONTROLS" if is_ps else "XBOX CONTROLS"
    takeoff_btn = "Triangle         Take off" if is_ps else "Y button        Take off"
    land_btn    = "Cross (X)        Land"     if is_ps else "A button        Land"
    lines = [
        ("", (255, 255, 255)),
        (ctrl_label, (0, 200, 255)),
        ("Left Stick      Move fwd/back/left/right", (200, 200, 200)),
        ("Right Stick X   Yaw", (200, 200, 200)),
        ("R2              Go up", (200, 200, 200)) if is_ps else ("Right Trigger   Go up", (200, 200, 200)),
        ("L2              Go down", (200, 200, 200)) if is_ps else ("Left Trigger    Go down", (200, 200, 200)),
        (takeoff_btn, (200, 200, 200)),
        (land_btn, (200, 200, 200)),
        ("", (255, 255, 255)),
        ("KEYBOARD CONTROLS", (0, 200, 255)),
        ("W / S           Forward / Back", (200, 200, 200)),
        ("A / D           Left / Right", (200, 200, 200)),
        ("Q / E           Yaw right / left", (200, 200, 200)),
        ("Up / Down       Up / Down", (200, 200, 200)),
        ("T               Take off", (200, 200, 200)),
        ("L               Land", (200, 200, 200)),
        ("ESC             Quit", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("TRIM (fix drift)", (0, 200, 255)),
        ("I / K           Trim forward / back", (200, 200, 200)),
        ("J / ;           Trim left / right", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("R               Recover after crash", (255, 100, 100)),
    ]
    for i, (text, color) in enumerate(lines):
        surface = font.render(text, True, color)
        screen.blit(surface, (15, 390 + i * 20))

    # --- AI command UI ---
    # Status line
    if ai_status:
        status_color = (255, 80, 80) if ai_status.startswith("Error") else (255, 220, 50)
        screen.blit(font.render(ai_status[:55], True, status_color), (15, 858))
    # Text input box
    box_color = (70, 130, 200) if text_input_active else (45, 45, 45)
    pygame.draw.rect(screen, box_color, (15, 878, 390, 18), border_radius=3)
    if text_input_active:
        display_text = "/" + text_input_buffer + "|"
    else:
        display_text = "Press / to type  |  Hold V to speak"
    screen.blit(font.render(display_text, True, (255, 255, 255)), (18, 880))

    pygame.display.flip()

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

open_audio_stream()

cflib.crtp.init_drivers(enable_debug_driver=False)
print(f"Preferred radio URI: {DEFAULT_URI}")

left_x = left_y = right_x = 0.0
left_trigger = right_trigger = -1.0
running = True
airborne = False
height = HOVER_HEIGHT
smoothed_height = HOVER_HEIGHT
keys_held = set()
battery_pct = 0
battery_volt = 0.0
battery_history = []
voltage_samples = []
smoothed_voltage = 0.0
resting_voltage = 0.0
load_compensation_factor = 0.0
trim_vx = 0.0
trim_vy = 0.0
heading_rad = 0.0
state_z_m = 0.0
range_z_m = None

# --- AI command state ---
ai_steps: list = []
ai_step_index: int = 0
ai_step_start: float = 0.0
ai_status: str = ""
ai_voice_recording: bool = False

# --- Text input state ---
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
    print("Taking off...")
    base_world_z = state_z_m if altitude_telemetry_ready() else 0.0
    target_world_z = current_takeoff_target()
    send_world_hover_setpoint(cf, 0, 0, 0, base_world_z)
    time.sleep(0.1)
    h = base_world_z
    for _ in range(50):
        h += (target_world_z - base_world_z) / 50
        send_world_hover_setpoint(cf, 0, 0, 0, h)
        time.sleep(0.05)
    print("Airborne!")
    return target_world_z

def do_land(cf, current_height):
    print("Landing...")
    h = current_height
    steps = max(1, int(h / 0.01))
    for _ in range(steps):
        h = max(0.0, h - 0.01)
        send_world_hover_setpoint(cf, 0, 0, 0, h)
        time.sleep(0.02)
    cf.commander.send_stop_setpoint()
    print("Landed.")


def _is_manual_input_active() -> bool:
    """True if any stick/key input exceeds the deadzone — used to cancel AI sequences."""
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
    ai_status = f"Executing step {ai_step_index + 1}/{total}" if ai_step_index < total else "Done"


def tick_ai_executor(cf) -> bool:
    """
    Called every frame. Drains the AI result queue, advances the active step,
    and sends the appropriate setpoint. Returns True while the AI is in control.
    """
    global ai_steps, ai_step_index, ai_step_start, ai_status, airborne, height, smoothed_height

    # Drain result queue (non-blocking)
    while True:
        try:
            kind, payload = ai_result_queue.get_nowait()
        except queue.Empty:
            break
        if kind == "status":
            ai_status = payload
        elif kind == "steps":
            ai_steps = payload
            ai_step_index = 0
            ai_step_start = time.time()
            total = len(payload)
            ai_status = f"Executing step 1/{total}" if total else "Done"

    if not ai_steps:
        return False

    # Manual stick/key cancels the sequence immediately
    if _is_manual_input_active():
        ai_steps = []
        ai_step_index = 0
        ai_status = "Cancelled (manual override)"
        return False

    if ai_step_index >= len(ai_steps):
        ai_steps = []
        ai_status = "Done"
        return False

    step    = ai_steps[ai_step_index]
    now     = time.time()
    elapsed = now - ai_step_start
    action  = step["action"]
    total   = len(ai_steps)

    if action == "takeoff":
        if not airborne:
            height = do_takeoff(cf)
            airborne = True
            smoothed_height = height
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
        degrees_target = abs(step.get("degrees", 90))
        direction = 1.0 if step.get("degrees", 90) >= 0 else -1.0
        if "degrees_done" not in step:
            step["degrees_done"] = 0.0
            step["_last_tick"] = now
        dt = now - step["_last_tick"]
        step["_last_tick"] = now
        yawrate = direction * MAX_YAW_RATE * 0.8
        step["degrees_done"] += abs(yawrate) * dt
        if airborne:
            send_world_hover_setpoint(cf, 0, 0, yawrate, height)
        if step["degrees_done"] >= degrees_target:
            _advance_step(total)

    return True


with connect_crazyflie(rw_cache="./cache") as scf:
    cf = scf.cf

    LIPO_CURVE = [
        (4.20, 100), (4.15, 95), (4.10, 90), (4.05, 85),
        (4.00, 80),  (3.95, 75), (3.90, 70), (3.85, 65),
        (3.80, 60),  (3.75, 55), (3.70, 50), (3.65, 45),
        (3.60, 40),  (3.55, 35), (3.50, 30), (3.45, 25),
        (3.40, 20),  (3.35, 15), (3.30, 10), (3.20, 5),
        (3.00, 0),
    ]

    def voltage_to_pct(v, compensated=True):
        voltage = v
        if compensated and airborne and load_compensation_factor > 0:
            voltage = v + load_compensation_factor

        if voltage >= LIPO_CURVE[0][0]: return 100
        if voltage <= LIPO_CURVE[-1][0]: return 0
        for i in range(len(LIPO_CURVE) - 1):
            v_hi, p_hi = LIPO_CURVE[i]
            v_lo, p_lo = LIPO_CURVE[i + 1]
            if v_lo <= voltage <= v_hi:
                t = (voltage - v_lo) / (v_hi - v_lo)
                return int(p_lo + t * (p_hi - p_lo))
        return 0

    def update_voltage_smoothing(raw_voltage):
        global voltage_samples, smoothed_voltage, resting_voltage, load_compensation_factor

        voltage_samples.append(raw_voltage)
        if len(voltage_samples) > 50:
            voltage_samples.pop(0)

        smoothed_voltage = sum(voltage_samples) / len(voltage_samples)

        if not airborne:
            if len(voltage_samples) >= 10:
                recent_samples = voltage_samples[-10:]
                variance = sum((v - smoothed_voltage)**2 for v in recent_samples) / len(recent_samples)
                if variance < 0.001:
                    resting_voltage = smoothed_voltage

        if airborne and resting_voltage > 0 and smoothed_voltage < resting_voltage:
            load_compensation_factor = min(0.15, resting_voltage - smoothed_voltage)
        else:
            load_compensation_factor = 0.0

    log_conf = LogConfig(name="Flight", period_in_ms=100)
    log_conf.add_variable("pm.vbat", "float")
    log_conf.add_variable("stateEstimate.yaw", "float")
    log_conf.add_variable("stateEstimate.z", "float")
    log_conf.add_variable("range.zrange", "uint16_t")

    def battery_callback(_, data, __):
        global battery_pct, battery_volt, heading_rad, state_z_m, range_z_m
        raw_voltage = data["pm.vbat"]
        battery_volt = raw_voltage

        update_voltage_smoothing(raw_voltage)
        battery_pct = voltage_to_pct(smoothed_voltage, compensated=True)
        battery_history.append((time.time(), battery_pct))

        heading_rad = math.radians(data["stateEstimate.yaw"])
        state_z_m = float(data["stateEstimate.z"])
        raw_range = int(data["range.zrange"])
        range_z_m = raw_range / 1000.0 if 50 <= raw_range <= 3000 else None

    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(battery_callback)
    log_conf.start()

    cf.commander.send_stop_setpoint()
    time.sleep(0.2)
    print("Ready. Click the window, then press T or Y to take off.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                # --- Text input mode: intercept all keys ---
                if text_input_active:
                    if event.key == pygame.K_RETURN:
                        cmd = text_input_buffer.strip()
                        text_input_active = False
                        text_input_buffer = ""
                        pygame.key.stop_text_input()
                        if cmd:
                            submit_text_command(cmd, airborne, height)
                            ai_status = "Thinking..."
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
                        start_recording()
                        ai_status = "Listening..."
                    elif event.key == pygame.K_r:
                        print("Recovering...")
                        for _ in range(50):
                            cf.commander.send_setpoint(0, 0, 0, 0)
                            time.sleep(0.01)
                        cf.commander.send_stop_setpoint()
                        time.sleep(0.2)
                        cf.commander.send_stop_setpoint()
                        airborne = False
                        height = current_takeoff_target()
                        smoothed_height = height
                        print("Recovered. Press T to take off again.")
                    elif event.key == pygame.K_i:
                        trim_vx += 0.02
                    elif event.key == pygame.K_k:
                        trim_vx -= 0.02
                    elif event.key == pygame.K_j:
                        trim_vy += 0.02
                    elif event.key == pygame.K_SEMICOLON:
                        trim_vy -= 0.02
                    elif event.key == pygame.K_t and not airborne:
                        height = do_takeoff(cf)
                        airborne = True
                        smoothed_height = height
                    elif event.key == pygame.K_l and airborne:
                        do_land(cf, height)
                        airborne = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            elif event.type == pygame.TEXTINPUT and text_input_active:
                text_input_buffer += event.text

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_v and ai_voice_recording:
                    ai_voice_recording = False
                    stop_recording_and_submit(airborne, height)
                    ai_status = "Transcribing..."
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
                elif event.button == 0 and airborne:
                    do_land(cf, height)
                    airborne = False

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

                vx_body = clamp(vx_body + trim_vx, -MAX_FORWARD_SPEED, MAX_FORWARD_SPEED)
                vy_body = clamp(vy_body + trim_vy, -MAX_STRAFE_SPEED, MAX_STRAFE_SPEED)
                vz = clamp(vz, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
                yawrate = clamp(yawrate, -MAX_YAW_RATE, MAX_YAW_RATE)
                if altitude_telemetry_ready():
                    height = clamp(state_z_m, MIN_WORLD_HEIGHT, MAX_WORLD_HEIGHT)
                    smoothed_height = height
                vx_world, vy_world = body_to_world_velocity(vx_body, vy_body, heading_rad)
                cf.commander.send_velocity_world_setpoint(vx_world, vy_world, vz, yawrate)
            else:
                cf.commander.send_stop_setpoint()

        draw_ui()
        time.sleep(0.01)

    if airborne:
        do_land(cf, height)
    print("Done.")
