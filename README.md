# PromptPilot

**Prompt → Commands → Drone**

A natural-language pilot for the Bitcraze Crazyflie 2.x nano-drone.
Talk to it, type at it, or fly it manually with a keyboard or game controller.
Includes a live forward-camera feed (AI deck), GPT-4o vision ("what do you see?",
"follow that person"), and a realtime voice mode that speaks back to you.

```
┌─────────────────────────────────────────────────────────────┐
│  AI deck WiFi camera feed (live, sharpened)                 │
├─────────────────────────────────────────────────────────────┤
│  Battery 78%  (3.92 V)        ~6 min left  (-3.4 %/min)     │
│  z=1.27 m   range=1.30 m                                    │
│  ● FLYING                                                   │
├─────────────────────────────────────────────────────────────┤
│  CONTROLLER       │   KEYBOARD                              │
│  L.Stick  Move    │   W/S  Fwd/Back                         │
│  R.Stick  Yaw     │   A/D  Strafe                           │
│  Y/△      T/O     │   Q/E  Yaw                              │
│  A/✕      Land    │   T/L  Take off / Land                  │
│  X/□  hold Voice  │   V hold   Voice                        │
│                   │   /   Type AI command                   │
│                   │   R   Recover after crash               │
├─────────────────────────────────────────────────────────────┤
│  You: "Take off and fly in a circle, then land"             │
│  Drone: On it — full circle and landing.                    │
│  Step 2/3 — circle                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## What it does

### Voice & text control
- **Hold V (or ▢ on PlayStation, X on Xbox)** to talk. Release to send.
- **Press `/`** to type a command and hit Enter.
- The AI responds out loud and queues a sequence of flight steps.
- Say "stop" or press V mid-sequence to cancel.

### Flight tools the AI can call
| Tool | What it does |
|---|---|
| `takeoff` | Calibrates the Kalman estimator, then climbs to ~1.3 m hover |
| `land` | Smooth descent + motor cut |
| `move(vx, vy, vz, yaw, duration)` | Open-loop velocity command for a duration |
| `rotate(degrees)` | In-place spin, broken into 45° chunks with pauses to kill drift |
| `circle(radius, direction)` | Continuous orbit (default 0.5 m, 9 s for 360°) |
| `drift(duration)` | Side-to-side oscillating strafe + yaw — like a drift dance |
| `wait(duration)` | Hover in place |
| `stop` | Cancel current sequence |
| `describe_scene` | GPT-4o looks through the front camera and describes what it sees |
| `follow_subject(subject)` | Continuous vision tracking — yaws & advances toward a target |

### Manual flight
- Game controller (Xbox / PlayStation) auto-detected on launch.
- Keyboard fallback if no controller.
- Snap-to-zero on tiny stick noise so the drone holds position when idle.

### Live feed
- The AI deck (Bitcraze ESP32-CAM expansion) streams 320×244 Bayer frames
  over WiFi to TCP `192.168.4.1:5000`.
- Code applies CLAHE contrast + sharpening and upscales to 860×300.
- Camera connect/disconnect events are pushed to the AI in realtime
  (`[SYSTEM EVENT] Camera reconnected`) so it knows when it can / can't see.

### Crash awareness
- Reads the Crazyflie supervisor bitmask every 100 ms.
- When tumbled / locked, shows **CRASHED** badge and tells the AI.
- Press **R** (or controller Menu/Options) to send a crash-recovery request.

### RESTART without Ctrl-C
- **F5** or click the `RESTART` button (top-right of the camera feed).
- Lands the drone, disconnects, and re-execs the script in place.

---

## Hardware

**Required**
- Crazyflie 2.0 / 2.1 / 2.1+ with up-to-date firmware
- Crazyradio PA dongle (USB, plugged into your Mac/PC)
- Flow deck v2 (optical flow + ZRanger — required for hover stability)
- AI deck (for the camera feed)
- BC-LIPO-350mAh-1S (or stock 250 mAh) LiPo battery

**Optional**
- Xbox or PlayStation controller (any pygame-compatible HID)

---

## Setup

### 1. Python 3.11

```bash
python3.11 -m venv ~/drone_venv
source ~/drone_venv/bin/activate
pip install --upgrade pip
```

### 2. Dependencies

```bash
pip install \
    cflib \
    pygame \
    opencv-python \
    numpy \
    sounddevice \
    websocket-client \
    openai \
    python-dotenv
```

### 3. OpenAI API key

Create a `.env` file at the repo root:

```env
OPENAI_API_KEY=sk-...
```

### 4. Crazyflie firmware

Flash the Crazyflie with firmware ≥ 2024.10 so `kalman.resetEstimation`,
`platform.send_crash_recovery_request()`, and the `supervisor.info` log group
exist. The bundled `firmware/firmware-cf2-2025.12.1.zip` works.

```bash
python -m cfloader flash firmware/firmware-cf2-2025.12.1.zip stm32-fw
```

### 5. AI deck firmware (camera streaming)

Flash the AI deck with the WiFi-streamer image:

```bash
python tools/flash_aideck.py firmware/aideck_gap8_wifi_img_streamer_with_ap.bin
```

After flashing and rebooting, the deck broadcasts an open WiFi network
named something like `Bitcraze AI-deck Example`.

---

## Running it

```bash
~/drone_venv/bin/python main.py
```

You'll see something like:

```
pygame 2.6.1 (SDL 2.28.4, Python 3.11.11)
Xbox controller detected.
[RT] Connected to Realtime API
Drone connected at: radio://0/80/2M/E7E7E7E7E7
Camera connected! Waiting for frames...
Ready. Click the window, then press T or Y to take off.
```

### To get the camera feed
Two WiFi networks are involved if you also want internet for the OpenAI API:

- Connect your **built-in WiFi (en0)** to your home network for the OpenAI API.
- Connect a **USB WiFi dongle (en1)** to the AI deck's `Bitcraze AI-deck` SSID.

…or accept "no internet while flying" and just connect to the deck's WiFi.

Verify the deck is reachable:

```bash
ping 192.168.4.1
```

If it times out → you're not on the deck's WiFi or the deck isn't broadcasting.

---

## Controls reference

### Keyboard
| Key | Action |
|---|---|
| `W` / `S` | Forward / Back (1.0 m/s) |
| `A` / `D` | Strafe Left / Right (0.7 m/s) |
| `Q` / `E` | Yaw Left / Right (60 °/s) |
| `↑` / `↓` | Up / Down (0.35 m/s) |
| `T` | Take off |
| `L` | Land |
| `R` | Recover after crash |
| `V` (hold) | Push-to-talk |
| `/` | Type AI command |
| `F5` | Restart program |
| `Esc` | Quit (lands first) |

### Xbox / PlayStation controller
| Control | Action |
|---|---|
| Left stick | Forward / Back / Strafe |
| Right stick X | Yaw |
| RT / R2 | Up |
| LT / L2 | Down |
| Y / Triangle | Take off |
| A / Cross | Land |
| Menu / Options | Recover |
| X / Square (hold) | Push-to-talk |

---

## Speed & flight tuning

All in [main.py](main.py) at the top. Conservative defaults — bump if you have
the room.

```python
HOVER_HEIGHT       = 1.3   # m  — takeoff target
MAX_FORWARD_SPEED  = 1.0   # m/s
MAX_STRAFE_SPEED   = 0.7   # m/s
MAX_VERTICAL_SPEED = 0.35  # m/s
MAX_YAW_RATE       = 80.0  # deg/s
```

Sane upper bounds for Flow-deck-only stability: 2.0 m/s horizontal, 1.0 m/s
vertical, 120 °/s yaw. Past that, optical flow loses tracking.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Drone drifts forward on takeoff | Kalman bias / Flow-deck on bad surface | Land on textured surface (carpet, paper). The 1.5 s "Calibrating estimator…" wait already runs each takeoff. |
| Drone goes "crazy" on takeoff | Optical-flow-defeating surface (glass, mirror, plain white) | Move to textured floor. |
| Drifts in a small circle during 360° | Yaw confuses optical flow | Already mitigated by chunked rotation (45° + pause). |
| Battery shows 80% then plunges to 20% in flight | Voltage-sag confusion | Already fixed — % is computed from ground-only voltage samples. |
| Camera frozen during takeoff/landing | Old: main thread blocked | Already fixed — `do_takeoff` / `do_land` pump events while waiting. |
| Multi-step sequence drops `rotate` | Old: `ai_steps` overwritten | Already fixed — uses `extend()`. |
| `Camera reconnecting... timed out` forever | Mac on wrong WiFi | `networksetup -getairportnetwork en0` then connect to the deck's SSID. |
| `Error no LogEntry to handle id=2` | Cosmetic — logging quirk | Harmless. |

---

## Repo layout

```
PromptPilot/
├─ main.py                    # main loop, GUI, manual control, AI executor
├─ tools/
│   ├─ ai_commands.py         # GPT-4o + Realtime API + flight tool schemas
│   ├─ first_flight.py        # Bitcraze hello-world
│   ├─ flash_aideck.py        # AI-deck firmware flasher
│   ├─ opencv-viewer.py       # standalone camera viewer
│   ├─ scan.py                # Crazyradio link scanner
│   └─ test_connection.py     # connectivity sanity check
├─ firmware/                  # bundled .zip / .bin firmware images
├─ models/                    # YOLOv8 ONNX/PT (offline vision experiments)
├─ cache/                     # cflib parameter cache (auto-generated)
├─ .env                       # OPENAI_API_KEY (you create this)
└─ .python-version            # 3.11.11
```

---

## How the AI loop works

1. You speak / type a command.
2. `tools/ai_commands.py` sends it (with current drone state) to GPT-4o.
3. GPT-4o replies with text **and** one or more `function_call`s
   (`takeoff`, `move`, `rotate`, `circle`, `drift`, …).
4. Each function call is validated and pushed onto a queue.
5. `main.py`'s `tick_ai_executor` pops steps off the queue every frame
   and turns them into `cflib` commander setpoints.
6. The Realtime API (separate WebSocket) plays GPT-4o's spoken reply
   while flight executes.

System events (crash, camera connect/disconnect, recovery) are pushed back
into the conversation so the AI is always aware of physical state.

---

## License & credits

Built on top of:
- [Bitcraze cflib](https://github.com/bitcraze/crazyflie-lib-python)
- [Bitcraze AI-deck examples](https://github.com/bitcraze/aideck-gap8-examples)
- [OpenAI Python SDK](https://github.com/openai/openai-python) (Realtime + Chat Completions)
- [pygame](https://www.pygame.org), [OpenCV](https://opencv.org)

PromptPilot itself is MIT-licensed. Fly responsibly. The Crazyflie is a real
flying machine — even at 1 m/s a propeller in someone's eye is a bad day.
Always have a clear flight area and a finger near `Esc`.
