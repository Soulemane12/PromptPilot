# PromptPilot

**Prompt → Commands → Drone**

---

<p align="center">
  <img src="https://www.bitcraze.io/images/logo/Bitcraze_Logo_1024px.png" alt="Bitcraze Logo" width="200"/>
</p>

---

## Crazyflie Client Installation

> **Requirements:** Python 3.10+ &nbsp;|&nbsp; Recommended: use a [Python virtual environment](https://docs.python.org/3/library/venv.html)

---

### Platform Prerequisites

<details>
<summary><strong>Ubuntu / Linux</strong></summary>

```bash
sudo apt install git python3-pip libxcb-xinerama0 libxcb-cursor0
pip3 install --upgrade pip
```

> **udev permissions** — Using Crazyradio on Linux requires setting udev permissions. See the [cflib installation guide](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/) for details.

</details>

<details>
<summary><strong>Windows</strong></summary>

1. Install [Python 3](https://www.python.org/downloads/) — check **"Add to PATH"** during setup.
2. Verify:
   ```powershell
   python --version
   pip --version
   ```
3. Upgrade pip:
   ```powershell
   pip install --upgrade pip
   ```
4. Install [Git](https://git-scm.com/downloads) and verify with `git --version`.
5. **Python 3.13 only:** install [Visual Studio](https://visualstudio.microsoft.com/) with the **Desktop Development with C++** workload.
6. Install [Crazyradio drivers](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

</details>

<details>
<summary><strong>macOS</strong></summary>

- Requires macOS 11 (Big Sur) or later — works on both x86 and Apple Silicon.
- Compatible with the Apple-provided Python 3 (≥ 3.10) or Homebrew Python.

</details>

<details>
<summary><strong>Raspberry Pi</strong></summary>

- Requires Raspberry Pi OS Trixie or later.
- Must create a Python venv to install the client on Trixie.
- GUI works on both Pi 4 and Pi 5 (Pi 5 recommended).
- Set USB permissions as described in the Ubuntu/Linux section.

</details>

---

## Installation

### From PyPI *(Recommended)*

```bash
pip install cfclient
```

On macOS:

```bash
python3 -m pip install cfclient
```

Launch the client:

```bash
cfclient
# or
python3 -m cfclient.gui
```

---

### From Source *(Development)*

```bash
git clone https://github.com/bitcraze/crazyflie-clients-python
cd crazyflie-clients-python

# Basic install
pip install -e .

# With dev tools
pip install -e .[dev]
```

> **Note:** Avoid running `pip` with `sudo`. If prompted for an admin password, use `--user` instead (e.g. `python3 -m pip install --user -e .`).

---

## Development Tools *(Optional)*

### Pre-commit Hooks

```bash
pip install pre-commit
cd crazyflie-clients-python
pre-commit install
pre-commit run --all-files
```

### Editing GUI `.ui` Files

Use **QtCreator** to edit GUI layout files.

- **Windows/Mac:** Download from the [Qt website](https://www.qt.io/download).
- **Ubuntu:** `sudo apt install qtcreator`

### Debugging in VSCode

Add this to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Crazyflie client",
      "type": "python",
      "request": "launch",
      "module": "cfclient.gui"
    }
  ]
}
```

---

<p align="center">
  <sub>© 2026 <a href="https://www.bitcraze.io">Bitcraze AB</a></sub>
</p>
