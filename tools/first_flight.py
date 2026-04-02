import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

DEFAULT_ADDR = "E7E7E7E7E7"

def scan_full_uri(timeout_s: float = 5.0) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        found = cflib.crtp.scan_interfaces()

        candidates = []
        for item in found:
            uri = item[0] if isinstance(item, (tuple, list)) and item else item
            uri = (uri or "").strip()
            if not uri:
                continue
            candidates.append(uri)

        for uri in candidates:
            if uri.startswith("radio://") and uri.count("/") >= 5:
                return uri

        for uri in candidates:
            if uri.startswith("radio://") and uri.count("/") == 4:
                return f"{uri}/{DEFAULT_ADDR}"

        time.sleep(0.2)

    raise RuntimeError("No Crazyflie found. Power it on and plug in Crazyradio.")

def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    URI = scan_full_uri(timeout_s=8.0)
    print("Using URI:", URI)

    cf = Crazyflie(rw_cache="./cache")
    with SyncCrazyflie(URI, cf=cf) as scf:
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        with MotionCommander(scf, default_height=0.3) as mc:
            time.sleep(1.0)
            mc.forward(0.3)
            time.sleep(0.5)
            mc.turn_right(90)
            time.sleep(0.5)
            mc.back(0.3)
            time.sleep(0.5)

    print("Done.")

if __name__ == "__main__":
    main()