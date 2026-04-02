import cflib.crtp

print("Initializing drivers...")
cflib.crtp.init_drivers()

print("Scanning for Crazyflie drones...")
interfaces = cflib.crtp.scan_interfaces()

if interfaces:
    print(f"\n Found {len(interfaces)} drone(s):")
    for i in interfaces:
        print(f"  → {i[0]}")
    print("\nYou're ready to fly!")
else:
    print("\n No drones found. Make sure:")
    print("  1. Crazyradio is plugged in")
    print("  2. Crazyflie drone is powered on")
