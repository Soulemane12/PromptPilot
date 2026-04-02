import cflib.crtp

cflib.crtp.init_drivers(enable_debug_driver=False)
print("Scanning...")
for uri in cflib.crtp.scan_interfaces():
    print(uri)

