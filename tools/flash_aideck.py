#!/usr/bin/env python3
import sys
import cflib.crtp
from cflib.bootloader import Bootloader, Target
from cflib.bootloader.boottypes import TargetTypes

ORIGINAL_BIN = '/Users/soulemanesow/Desktop/drone/PromptPilot/aideck_gap8_wifi_img_streamer_with_ap.bin'
JPEG_IMG = '/Users/soulemanesow/Desktop/drone/aideck-gap8-examples/examples/other/wifi-img-streamer/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img'
URI = 'radio://0/80/2M'

cflib.crtp.init_drivers()

class FakeNRF51Target:
    start_page = 108  # s130 soft device
    version = None
    flash_pages = 232
    page_size = 1024

def progress(msg, percent):
    print(f'  {percent}% - {msg}')

mode = sys.argv[1] if len(sys.argv) > 1 else 'jpeg'
firmware = ORIGINAL_BIN if mode == 'restore' else JPEG_IMG

print(f'Mode: {mode}')
print('Connecting to drone (warm boot)...')
bl = Bootloader(URI)
bl.progress_cb = progress
bl.start_bootloader(warm_boot=True)

# Inject fake NRF51 target if missing (handles crashing GAP8 firmware)
if TargetTypes.NRF51 not in bl._cload.targets:
    bl._cload.targets[TargetTypes.NRF51] = FakeNRF51Target()

print('Connected. Flashing firmware...')
bl.flash(firmware, [Target('deck', 'bcAI:gap8', 'fw', [], [])], boot_delay=5.0)

print('Flash complete! Resetting drone...')
bl.reset_to_firmware()
bl.close()
print('Done. Power cycle the drone and reconnect to WiFi.')
