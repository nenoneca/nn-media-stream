#!/usr/bin/env python3
"""Pure hardware reset of the ESP32-P4 via the UART's EN line (DTR/RTS).

This is the reset esptool performs *after* writing flash, isolated from the
flash write itself: pull EN low, release, let the app boot.  No bootloader
entry, no erase, no programming — so comparing this against the console
`reboot` command (esp_restart) separates "reset path" from "software restart",
and comparing it against a full `idf.py flash` separates it from the write.
"""
import sys, time, serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
s = serial.Serial(port, 115200)
s.setDTR(False)          # IO0 high -> run mode, not download
s.setRTS(True)           # EN low  -> hold in reset
time.sleep(0.15)
s.setRTS(False)          # EN high -> release, boot the app
time.sleep(0.05)
s.close()
print("hardware reset (EN toggled) on", port)
