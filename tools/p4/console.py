import serial, sys, time
port, cmds, wait = sys.argv[1], sys.argv[2].split("|"), float(sys.argv[3])
s = serial.Serial(port, 115200, timeout=0.2)
time.sleep(0.3); s.reset_input_buffer()
for c in cmds:
    if c.strip(): s.write((c + "\r\n").encode()); print(">>> " + c, flush=True)
    t = time.time()
    while time.time() - t < wait:
        d = s.read(4096)
        if d: sys.stdout.write(d.decode("utf8","replace")); sys.stdout.flush()
s.close()
