#!/usr/bin/env python3
"""Latency measured from FILES, not from the playlist date (which is
unreliable: cam0's runs behind, cam1's runs ahead).

  what a player sees = age of the newest segment
                     + the ~3 segments it buffers before starting
"""
import os, re, time, statistics
for cam in ("cam0","cam1","cam3"):
    d = f"/dev/shm/nn-hls-{cam}"
    p = f"{d}/live.m3u8"
    if not os.path.exists(p): continue
    durs = [float(x) for x in re.findall(r"#EXTINF:([0-9.]+)", open(p).read())]
    ts = [os.path.getmtime(os.path.join(d,f)) for f in os.listdir(d) if f.endswith(".ts")]
    if not ts or not durs: continue
    newest_age = time.time() - max(ts)
    seg = statistics.median(durs)
    print(f"{cam}: segment {seg:.1f}s | newest {newest_age:.1f}s old | "
          f"player starts ~{newest_age + 3*seg:.1f}s behind live")

# Run on the media host:  python3 tests/hls_latency_check.py
# Target is <5 s.  Do NOT measure this from EXT-X-PROGRAM-DATE-TIME: that
# date is anchored when ffmpeg starts and only tracks media time, so it
# drifts behind on cam0 and sits ahead on cam1 while the video itself is
# current (see task #96).
