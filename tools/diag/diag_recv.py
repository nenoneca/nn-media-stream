#!/usr/bin/env python3
"""Diag frame receiver — runs on the DEV HOST, the camera connects to it.

Listens on a TCP port; the camera's diag console connects and pushes ISP
pipeline frames.  Every frame is stored verbatim, described in a sidecar, and
(when the format is understood) rendered to a PNG for eyeballing.

WIRE FORMAT — 40-byte header, then the payload:

    off  size  field
      0     4  magic         b"NNDG" — frame start marker
      4    12  description   ASCII, NUL-padded (e.g. "isp_out", "raw", ...)
     16     4  width         pixels
     20     4  height        pixels
     24     4  fourcc        4 chars, e.g. 'RGB3' 'BA81' 'YU12' (also accepted
                             as a little-endian u32 of the same chars)
     28     4  stride        BYTES per row, including any padding
     32     4  slice_height  padded row count — the vertical analogue of stride;
                             planar chroma starts at stride*slice_height
     36     4  data_size     payload bytes that follow
     40  data_size  data

All integers little-endian by default (--big-endian to flip).

FRAMING AND EXPIRY: the receiver never trusts stream position.  It scans forward
for the magic, so a truncated or corrupt frame costs one frame instead of
desyncing the connection forever.  The payload is then read INCREMENTALLY: if no
bytes arrive for --idle seconds (default 3) the frame is declared EXPIRED, the
partial payload is saved anyway (a partial frame still shows geometry and colour
faults), and the receiver resynchronises on the next magic.  This matters because
the sender streams a ~7 MB frame straight off the camera's capture buffer over
Wi-Fi — a stall means the device wedged, not that more data is coming.

WHY STRIDE AND SLICE HEIGHT ARE THE POINT: a consumer that assumes
stride==width skews every row; one that assumes slice_height==height reads the
chroma planes from the wrong offset.  Either mistake shows up as an edge or
colour artifact that looks like a sensor/ISP fault but is pure bookkeeping.
This receiver therefore reconstructs strictly by the header, and shouts when
stride != width*bpp or slice_height != height so the padding is never silent.

Usage:
    python3 diag_recv.py                      # listen on 0.0.0.0:6070
    python3 diag_recv.py --port 7000 --out ~/diagcaps
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import struct
import sys

MAGIC = b"NNDG"

# 4-byte magic + 12-byte description + 6 x u32 (w,h,fourcc,stride,slice,size).
HDR_STRUCT_LE = struct.Struct("<4s12s6I")
HDR_STRUCT_BE = struct.Struct(">4s12s6I")


def fourcc_str(v) -> str:
    """fourcc may arrive as 4 raw chars packed in a u32, either byte order."""
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    b = struct.pack("<I", v & 0xFFFFFFFF)
    s = b.decode("ascii", "replace")
    return s if all(32 <= c < 127 for c in b) else "0x%08X" % v


class Stream:
    """Buffered reader with a per-read idle deadline and magic resync."""

    def __init__(self, sock: socket.socket, idle: float):
        self.sock, self.idle, self.buf, self.eof = sock, idle, bytearray(), False

    def _fill(self) -> bool:
        """One recv bounded by the idle timeout. False = idle expiry or EOF."""
        self.sock.settimeout(self.idle)
        try:
            chunk = self.sock.recv(1 << 16)
        except socket.timeout:
            return False
        if not chunk:
            self.eof = True
            return False
        self.buf += chunk
        return True

    def read(self, n: int) -> bytes:
        """Up to n bytes; short (possibly empty) if the stream went idle."""
        while len(self.buf) < n and self._fill():
            pass
        out, self.buf = bytes(self.buf[:n]), self.buf[n:]
        return out

    def sync(self) -> bool:
        """Advance to the next MAGIC. False if the stream ends/idles first."""
        skipped = 0
        while True:
            i = self.buf.find(MAGIC)
            if i >= 0:
                if i:
                    print("[sync] skipped %d byte(s) of junk" % (skipped + i), flush=True)
                del self.buf[:i]
                return True
            skipped += max(0, len(self.buf) - 3)
            del self.buf[:max(0, len(self.buf) - 3)]   # keep a possible split magic
            if not self._fill():
                return False


def to_png(data: bytes, w: int, h: int, cc: str, stride: int, slice_h: int, path: str) -> str:
    """Render the payload honouring stride/slice_height. Returns a status note."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return "no numpy/PIL — raw only"

    cc = cc.upper().strip("\x00")
    sh = slice_h or h
    # A partial (EXPIRED) payload still shows geometry and colour faults, so for
    # the row-major formats render the rows that DID arrive instead of bailing.
    if stride and cc in ("RGB3", "RGB8", "RGB ", "BA81", "GRBG", "RGGB", "BGGR",
                         "GBRG", "GREY", "Y8  ") and len(data) < stride * h:
        h = len(data) // stride
        if h < 2:
            return "too little data to render (%d B)" % len(data)
    try:
        if cc in ("RGB3", "RGB8", "RGB "):                  # packed RGB888
            need = stride * h
            if len(data) < need:
                return "short payload (%d < %d)" % (len(data), need)
            a = np.frombuffer(data[:need], np.uint8).reshape(h, stride)
            img = a[:, : w * 3].reshape(h, w, 3)
        elif cc in ("BA81", "GRBG", "RGGB", "BGGR", "GBRG"):  # 8-bit Bayer
            need = stride * h
            if len(data) < need:
                return "short payload (%d < %d)" % (len(data), need)
            a = np.frombuffer(data[:need], np.uint8).reshape(h, stride)[:, :w]
            # 2x2 block demosaic (half res) — geometry/colour check, not quality
            b, g0, g1, r = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
            n = min(b.shape[0], r.shape[0]), min(b.shape[1], r.shape[1])
            g = ((g0[: n[0], : n[1]].astype(np.uint16) + g1[: n[0], : n[1]]) // 2)
            img = np.dstack([r[: n[0], : n[1]], g, b[: n[0], : n[1]]]).astype(np.uint8)
        elif cc in ("YU12", "I420", "IYUV", "YV12"):         # planar 4:2:0
            y_sz = stride * sh
            c_stride, c_h = stride // 2, sh // 2
            need = y_sz + 2 * c_stride * c_h
            if len(data) < need:
                return "short payload (%d < %d)" % (len(data), need)
            Y = np.frombuffer(data[:y_sz], np.uint8).reshape(sh, stride)[:h, :w]
            u_off = y_sz
            v_off = y_sz + c_stride * c_h
            if cc == "YV12":
                u_off, v_off = v_off, u_off
            U = np.frombuffer(data[u_off:u_off + c_stride * c_h], np.uint8).reshape(c_h, c_stride)[: h // 2, : w // 2]
            V = np.frombuffer(data[v_off:v_off + c_stride * c_h], np.uint8).reshape(c_h, c_stride)[: h // 2, : w // 2]
            U = U.repeat(2, 0).repeat(2, 1)[:h, :w].astype(np.float32) - 128.0
            V = V.repeat(2, 0).repeat(2, 1)[:h, :w].astype(np.float32) - 128.0
            Yf = Y.astype(np.float32)
            img = np.clip(np.dstack([Yf + 1.402 * V,
                                     Yf - 0.344136 * U - 0.714136 * V,
                                     Yf + 1.772 * U]), 0, 255).astype(np.uint8)
        elif cc == "OUYY":
            # ESP32-P4 native YUV420 == ESP_COLOR_FOURCC_OUYY_EVYY.
            # Each line is w/2 packed 3-byte groups (chroma, Y, Y); the chroma
            # type ALTERNATES per line -- odd lines carry V, even lines carry U
            # (verified against the live encoder output: mean RGB matched to
            # within 0.7 counts).  There are no U/V planes; decoding it as I420
            # yields noise.  BT.709 limited range, matching the S_FMT request.
            need = stride * h
            if len(data) < need:
                return "short payload (%d < %d)" % (len(data), need)
            g = np.frombuffer(data[:need], np.uint8).reshape(h, w // 2, 3).astype(np.float32)
            Y = np.empty((h, w), np.float32)
            Y[:, 0::2] = g[:, :, 1]
            Y[:, 1::2] = g[:, :, 2]
            C = g[:, :, 0]
            # Verified against the live encoder output: EVEN lines carry U,
            # ODD lines carry V (mean RGB matched to within 0.7 counts; the
            # opposite assignment is off by ~14).
            U = np.repeat(np.repeat(C[0::2], 2, 0), 2, 1)[:h, :w]   # even lines
            V = np.repeat(np.repeat(C[1::2], 2, 0), 2, 1)[:h, :w]   # odd  lines
            Yf = (Y - 16.0) * (255.0 / 219.0)
            Cb = (U - 128.0) * (255.0 / 224.0)
            Cr = (V - 128.0) * (255.0 / 224.0)
            img = np.clip(np.dstack([Yf + 1.5748 * Cr,
                                     Yf - 0.1873 * Cb - 0.4681 * Cr,
                                     Yf + 1.8556 * Cb]), 0, 255).astype(np.uint8)
        elif cc in ("GREY", "Y8  "):
            a = np.frombuffer(data[: stride * h], np.uint8).reshape(h, stride)[:, :w]
            img = np.dstack([a, a, a])
        else:
            return "fourcc %r not rendered (raw saved)" % cc
        Image.fromarray(img).save(path)
        return "png ok"
    except Exception as e:                                    # noqa: BLE001
        return "render failed: %r" % (e,)


def handle(conn: socket.socket, addr, args) -> None:
    print("[conn] %s:%d" % addr, flush=True)
    hdr_struct = HDR_STRUCT_BE if args.big_endian else HDR_STRUCT_LE
    st = Stream(conn, args.idle)
    n = 0
    while True:
        if not st.sync():
            print("[conn] %s:%d %s after %d frame(s)"
                  % (addr[0], addr[1], "closed" if st.eof else "idle", n), flush=True)
            return
        raw = st.read(hdr_struct.size)
        if len(raw) < hdr_struct.size:
            print("[!] header truncated (%d/%d) — %s"
                  % (len(raw), hdr_struct.size, "closed" if st.eof else "idle"), flush=True)
            return
        _magic, desc, w, h, cc_raw, stride, slice_h, size = hdr_struct.unpack(raw)
        desc = desc.split(b"\x00", 1)[0].decode("ascii", "replace") or "frame"
        cc = fourcc_str(cc_raw)

        if size > args.max_bytes:
            print("[!] refusing %d-byte payload (>%d); resyncing" % (size, args.max_bytes),
                  flush=True)
            continue

        data = st.read(size)                      # short read == idle/EOF, not fatal
        expired = len(data) < size
        if expired:
            print("[!] frame %d EXPIRED: %d/%d bytes, no data for %.1fs — saving partial"
                  % (n + 1, len(data), size, args.idle), flush=True)
            if st.eof and not data:
                return

        n += 1
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(args.out, "%s_%s_%03d" % (ts, desc.replace("/", "_"), n))
        with open(base + ".bin", "wb") as f:
            f.write(data)

        bpp = {"RGB3": 3, "RGB8": 3, "BA81": 1, "GREY": 1, "YU12": 1, "I420": 1}.get(cc.upper(), 0)
        notes = []
        if cc.upper() == "OUYY":            # w/2 groups of 3 bytes per line
            notes.append("ESP32-P4 native YUV420 (OUYY/EVYY line-packed, not I420)")
        if expired:
            notes.append("EXPIRED: received %d of %d bytes (%.1f%%)"
                         % (len(data), size, 100.0 * len(data) / size if size else 0.0))
        if bpp and stride != w * bpp:
            notes.append("STRIDE PADDING: %d bytes/row vs %d used (%+d)"
                         % (stride, w * bpp, stride - w * bpp))
        if slice_h and slice_h != h:
            notes.append("SLICE PADDING: %d rows vs %d visible (%+d)"
                         % (slice_h, h, slice_h - h))
        png = to_png(data, w, h, cc, stride, slice_h, base + ".png")

        meta = {"description": desc, "width": w, "height": h, "fourcc": cc,
                "fourcc_raw": cc_raw, "stride": stride, "slice_height": slice_h,
                "data_size": size, "received_bytes": len(data), "expired": expired,
                "notes": notes, "render": png, "peer": "%s:%d" % addr}
        with open(base + ".json", "w") as f:
            json.dump(meta, f, indent=1)

        print("[frame %d] %s %dx%d %s stride=%d slice=%d %d B -> %s (%s)"
              % (n, desc, w, h, cc, stride, slice_h, len(data),
                 os.path.basename(base) + ".bin", png), flush=True)
        for note in notes:
            print("           ! " + note, flush=True)
        if st.eof and not st.buf:
            print("[conn] %s:%d closed after %d frame(s)" % (addr[0], addr[1], n), flush=True)
            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="bind address")
    ap.add_argument("--port", type=int, default=6070, help="listen port")
    ap.add_argument("--out", default=os.path.expanduser("~/diagcaps"),
                    help="directory for .bin/.png/.json per frame")
    ap.add_argument("--big-endian", action="store_true",
                    help="header integers are big-endian (default: little)")
    ap.add_argument("--idle", type=float, default=3.0,
                    help="seconds without data before a frame is EXPIRED (default 3)")
    ap.add_argument("--max-bytes", type=int, default=64 << 20,
                    help="reject payloads larger than this (desync guard)")
    ap.add_argument("--once", action="store_true", help="exit after one connection")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    print("listening on %s:%d  ->  %s" % (args.host, args.port, args.out), flush=True)
    print("header: 'NNDG' | 12s desc | u32 w,h,fourcc,stride,slice_height,data_size | data",
          flush=True)
    print("frames idle for >%.1fs are marked EXPIRED and the partial payload kept"
          % args.idle, flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            conn.settimeout(args.idle)
            try:
                handle(conn, addr, args)
            except socket.timeout:
                print("[!] %s:%d timed out" % addr, flush=True)
            except Exception as e:                            # noqa: BLE001
                print("[!] %s:%d error: %r" % (addr[0], addr[1], e), flush=True)
            finally:
                conn.close()
            if args.once:
                return 0
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
