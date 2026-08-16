#!/usr/bin/env python3
"""
Decrypting receiver: terminate a device's nn_sectun stream and write the raw
H.264 (Annex-B) to a file/stdout (so ffmpeg/GStreamer can consume it).  A debug
alternative to the full video service.  Needs: pip install nn-media-sink.

  device (encrypt) ── TCP ──▶ secure_receiver.py (decrypt) ──▶ raw .h264

Usage: secure_receiver.py [--port 8888] [--out FILE|-] [--keydir DIR]
"""
import argparse
import socket
import sys

from nn_media_sink.sectun import SecureSession
from nn_media_sink import keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--out", default="secure_cam.h264")
    ap.add_argument("--keydir", default="/tmp/ble_hub_data")
    ap.add_argument("--max-bytes", type=int, default=0)
    args = ap.parse_args()

    priv = keys.load_or_generate(args.keydir)
    print(f">> stream service X25519 pub: {keys.public_bytes(priv).hex()}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f">> listening (encrypted) on 0.0.0.0:{args.port}")
    out = sys.stdout.buffer if args.out == "-" else open(args.out, "wb")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f">> device connected from {addr}")
        try:
            sess = SecureSession.accept(conn, priv)
            print(f">> handshake OK; device {sess.device_pub.hex()[:16]}...")
            total = 0
            while True:
                data = sess.recv()
                out.write(data); out.flush(); total += len(data)
                if total % (256 * 1024) < len(data):
                    print(f"   decrypted {total // 1024} KB")
                if args.max_bytes and total >= args.max_bytes:
                    print(f">> reached {total} bytes; done"); return
        except (ConnectionError, ValueError) as e:
            print(f">> session ended: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
