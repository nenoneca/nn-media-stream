#!/usr/bin/env python3
"""
Encrypted device control-channel server (the hub side of nn_ctrl).

A device dials in over nn_sectun and serves a request/reply protocol; this
drives it (PING / STATUS).  Needs: pip install nn-media-sink.

  device (nn_ctrl, client) ── encrypted TCP ──▶ ctrl_server.py

Usage: ctrl_server.py [--port 8770] [--keydir DIR] [--poll 5] [--rounds 0]
"""
import argparse
import socket
import time

from nn_media_sink.sectun import SecureSession
from nn_media_sink import keys

OP_PING = 0x01
OP_STATUS = 0x02


def serve_one(conn, priv, poll, rounds):
    sess = SecureSession.accept(conn, priv)
    print(f">> handshake OK; device {sess.device_pub.hex()[:16]}...")
    n = 0
    while rounds == 0 or n < rounds:
        sess.send(bytes([OP_PING]))
        rep = sess.recv()
        print(f"   PING  → 0x{rep[0]:02x} {rep[1:].decode(errors='replace')!r}")
        sess.send(bytes([OP_STATUS]))
        rep = sess.recv()
        print(f"   STATUS→ 0x{rep[0]:02x} {rep[1:].decode(errors='replace')!r}")
        n += 1
        if rounds and n >= rounds:
            break
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--keydir", default="/tmp/ble_hub_data")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=0)
    args = ap.parse_args()

    priv = keys.load_or_generate(args.keydir)
    print(f">> hub X25519 pub: {keys.public_bytes(priv).hex()}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f">> control server (encrypted) on 0.0.0.0:{args.port}")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f">> device control connection from {addr}")
        try:
            serve_one(conn, priv, args.poll, args.rounds)
        except (ConnectionError, ValueError) as e:
            print(f">> control session ended: {e}")
        finally:
            conn.close()
        if args.rounds:
            return


if __name__ == "__main__":
    main()
