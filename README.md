# nn-media-stream

Real-time camera video streaming for the nn system: an encrypted device→hub
H.264 pipeline that ends in browser-playable WebRTC.

```
┌─ source SDK (FreeRTOS / ESP-IDF, C) ─┐   ┌─ service ─┐   ┌─ sink SDK (hub, Python) ─┐
│ camera → H.264 → nn_sectun (encrypt) │──▶│ GStreamer │──▶│ aiortc → WebRTC → browser │
│        nn_video framing              │   │ tee + RTP │   │  (decrypt is in service)  │
└──────────────────────────────────────┘   └───────────┘   └───────────────────────────┘
        TCP (encrypted, NNS1)               RTP/UDP                WebRTC
```

## Components

| Dir | Runs on | What |
|-----|---------|------|
| `source-sdk/` | device (FreeRTOS / ESP-IDF) | C SDK to **send** video: `nn_sectun` (secure session, client/encrypt) + `nn_video` (H.264 framing). Drop the components into an ESP-IDF project. |
| `service/` | a Linux host with GStreamer | The **video streaming service**: terminates the device's encrypted session, runs `appsrc ! tee`, and appends RTP/UDP branches on request. |
| `sink-sdk/` | the hub | Python package `nn_media_sink`: `sectun` (decrypt), `keys` (X25519), `gateway` (RTP→WebRTC via aiortc). |
| `tools/` | dev/test | `secure_receiver.py`, `ctrl_server.py`, `webrtc_test_client.py`. |
| `docs/protocol.md` | — | The NNS1 secure-session + framing + RTP contract. |

## Quick start (host/hub side)

```bash
pip install ./sink-sdk[webrtc]          # nn_media_sink + aiortc/av/aiohttp

# 1) video streaming service (needs system GStreamer / python3-gi)
python3 service/video_service.py --stream-port 8888 --control-port 8899 --keydir <keydir>

# 2) hub WebRTC gateway
python3 -m nn_media_sink.gateway --port 8769 --service-url http://127.0.0.1:8899

# 3) open a stream + view
curl -XPOST http://<hub>:8769/api/v1/cameras/cam1/stream -d '{"protocol":"webrtc"}'
# open the returned view_url (/webrtc/<sid>) in a browser
```

The hub's REST app mounts the same routes via `setup_video_routes()` (in the
nn-hub repo, opt-in by `NN_VIDEO_SERVICE_URL`).

## Verifying a new machine

`service/test_pipeline.py` runs the whole pipeline against a synthetic H.264 stream —
no camera needed — and is the first thing to run after moving the media server to new
hardware. It starts its own instance in a temp dir on alternate ports, so it is safe
to run beside a live service.

```bash
python3 service/test_pipeline.py                             # decode/HLS/snapshot/event
python3 service/test_pipeline.py --model ~/models/yolox_s.cix   # + NPU inference
python3 service/test_pipeline.py --model ~/models/yolox_s.onnx \
        --image bus.jpg --expect person,bus                  # + detection accuracy
```

It checks HW decode, HLS segment production and geometry, snapshot decode, that motion
arms YOLO and inference runs (reporting ms/frame), and that an event muxes to mp4 and
uploads to the hub. Add `--enc-qp 36` on the CIX P1 (its encoder ignores `video_bitrate`);
omit it on the beagle, whose WAVE5 honours bitrate. Exit code 0 = all checks passed.

It does NOT substitute for the real camera: the fixture is conformant x264, whereas the
ESP32-P4's bitstream is not, and the encrypted ingest, pairing, `/webrtc` and A/V sync
are all bypassed by `--test-file`.

## Status

Validated end-to-end on hardware: live OV5647 camera → encrypted uplink →
service → hub → WebRTC client decoded live 800×800 frames. Service↔hub is
RTP/UDP; WebRTC is entirely on the hub (aiortc). See `docs/protocol.md`.
