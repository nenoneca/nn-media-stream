# nn media streaming — wire protocols

Two contracts span the source SDK (device, FreeRTOS/C) and the sink SDK
(hub, Python).

## 1. `nn_sectun` secure session ("NNS1")

An authenticated, encrypted session over a connected TCP socket, keyed by the
peers' provisioned X25519 identities (X25519 + HKDF-SHA256 + AES-256-GCM).
The **source** is the client/initiator; the **sink** (video service / control
server) is the responder.

### Handshake (client → server, once, right after connect)

```
"NNS1"        4 bytes  magic
u8 version =  1
u8 flags   =  0
eph_pub[32]            client ephemeral X25519 public key
dev_pub[32]            client (device) static X25519 public key
```

### Key schedule (both sides derive identically)

```
ss_e  = X25519(eph,  peer_static)     # == X25519(server_priv, eph_pub)
ss_s  = X25519(dev,  peer_static)     # == X25519(server_priv, dev_pub)
okm   = HKDF-SHA256(ss_e || ss_s, salt = eph_pub, info = "nn-sectun-v1", len = 64)
k_c2s = okm[0..32]    # client → server
k_s2c = okm[32..64]   # server → client
```

### Records (either direction, after the handshake)

```
u32 BE  ct_len            # AES-256-GCM ciphertext length (includes the 16B tag)
ct[ct_len]
```

Nonce = `4 zero bytes || u64 BE counter`, a per-direction counter starting at 0
and incremented per record (not transmitted — TCP preserves order). AAD empty.
The stream is chunked into records of ≤ 4096 plaintext bytes.

Implementations: `source-sdk/components/nn_sectun` (C, encrypt/client) and
`sink-sdk/nn_media_sink/sectun.py` (Python, decrypt/responder).

## 2. Video framing (`nn_video`)

Inside the secure session the payload is a raw H.264 Annex-B byte-stream
(start-code delimited NAL units). When the device reaches the sink over an
intermediate link that needs datagram framing (e.g. the P4→C6 SPI link), it is
fragmented with `nn_video.h`:

```
nn_vid_hdr_t (8 bytes): magic 'V' (0x56), flags {KEY,START,END}, seq, frag, nfrag
payload (≤ 1016 bytes)
```

The H.264 NAL stream is self-delimiting, so the sink can re-chunk freely.

## 3. Service → hub hand-off (RTP/UDP)

The video streaming service decrypts the device stream and re-publishes it to a
consumer by appending a GStreamer branch:

```
tee. ! queue ! rtph264pay config-interval=-1 pt=96 ! udpsink host=<hub> port=<n>
```

i.e. standard RTP/H.264 (RFC 6184, payload type 96, 90 kHz). The hub ingests it
with aiortc and re-publishes over WebRTC. WebRTC is done entirely on the hub.
