# nn media — source SDK (FreeRTOS / ESP-IDF)

The device side that **sends** video into the system. Drop these components into
an ESP-IDF project (add this directory to `EXTRA_COMPONENT_DIRS`).

## Components

- **`nn_sectun`** — the encrypted session (client/initiator side of the NNS1
  protocol in `../docs/protocol.md`). Opens a session over a connected socket
  and encrypts an H.264 byte-stream to the service:
  ```c
  nn_sectun_t s;
  nn_sectun_client_handshake(&s, fd, service_pubkey /* 32B X25519 */);
  nn_sectun_send(&s, h264_chunk, len);   /* repeat per chunk */
  ```
- **`nn_video`** — `nn_video.h`: an 8-byte fragmentation header for carrying the
  H.264 stream across datagram links (e.g. the P4↔C6 SPI link) before it reaches
  the network uplink.

## Crypto provider dependency

`nn_sectun` does the X25519 / HKDF-SHA256 / AES-256-GCM via a crypto provider.
The reference provider is **`nn_prov`** (the camera-system firmware), which owns
the device's provisioned X25519 identity and exposes:

```c
nn_prov_crypto_gen_x25519(pub, priv);
nn_prov_crypto_x25519(priv, peer, out);
nn_prov_crypto_device_x25519(peer, out);     /* uses the device static key */
nn_prov_crypto_hkdf(...);
nn_prov_crypto_aesgcm(...);
nn_prov_get_device_pubkey(out32);
```

To use this SDK standalone, provide a component named `nn_prov` implementing
`<nn_prov/nn_prov_crypto.h>` + `nn_prov_get_device_pubkey()` over your platform's
crypto (e.g. ESP-IDF PSA / mbedTLS).  **TODO:** factor this into a formal
`nn_sectun_crypto_if` interface so the SDK ships without the `nn_prov` name.

## Provenance

Extracted from the nn camera firmware (`media/components/`). The full device
also uses `nn_netstream` (Wi-Fi STA + reconnecting TCP uplink) and `nn_prov`
(BLE provisioning) — included there, not vendored here, as they are app-level.
