/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

/*
 * nn_sectun — an authenticated, encrypted session over a connected TCP socket,
 * reusing the device's provisioned X25519 identity (via nn_prov_crypto).  Used
 * by the video uplink (nn_netstream) and the hub control channel (nn_ctrl).
 *
 * Wire protocol "NNS1" (client = the C6, server = the stream/control service):
 *
 *   Handshake (client → server, once, right after connect):
 *     "NNS1"            4B magic
 *     u8 version = 1
 *     u8 flags = 0
 *     eph_pub[32]       client ephemeral X25519 public key
 *     dev_pub[32]       client (device) static X25519 public key
 *
 *   Key schedule (both sides compute identically):
 *     ss_e   = X25519(eph,  peer_static)      // == X25519(server_priv, eph_pub)
 *     ss_s   = X25519(dev,  peer_static)      // == X25519(server_priv, dev_pub)
 *     okm    = HKDF-SHA256(ss_e || ss_s, salt=eph_pub, info="nn-sectun-v1", 64)
 *     k_c2s  = okm[0..32]    (client→server)
 *     k_s2c  = okm[32..64]   (server→client)
 *
 *   Records (either direction, after handshake):
 *     u32 BE  ct_len           // AES-256-GCM ciphertext length (incl 16B tag)
 *     ct[ct_len]
 *   Nonce = 4 zero bytes || u64 BE per-direction counter (starts 0, ++/record;
 *   not transmitted — TCP keeps order).  AAD = empty.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define NN_SECTUN_RECORD_MAX  4096   /* max plaintext per record */
#define NN_SECTUN_OVERHEAD    (4 + 16)  /* length prefix + GCM tag */

typedef struct {
    int      fd;
    uint8_t  k_tx[32];   /* our send key   (client: k_c2s) */
    uint8_t  k_rx[32];   /* our recv key   (client: k_s2c) */
    uint64_t ctr_tx;
    uint64_t ctr_rx;
} nn_sectun_t;

/* Perform the client handshake on an already-connected socket `fd`, deriving
 * the session keys against `peer_pub` (the server's static X25519 pubkey).
 * Sends the HELLO.  Returns ESP_OK on success. */
esp_err_t nn_sectun_client_handshake(nn_sectun_t *s, int fd,
                                     const uint8_t peer_pub[32]);

/* Encrypt `len` bytes and write them as one or more records (split at
 * NN_SECTUN_RECORD_MAX).  Blocking; returns ESP_FAIL on socket/crypto error. */
esp_err_t nn_sectun_send(nn_sectun_t *s, const uint8_t *data, size_t len);

/* Read and decrypt exactly one record into `out` (capacity `cap`).  Blocking;
 * *out_len receives the plaintext length.  ESP_FAIL on socket/auth error. */
esp_err_t nn_sectun_recv(nn_sectun_t *s, uint8_t *out, size_t cap, size_t *out_len);

#ifdef __cplusplus
}
#endif
