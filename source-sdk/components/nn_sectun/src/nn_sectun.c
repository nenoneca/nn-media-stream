/* SPDX-License-Identifier: Apache-2.0 */
#include "nn_sectun/nn_sectun.h"
#include "nn_prov/nn_prov_crypto.h"
#include "nn_prov/nn_prov.h"

#include "esp_log.h"
#include "lwip/sockets.h"
#include <string.h>

static const char *TAG = "nn_sectun";

#define MAGIC0 'N'
#define MAGIC1 'N'
#define MAGIC2 'S'
#define MAGIC3 '1'
#define VERSION 1
static const uint8_t INFO[] = "nn-sectun-v1";
#define INFO_LEN (sizeof(INFO) - 1)

/* ── socket helpers (handle partial send/recv) ──────────────────────────── */
static esp_err_t write_all(int fd, const uint8_t *p, size_t n)
{
    while (n) {
        int w = send(fd, p, n, 0);
        if (w <= 0) return ESP_FAIL;
        p += w; n -= w;
    }
    return ESP_OK;
}

static esp_err_t read_all(int fd, uint8_t *p, size_t n)
{
    while (n) {
        int r = recv(fd, p, n, 0);
        if (r <= 0) return ESP_FAIL;
        p += r; n -= r;
    }
    return ESP_OK;
}

static void make_nonce(uint8_t nonce[12], uint64_t ctr)
{
    memset(nonce, 0, 4);
    for (int i = 0; i < 8; i++) nonce[4 + i] = (uint8_t)(ctr >> (56 - 8 * i));
}

/* ── handshake ──────────────────────────────────────────────────────────── */
esp_err_t nn_sectun_client_handshake(nn_sectun_t *s, int fd,
                                     const uint8_t peer_pub[32])
{
    uint8_t eph_pub[32], eph_priv[32], dev_pub[32];
    if (nn_prov_crypto_gen_x25519(eph_pub, eph_priv) != ESP_OK) return ESP_FAIL;
    nn_prov_get_device_pubkey(dev_pub);

    uint8_t secret[64];
    if (nn_prov_crypto_x25519(eph_priv, peer_pub, secret) != ESP_OK) goto fail;
    if (nn_prov_crypto_device_x25519(peer_pub, secret + 32) != ESP_OK) goto fail;

    uint8_t okm[64];
    if (nn_prov_crypto_hkdf(secret, 64, eph_pub, 32, INFO, INFO_LEN, okm, 64) != ESP_OK) goto fail;

    memset(s, 0, sizeof *s);
    s->fd = fd;
    memcpy(s->k_tx, okm, 32);        /* client→server */
    memcpy(s->k_rx, okm + 32, 32);   /* server→client */

    uint8_t hello[4 + 2 + 32 + 32];
    hello[0] = MAGIC0; hello[1] = MAGIC1; hello[2] = MAGIC2; hello[3] = MAGIC3;
    hello[4] = VERSION; hello[5] = 0;
    memcpy(hello + 6, eph_pub, 32);
    memcpy(hello + 38, dev_pub, 32);

    memset(secret, 0, sizeof secret);
    memset(eph_priv, 0, sizeof eph_priv);
    memset(okm, 0, sizeof okm);

    if (write_all(fd, hello, sizeof hello) != ESP_OK) { ESP_LOGE(TAG, "hello send failed"); return ESP_FAIL; }
    ESP_LOGI(TAG, "secure session up (peer %02x%02x%02x%02x..)",
             peer_pub[0], peer_pub[1], peer_pub[2], peer_pub[3]);
    return ESP_OK;

fail:
    memset(secret, 0, sizeof secret);
    memset(eph_priv, 0, sizeof eph_priv);
    ESP_LOGE(TAG, "handshake key derivation failed");
    return ESP_FAIL;
}

/* ── records ────────────────────────────────────────────────────────────── */
static esp_err_t send_one(nn_sectun_t *s, const uint8_t *data, size_t len)
{
    uint8_t nonce[12];
    make_nonce(nonce, s->ctr_tx);
    uint8_t ct[NN_SECTUN_RECORD_MAX + 16];
    size_t ct_len = sizeof ct;
    if (nn_prov_crypto_aesgcm(false, s->k_tx, nonce, NULL, 0, data, len,
                              ct, sizeof ct, &ct_len) != ESP_OK) return ESP_FAIL;
    s->ctr_tx++;

    uint8_t hdr[4];
    hdr[0] = (uint8_t)(ct_len >> 24); hdr[1] = (uint8_t)(ct_len >> 16);
    hdr[2] = (uint8_t)(ct_len >> 8);  hdr[3] = (uint8_t)ct_len;
    if (write_all(s->fd, hdr, 4) != ESP_OK) return ESP_FAIL;
    return write_all(s->fd, ct, ct_len);
}

esp_err_t nn_sectun_send(nn_sectun_t *s, const uint8_t *data, size_t len)
{
    while (len) {
        size_t chunk = len > NN_SECTUN_RECORD_MAX ? NN_SECTUN_RECORD_MAX : len;
        if (send_one(s, data, chunk) != ESP_OK) return ESP_FAIL;
        data += chunk; len -= chunk;
    }
    return ESP_OK;
}

esp_err_t nn_sectun_recv(nn_sectun_t *s, uint8_t *out, size_t cap, size_t *out_len)
{
    uint8_t hdr[4];
    if (read_all(s->fd, hdr, 4) != ESP_OK) return ESP_FAIL;
    size_t ct_len = (size_t)hdr[0] << 24 | (size_t)hdr[1] << 16 |
                    (size_t)hdr[2] << 8 | hdr[3];
    if (ct_len < 16 || ct_len > NN_SECTUN_RECORD_MAX + 16) { ESP_LOGE(TAG, "bad record len %u", (unsigned)ct_len); return ESP_FAIL; }

    uint8_t ct[NN_SECTUN_RECORD_MAX + 16];
    if (read_all(s->fd, ct, ct_len) != ESP_OK) return ESP_FAIL;

    uint8_t nonce[12];
    make_nonce(nonce, s->ctr_rx);
    if (nn_prov_crypto_aesgcm(true, s->k_rx, nonce, NULL, 0, ct, ct_len,
                              out, cap, out_len) != ESP_OK) return ESP_FAIL;
    s->ctr_rx++;
    return ESP_OK;
}
