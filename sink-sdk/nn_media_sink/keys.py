"""X25519 identity-key helpers for the sink SDK (self-contained — only needs
`cryptography`).  Wire/format-compatible with the hub's key store so a service
can decrypt streams from devices provisioned against the hub key:
a raw 32-byte private key at <dir>/<name>."""
from __future__ import annotations
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization

DEFAULT_KEY_NAME = "hub_enc_key.raw"


def load_or_generate(key_dir, name: str = DEFAULT_KEY_NAME) -> X25519PrivateKey:
    """Load a raw 32-byte X25519 private key from key_dir/name, or create it."""
    path = Path(key_dir) / name
    if path.exists():
        return X25519PrivateKey.from_private_bytes(path.read_bytes())
    key = X25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption()))
    return key


def public_bytes(key: X25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
