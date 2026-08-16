"""
nn_media_sink — the receive/consume ("sink") side of the nn media streaming
system, intended to run in the hub.

  - sectun  : terminate the nn_sectun secure session (decrypt a device stream)
  - keys    : X25519 identity-key load/generate (cryptography only)
  - gateway : re-publish an RTP/H.264 stream to end users over WebRTC (aiortc)

`sectun` and `keys` need only `cryptography`.  `gateway` additionally needs
`aiortc`, `av` and `aiohttp` and is imported lazily so the lighter pieces work
without it.
"""
from . import keys          # noqa: F401
from .sectun import SecureSession  # noqa: F401

__all__ = ["keys", "SecureSession"]
__version__ = "0.1.0"
