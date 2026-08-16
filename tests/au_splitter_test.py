"""The access-unit splitter in transcoded_client.

A camera sends ~1.4 KB fragments; the transcode service needs whole access
units, one per pool slot.  Getting a boundary wrong is not loud — it shows
up as a decoder quietly discarding most frames — so the rules are pinned
here.

Run: python3 tests/au_splitter_test.py [captured.h264]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "service"))


def _load_boundary():
    """Import just the splitter, without protobuf/gi being present."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "service", "transcoded_client.py")
    src = open(path).read()
    body = src[src.index("    def _au_boundary"):
               src.index("    @staticmethod\n    def _is_kf")]
    body = "\n".join(ln[4:] if ln.startswith("    ") else ln
                     for ln in body.split("\n"))
    ns = {}
    exec(body, ns)          # noqa: S102 - deliberate, keeps the test dep-free
    return ns["_au_boundary"]


boundary = _load_boundary()

SPS = b"\x00\x00\x01\x67ss"
PPS = b"\x00\x00\x01\x68pp"
IDR = b"\x00\x00\x01\x65" + b"I" * 20
SLICE = b"\x00\x00\x01\x41" + b"P" * 20


def test_rules():
    # an access unit that has not ended yet must NOT be cut
    assert boundary(SPS + PPS + IDR) == -1
    # the next slice opens the next access unit
    assert boundary(SPS + PPS + IDR + SLICE) == len(SPS + PPS + IDR)
    # parameter sets belong to the AU that FOLLOWS them
    assert boundary(SLICE + SPS + PPS + IDR) == len(SLICE)
    assert boundary(SLICE + SLICE) == len(SLICE)
    # 4-byte start codes are found at the zero byte, not one past it
    four = b"\x00\x00\x00\x01\x41" + b"P" * 8
    assert boundary(SLICE + four) == len(SLICE)
    print("rules PASS")


def test_real_stream(path):
    """Every AU must carry exactly one VCL NAL, and none may be lost."""
    data = open(path, "rb").read()
    acc, aus = bytearray(), []
    for i in range(0, len(data), 1400):          # uplink-sized fragments
        acc.extend(data[i:i + 1400])
        while True:
            cut = boundary(acc)
            if cut <= 0:
                break
            aus.append(bytes(acc[:cut]))
            acc = bytearray(acc[cut:])
    if acc:
        aus.append(bytes(acc))

    def nal_types(b):
        out, i = [], b.find(b"\x00\x00\x01")
        while i != -1 and i + 3 < len(b):
            out.append(b[i + 3] & 0x1f)
            i = b.find(b"\x00\x00\x01", i + 3)
        return out

    vcl_total = sum(t in (1, 5) for a in aus for t in nal_types(a))
    stream_vcl = sum(t in (1, 5) for t in nal_types(data))
    multi = [a for a in aus if sum(t in (1, 5) for t in nal_types(a)) != 1]
    assert vcl_total == stream_vcl, f"lost slices: {vcl_total} vs {stream_vcl}"
    assert not multi, f"{len(multi)} AUs do not hold exactly one slice"
    print(f"real stream PASS: {len(aus)} AUs, {vcl_total} slices, "
          f"one slice each")


if __name__ == "__main__":
    test_rules()
    sample = sys.argv[1] if len(sys.argv) > 1 else None
    if sample and os.path.exists(sample):
        test_real_stream(sample)
    else:
        print("no capture given — skipping the real-stream check "
              "(grab one from /api/v1/cameras/<id>/ws)")
