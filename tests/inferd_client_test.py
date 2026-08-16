"""Exercise nn-inferd end to end: hello, pool via SCM_RIGHTS, real frames."""
import os, socket, struct, sys, time
sys.path.insert(0, "/opt/nn-accel/py"); sys.path.insert(0, "/opt/nn-accel/gen")
import nn_accel_pb2 as pb
import numpy as np

MAGIC = 0x4E4E5031
SLOTS, W, H = 4, 640, 640
SLOT = W * H * 3
HDR = 800

def make_pool():
    """Producer side in Python (the C version is nn_pool_create)."""
    import mmap
    fd = os.memfd_create("nn_pool", 0)
    pg = os.sysconf("SC_PAGESIZE")
    hdr_len = (HDR + pg - 1) // pg * pg
    slot_b = (SLOT + pg - 1) // pg * pg
    total = hdr_len + slot_b * SLOTS
    os.ftruncate(fd, total)
    m = mmap.mmap(fd, total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    struct.pack_into("<8I", m, 0, MAGIC, SLOTS, slot_b, hdr_len, W, H, W*3, 1)
    return fd, m, hdr_len, slot_b

fd, m, hdr_len, slot_b = make_pool()
s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
s.connect("/tmp/nn-inferd.sock")

# hello -> capabilities
s.send(pb.Msg(id=1, hello=pb.HelloReq(client="test", version="0.1")).SerializeToString())
r = pb.Msg(); r.ParseFromString(s.recv(65536))
print("devices:", [(d.id, d.parallelism, d.healthy) for d in r.caps.devices])

# open pool: the fd goes ONCE, with SCM_RIGHTS
spec = pb.Msg(id=2, open_pool=pb.PoolSpec(owner="camtest", slots=SLOTS,
        slot_bytes=slot_b, format=pb.PIX_RGB888, width=W, height=H, stride=W*3))
socket.send_fds(s, [spec.SerializeToString()], [fd])
r = pb.Msg(); r.ParseFromString(s.recv(65536))
pool_id = r.pool_id.id
print("pool id:", pool_id)

def fill(slot, val):
    off = hdr_len + slot * slot_b
    m[off:off+SLOT] = bytes([val]) * SLOT
    struct.pack_into("<I", m, 32 + 4*slot, 2)      # BUSY
    struct.pack_into("<Q", m, 288 + 8*slot, slot)

lat, qs = [], []
t0 = time.time()
N = 8
for i in range(N):
    slot = i % SLOTS
    fill(slot, 100 + i)
    s.send(pb.Msg(id=10+i, infer=pb.InferReq(pool=pool_id, slot=slot, seq=i,
                  ts_ms=int(time.time()*1000), deadline_ms=2000)).SerializeToString())
    resp = pb.Msg(); resp.ParseFromString(s.recv(65536))
    rr = resp.infer_resp
    lat.append(rr.latency_us); qs.append(rr.queue_us)
    st = struct.unpack_from("<I", m, 32 + 4*slot)[0]
    if i == 0:
        print(f"first: dev={rr.device_id} dets={len(rr.dets)} "
              f"latency={rr.latency_us/1000:.0f}ms queue={rr.queue_us}us "
              f"slot_state_after={st} (0=FREE)")
el = time.time() - t0
print(f"{N} inferences in {el:.1f}s = {N/el:.1f}/s | "
      f"median latency {sorted(lat)[N//2]/1000:.0f}ms | max queue {max(qs)}us")

# deadline A: a frame captured 500 ms ago with a 100 ms budget is stale
fill(0, 7)
s.send(pb.Msg(id=99, infer=pb.InferReq(pool=pool_id, slot=0, seq=999,
       ts_ms=int(time.time()*1000) - 500, deadline_ms=100)).SerializeToString())
resp = pb.Msg(); resp.ParseFromString(s.recv(65536))
print("stale frame (500ms old, 100ms budget) -> dropped:", resp.infer_resp.dropped)

# deadline B: a FRESH frame with the same budget must still run — the check
# must not simply drop everything that carries a deadline
fill(1, 8)
s.send(pb.Msg(id=100, infer=pb.InferReq(pool=pool_id, slot=1, seq=1000,
       ts_ms=int(time.time()*1000), deadline_ms=100)).SerializeToString())
resp = pb.Msg(); resp.ParseFromString(s.recv(65536))
print("fresh frame, same budget -> dropped:", resp.infer_resp.dropped,
      "| ran on", resp.infer_resp.device_id or "(none)")

# deadline C: an implausible stamp (clock skew) must NOT cause a drop
fill(2, 9)
s.send(pb.Msg(id=101, infer=pb.InferReq(pool=pool_id, slot=2, seq=1001,
       ts_ms=1, deadline_ms=100)).SerializeToString())
resp = pb.Msg(); resp.ParseFromString(s.recv(65536))
print("skewed clock (ts=1) -> dropped:", resp.infer_resp.dropped, "(want False)")
s.close()
