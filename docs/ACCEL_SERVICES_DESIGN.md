# Accelerator services: nn-inferd / nn-transcoded — design

Status: **design** (2026-08-08).

## 1. Shape

Camera pipelines stay one process each (fault isolation, independent
restart — see the measured trade-off in the session notes). The two things
worth sharing move OUT into their own services:

```
 nn-video@cam0 ─┐                          ┌─ CIX AIPU (N jobs)
 nn-video@cam1 ─┼─ shm frames + protobuf ─▶│─ TI C7x TIDL
 nn-video@cam3 ─┘        (AF_UNIX)         ├─ Coral EdgeTPU
                                           ├─ Hailo
      same pattern ───▶ nn-transcoded ────▶└─ CPU (ORT)
                                              WAVE5 / CIX venc sessions
```

Why: today every camera process loads its own copy of the model (one is
loaded and never used) and owns its own hardware encoder sessions. A
service can hold ONE instance per device, schedule fairly across cameras,
and recover a wedged device without taking a camera down.

## 2. Zero-copy frame transport

Control messages are protobuf over `AF_UNIX SOCK_SEQPACKET`; **pixels never
travel over the socket**.

* The camera allocates a **frame pool**: `memfd_create` + `ftruncate`,
  mapped once on both sides. The fd is handed to the service with
  `SCM_RIGHTS` at `OpenPool`, so the kernel — not a copy — shares the pages.
* A pool is a ring of fixed-size slots (`slots × slot_bytes`), described
  once: format, width, height, stride, plane offsets.
* `Infer{pool_id, slot, seq, model_id, ts_ms}` is ~40 bytes. The service
  reads the pixels in place and replies with detections (small → socket is
  fine).
* Slot lifecycle is explicit: producer marks a slot BUSY before sending,
  the service releases it in the response (or via `Release` for async).
  A slot is never reused while BUSY, so there is no tear.
* Where the platform allows it, the pool is a **dma-buf** instead of a
  memfd (`/dev/dma_heap/*` exists on both boards, and V4L2/GStreamer can
  export capture buffers): the accelerator then reads the very buffer the
  camera DMA'd into — zero copy AND zero mapping cost.
* Backpressure is a feature: if every slot is BUSY the camera drops the
  frame, which is exactly what the fps cap already does.

Crash safety: a memfd/dma-buf dies when the last fd closes, so a crashed
camera leaks nothing; a crashed service is reconnected to like the sectun
uplink (re-`OpenPool`, resume).

## 3. Device inventory and parallelism

The service enumerates accelerators at startup and advertises them:

```proto
message Device {
  string id = 1;            // "aipu0", "c7x0", "coral0", "hailo0", "cpu"
  string kind = 2;          // AIPU | TIDL | EDGETPU | HAILO | CPU
  uint32 parallelism = 3;   // in-flight jobs this device sustains
  repeated string models = 4;
}
```

Total concurrency `N = Σ parallelism`, and N worker threads pull from one
queue, so a fast device is never idle behind a slow one. Fairness is
per-camera round-robin: one busy camera cannot starve the others, and each
camera has a queue cap (drop-oldest) so a stalled device can't grow memory
without bound.

Models are preloaded per device at startup — a session create is the
expensive step (and on TIDL a mid-flight create is what wedges the C7x).

## 4. Interfaces (proto sketch)

```proto
service Infer {
  rpc Hello(HelloReq) returns (Capabilities);     // devices, formats, models
  rpc OpenPool(PoolSpec) returns (PoolId);        // fd passed via SCM_RIGHTS
  rpc Run(InferReq) returns (InferResp);          // slot -> detections
  rpc Stats(StatsReq) returns (StatsResp);
}
service Transcode {
  rpc Hello(HelloReq) returns (Capabilities);     // encoders, sessions free
  rpc OpenPool(PoolSpec) returns (PoolId);        // NV12 in
  rpc Encode(EncodeReq) returns (EncodeResp);     // -> bitstream slot
}
```

Not gRPC: gRPC cannot pass file descriptors, and fd passing is the whole
point. Plain protobuf framing over SEQPACKET keeps one message = one
datagram, so no length-prefix parsing either.

## 5. Risks / open points

* **Head-of-line blocking** on a device that stops responding — per-device
  watchdog, drain the queue to another device, mark it unhealthy.
* **Container boundary** (the BeagleY app runs inside the edgeai LXC): the
  socket must be bind-mounted, and dma-buf fds cross that boundary fine.
* **Latency**: shm + SEQPACKET is tens of microseconds; the win over
  in-process is scheduling, not speed — measure before/after.
* **Versioning**: capability negotiation in `Hello`; protobuf field numbers
  never reused.
* **Security**: socket mode 0660 + a dedicated group; a pool fd grants
  access to those pages only.

## 6. Rollout

1. `nn_shmpool` (C + Python): memfd/dma-buf pool, slot states, fd passing.
2. `.proto` + generated stubs; `nn-inferd` with CPU/ORT + CIX backends,
   one device, no scheduler.
3. Cameras call it behind the existing `nn_infer` API (new PAL `ipc`), so
   nothing above changes.
4. Scheduler + multi-device inventory + fairness + metrics.
5. `nn-transcoded`; move the HLS re-encode branch onto it.
6. BeagleY app switches from in-process TIDL to the same PAL.
