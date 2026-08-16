# Inference policy & device config channel — design

Status: **design** (2026-08-08). Supersedes the single `infer_mode`
auto/on/off switch, which stays as the top-level enable.

## 1. Problem

Today a camera's detection behaviour is one global switch plus a hard-coded
`min_score` and an ad-hoc alert-class set in `event_engine`. We need, **per
device**:

* which classes may trigger an event video capture (from the classes the
  device's model actually supports — a capability, not a free-text list),
* how many inferences are aggregated before a decision (**default 5**),
* a **start** confidence threshold and a **stop** threshold (`stop <= start`)
  so a detection doesn't flap in and out,

and all of it settable **independently for edge and for service inference**
when a device supports both.

## 2. Model

### 2.1 Capability (device → hub, read-only)

Advertised in the status record and carried through camera registration:

```json
{"infer": {"model": "yolox_s_tidl", "w": 640, "h": 640, "fps": 5,
           "classes": [{"id": 0, "name": "person"}, ...],
           "max_agg": 30, "policy": true}}
```

`policy: true` means the device evaluates policy itself (§4). A device that
only reports raw detections gets its policy evaluated service-side instead —
the schema is identical either way, so the UI never changes.

The service publishes the same shape for its own engine, so the two columns
in the UI come from one code path.

### 2.2 Policy (hub → device / service, read-write)

```json
{"version": 7,
 "engines": {
   "edge":    {"enabled": true,  "classes": {...}},
   "service": {"enabled": false, "classes": {...}}
 }}
```

with each class entry:

```json
"person": {"capture": true, "agg": 5, "start": 0.60, "stop": 0.40}
```

* `capture` — this class may start/extend an event video.
* `agg` — aggregation window length in inferences (default **5**, `1..max_agg`).
* `start` — aggregated confidence that enters detect state.
* `stop` — aggregated confidence below which detect state ends.
  **Invariant `stop <= start`,** clamped server-side on write (a UI can't
  produce an invalid pair, and an API client gets its value corrected, not
  rejected).

Unlisted classes are inert: reported for display, never capture.

### 2.3 Aggregation semantics (must be identical in C and Python)

For each class, keep a ring of the last `agg` inference results. Each result
contributes that class's **highest confidence in that frame**, or `0` if the
class was absent. Aggregated confidence = **arithmetic mean of the ring**
(mean, not max: a single lucky frame must not trigger a recording, which is
the whole point of aggregating).

**Update is O(1)** — a running sum, never a re-scan of the window:

```
sum += new - ring[head];      /* value leaving the window */
ring[head] = new;             /* push back */
head = (head + 1) % N;
agg  = sum / N;
```

The ring holds **`conf_x1000` integers** (the wire format's own units), so
`sum` is an exact `int32` and the incremental add/subtract can never drift —
a float running sum accumulates rounding error over millions of updates and
would slowly desync the C and Python evaluators from each other. Thresholds
are compared in the same integer units (`start_x1000`, `stop_x1000`), so the
decision is exact and identical on both sides.

State machine per class:

```
idle   --(agg >= start)--> detect      → start/extend event capture
detect --(agg <  stop )--> idle        → event may close (existing quiet
                                          and max-length rules still apply)
```

Ring is filled with zeros at start, so a class needs genuinely sustained
confidence to fire. Changing `agg` resets that class's ring.

## 3. Storage & distribution

```
hub DB (authoritative, per camera)
   │  PUT /api/v1/cameras/{cam}/inference
   ▼
media service (applies its own "service" engine policy; caches to keydir)
   │  'C' config record on the existing encrypted stream (versioned)
   ▼
device (applies "edge" policy; persists to NVS/KV; acks with version)
```

The hub is the source of truth; the service is the only thing that talks to
the device, exactly as it already is for bitrate/GOP control.

## 4. Device config channel

No new socket: the encrypted record stream already carries typed records and
is the connection whose liveness we care about. New types:

| type | dir | payload |
|------|-----|---------|
| `'C'` 0x43 | service → device | JSON: `{v, engine:"edge", classes:{...}, sink:{host,port,pub}}` |
| `'K'` 0x4B | device → service | JSON ack: `{v, applied:true, err?}` |
| `'H'` 0x48 | device → service | heartbeat: `{up_s, fps, drops, v}` (~40 B, 10 s) |

* **Config is versioned.** The device stores `v` with the config; on connect
  it reports its `v` in the status record and the service pushes only if the
  hub's version is newer. Idempotent, so a reconnect storm costs nothing.
* **`sink`** = media service host/port/pubkey, i.e. the device can be
  *re-pointed* at another service without a reflash or console — the
  "media service sink settings" requirement.
* **Heartbeat** gives per-device liveness independent of video flow (a camera
  can be connected but not streaming), and feeds the hub's online state.
* The existing binary `0xC7` control (bitrate/GOP/IDR) stays as-is: it is a
  hot path and must not be JSON.

## 5. Evaluator placement

One implementation, two bindings:

* `nn-modules/modules/libs/nn_infer/src/policy.c` — portable C (nn_osal only),
  used by the Linux app and the ESP32-P4 app.
* `nn-media-stream/service/infer_policy.py` — the same state machine for the
  service engine, with a shared fixture file of input→expected transitions so
  both stay honest.

## 6. Rollout order

1. Schema + hub storage/API + validation (`stop <= start`, `agg` bounds).
2. Portable evaluator + service adoption (replaces `min_score` and the
   hard-coded alert classes in `event_engine`).
3. Config channel: records, versioning, persistence, ack — Linux camera
   first (fast iteration), then ESP32-P4.
4. UI: per-camera page with edge/service columns and a class table.

Backward compatibility: absent policy = today's behaviour (single threshold,
person/cat/dog capture), so nothing breaks mid-rollout.

## 7. Open questions

* Do we ever want an **OR across classes** for one event (e.g. person *or*
  dog) versus one event per class? Assumption: one event, any capturing class
  can start/extend it; the event's class list records what fired.
* Should the service be able to *override* a device's edge policy result for
  the same class (dual inference), or is `infer_mode` enough? Assumption:
  `infer_mode` decides who owns detection; both never evaluate at once.
