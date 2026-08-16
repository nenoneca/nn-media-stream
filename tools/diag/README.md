# ISP pipeline diagnostics

Host-side receiver for the camera's pipeline tap. Firmware side lives in
`device/modules/libs/nn_camera/diag/`.

## What can and cannot be observed

The ESP32-P4 ISP is **fixed-function with a single DMA output** — there is no way
to read a buffer between two blocks. The pipeline is:

```
CSI ─► BLC ─► BF ─► LSC ─► Demosaic ─► WBG ─► CCM ─► Gamma ─► RGB2YUV ─► SHARP ─► CROP ─► DMA
```

Two taps exist, selected by the capture pixel format (esp_video keeps the ISP
in-path for SBGGR8 / RGB565 / RGB24 / YUV420 / YUV422P):

| capture format | what you get |
|---|---|
| `SBGGR8` | Bayer domain — before demosaic / WBG / CCM / gamma / sharpen |
| `RGB24`  | RGB domain — after the full RGB chain, before RGB2YUV |
| `YUV420` | production tail — everything applied |

**Per-stage contribution comes from differencing**, not tapping: capture, then
`isp blk <ccm|wb|lsc|bf|sharpen|gamma|demosaic> 0`, capture again, compare.
Caveat measured 2026-08-03: the IPA rewrites `gamma`, `sharpen` and `bf` within
frames, so only `ccm`, `wb` and `lsc` toggle reliably from the console.

## Use

On the dev host:

    python3 media/diag/diag_recv.py --port 6070 --out ~/diagcaps

On the camera console:

    net wifi <ssid> <psk>          # if not already provisioned
    isp send <dev-host-ip> 6070 rgb_full

Each frame lands as three files: `.bin` (verbatim payload), `.json` (every
header field), `.png` (reconstructed **honouring stride and slice_height**).

## Wire format — 40-byte header, little-endian, then the payload

| off | size | field |
|---|---|---|
| 0  | 4 | magic `"NNDG"` — frame start marker |
| 4  | 12 | description, ASCII NUL-padded |
| 16 | 4 | width |
| 20 | 4 | height |
| 24 | 4 | fourcc |
| 28 | 4 | stride (BYTES per row, padding included) |
| 32 | 4 | slice_height (padded row count; planar chroma starts at stride×slice_height) |
| 36 | 4 | data_size |
| 40 | data_size | data |

The receiver **scans for the magic** rather than trusting stream position, so a
truncated frame costs one frame instead of desyncing the connection permanently.
It then reads the payload incrementally: if no bytes arrive for `--idle` seconds
(default 3) the frame is marked **EXPIRED**, the partial payload is saved and
still rendered (row-major formats render the rows that arrived), and the receiver
resynchronises on the next magic.

Stride and slice height come from the driver's own `G_FMT`, so padding is
reported rather than assumed — and the receiver flags it loudly. A consumer that
assumes `stride == width` skews every row; one that assumes
`slice_height == height` reads chroma from the wrong offset. Either looks like a
sensor or ISP fault while being pure bookkeeping.

## Interactive ISP tuning

Console commands (`isp <sub>`), all on the live pipeline — no reflash:

| command | what it sets | range |
|---|---|---|
| `bf <level>` | bayer denoise strength | 2–20 |
| `sharpen <hT> <lT> <hC> <mC>` | sharpen thresholds + coefficients | thresh 0–255, coeff 0–8 |
| `demosaic <ratio>` | demosaic gradient ratio | float |
| `wbg <r> <g> <b>` | hardware white-balance gains | milli, 1000 = unity |
| `wb <r> <b>` / `ccm <9 floats>` | colour via the CCM | matrix −4…4 |
| `blk <name> <0\|1>` | enable/disable a block | see below |
| `hold <mask>` | suppress IPA writes per block | bit0 bf, 1 demosaic, 2 sharpen, 3 gamma, 4 ccm |
| `reset` | re-apply firmware defaults, release all holds | — |
| `send <host> [port] [desc]` | push a frame to `diag_recv.py` | — |
| `ws <0\|1>` | periodic WebSocket push to the hub | on |

**Why `hold` exists.** The IPA recomputes BF / sharpen / gamma / demosaic / CCM
**every frame**, so a console write is overwritten within ~a frame and the
experiment measures nothing. `bf`, `sharpen` and `demosaic` therefore take the
hold for their own block automatically; `hold` is there to set it manually (e.g.
`hold 0x10` to freeze the CCM while tuning colour). `reset` releases everything.

Not exposed on the command line, deliberately: **gamma** (16 (x,y) pairs with a
power-of-2 spacing rule — 32 arguments is worse than a rebuild), **LSC** (needs
calibrated gain tables from a flat-field shoot — a file, not a command), and
**AF windows** (geometry; `focus <0-1023>` already covers the useful part).

## Why LSC is off

`isp_start_lsc()` needs four calibrated gain grids (`gain_r/gr/gb/b`, one gain
per cell). Nothing supplies them: the IMX708 tuning JSON has no `lsc` block, and
`esp_ipa` has no LSC module to generate one. So `lsc_enable` stays false.

This is a real gap for the Wide NoIR — wide optics vignette hardest, and on a
NoIR sensor shading is *colour* shading (IR falls off differently from visible).
LSC is precisely the block for "edges are the wrong colour"; it is switched off
only because nobody has calibrated it.


## Two sinks, one capture loop — turn the hub push off

DIAG builds run `diag_task`, which pushes a full 7.46 MB RGB888 frame to the hub
over WebSocket every `NN_CAMERA_DIAG_INTERVAL_MS`.  That push and the on-demand
`isp send` tap share the **single** capture loop, and one push blocks the loop
for 10-20 s inside a socket write that cannot be interrupted.  Symptom: `isp
send` reports `no frame in 90 s` while `cam stats` shows `csi_done` frozen — the
loop is alive but never returns to the tap.

Run **`isp ws 0`** before capturing.  `isp send` also pauses the push for the
duration of a grab and restores it afterwards, but it still has to wait out a
push already in flight.

## `G_FMT` under-reports stride — the tap corrects it

For RGB24 1920x1296 esp_video returns `bytesperline = 0` (and previously 1920, a
PIXEL count) with `sizeimage = 0`.  Taken literally that tells a receiver each row
is 1920 B when it is really 5760 — a 3x skew that looks exactly like a corrupt
sensor readout.  The tap therefore clamps stride up to the packed minimum for the
fourcc, logs a warning when it does, and falls back to the visible height for
`slice_height`.  Header values are truthful; the driver's are not.


## The ESP32-P4's "YUV420" is NOT I420 — this is why captures looked like noise

`esp_video` maps `V4L2_PIX_FMT_YUV420` onto `CAM_CTLR_COLOR_YUV420`, which is
`ESP_COLOR_FOURCC_OUYY_EVYY` — **O**dd line **U**YY, **E**ven line **V**YY:

```
line 0 (even):  U Y Y | U Y Y | ...   <- w/2 groups of 3 bytes = w/2*3 bytes
line 1 (odd) :  V Y Y | V Y Y | ...
```

Every line is `w/2` packed 3-byte groups of (chroma, Y, Y); the chroma component
alternates per line and is shared across the 2x2 pixel block. **There are no U/V
planes.** Total size is `w*h*3/2`, identical to I420 — which is exactly why the
mislabelling went unnoticed: the byte count, the buffer allocation and
`VIDIOC_QUERYBUF` all validate perfectly while the layout is completely
different. Decoding it as I420 produces noise with a scene-like ghost.

Diagnostics of the mislabelled frame, for future reference:

| symptom | cause |
|---|---|
| adjacent-row correlation peaks at 5760, not 1920 | 2880-byte line, x2 for U/V alternation |
| 3-byte periodicity, harmonics at lag 9/12 | the (C,Y,Y) group |
| "channel 1 == channel 2" at 0.996, channel 0 anti-correlated | the two Y samples vs the chroma |
| "luma plane" shows the scene twice side by side | 1920-byte row = 2/3 of a real line |

The tap therefore reports fourcc **`OUYY`** with `stride = w/2*3` for this format
rather than the driver's `YU12`, and `diag_recv.py` decodes it (BT.709, limited
range, **even lines U / odd lines V** — verified against the live encoder output,
mean RGB within 1.0 count).
