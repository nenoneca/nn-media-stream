#!/usr/bin/env python3
"""Build report.html over every sample in this directory.

Decodes each capture according to the fourcc in its .json sidecar, measures the
left-edge band, writes a thumbnail, and emits a single self-describing report.

Run:  python3 make_report.py
"""
from __future__ import annotations
import glob, json, os, html, datetime
import numpy as np
from PIL import Image

W, H = 1920, 1296
THUMBS = "thumbs"
os.makedirs(THUMBS, exist_ok=True)


def decode(path: str, cc: str, w: int, h: int):
    """Return an RGB uint8 array, or None if the payload is unusable."""
    d = np.fromfile(path, np.uint8)
    cc = (cc or "").upper().strip("\x00")
    try:
        if cc in ("RGB3", "RGB8"):
            n = w * h * 3
            if d.size < n:
                return None
            return d[:n].reshape(h, w, 3)
        if cc == "OUYY" or cc == "YU12":
            # ESP32-P4 native YUV420: per line, w/2 groups of (chroma, Y, Y);
            # even lines carry U, odd lines V.  YU12 captures predate the fix
            # and are the SAME layout, merely mislabelled.
            n = w // 2 * 3 * h
            if d.size < n:
                return None
            g = d[:n].reshape(h, w // 2, 3).astype(np.float32)
            Y = np.empty((h, w), np.float32)
            Y[:, 0::2] = g[:, :, 1]
            Y[:, 1::2] = g[:, :, 2]
            C = g[:, :, 0]
            U = np.repeat(np.repeat(C[0::2], 2, 0), 2, 1)[:h, :w]
            V = np.repeat(np.repeat(C[1::2], 2, 0), 2, 1)[:h, :w]
            Yf = (Y - 16.0) * (255.0 / 219.0)
            Cb = (U - 128.0) * (255.0 / 224.0)
            Cr = (V - 128.0) * (255.0 / 224.0)
            return np.clip(np.dstack([Yf + 1.5748 * Cr,
                                      Yf - 0.1873 * Cb - 0.4681 * Cr,
                                      Yf + 1.8556 * Cb]), 0, 255).astype(np.uint8)
        if cc in ("BA81", "BGGR", "GRBG", "RGGB", "GBRG"):
            n = w * h
            if d.size < n:
                return None
            a = d[:n].reshape(h, w)
            b, g0, g1, r = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
            k = min(b.shape[0], r.shape[0]), min(b.shape[1], r.shape[1])
            gg = (g0[:k[0], :k[1]].astype(np.uint16) + g1[:k[0], :k[1]]) // 2
            return np.dstack([r[:k[0], :k[1]], gg, b[:k[0], :k[1]]]).astype(np.uint8)
    except Exception:
        return None
    return None


def measure(rgb: np.ndarray):
    """Left-edge band width (px) and supporting numbers."""
    a = rgb.astype(float)
    L = a.mean(2)
    g = np.abs(np.diff(L, axis=1)).mean(0)
    med = float(np.median(g))
    width = 0
    for c in range(min(300, len(g))):
        if g[c] > med * 5:
            width = c + 1
    bg = a[:, :, 2] - a[:, :, 1]
    mid = a.shape[1] // 2
    dbg = float(bg[:, :32].mean() - bg[:, mid - 50:mid + 50].mean())
    return dict(width=width, edge=float(g[:16].mean()), med=med,
                luma=float(L.mean()), dbg=dbg)


# Experiment groups: prefix -> (title, what it tested, verdict)
GROUPS = [
    ("rgb_full", "First successful tap captures", "RGB888 via the on-demand tap once the PSRAM copy and hub-push contention were fixed.", ""),
    ("yuvok",  "YUV420, correct decode", "First captures decoded with the OUYY/EVYY layout.", "ok"),
    ("verify", "YUV420 verified against live", "Rendered capture vs the live encoder output.", "ok"),
    ("ok",     "OUYY header end-to-end", "Receiver rendering the P4 layout unaided.", "ok"),
    ("yuv",    "YUV420 mislabelled as I420", "16 captures announced YU12; decoded as I420 they are noise.", "bad"),
    ("cp",     "Snapshot tap (copy, not hold)", "Ruled out buffer-lifetime as the cause of bad YUV.", ""),
    ("inv",    "Cache invalidation", "Ruled out DMA/cache coherency.", ""),
    ("t",      "Band vs uptime (IPA free)", "Time series from cold boot.", ""),
    ("h",      "IPA sharpen held", "hold 4 — looked like a fix; later disproven.", "retracted"),
    ("fix",    "Sharpen hold shipped", "Verified at 60/120/180 s — under-powered.", "retracted"),
    ("v",      "Verification at 85 min", "3/6 banded: the sharpen fix did NOT hold.", "bad"),
    ("h31_",   "All IPA blocks frozen", "hold 31 — 6/8 still banded. IPA exonerated.", "key"),
    ("h4_",    "Sharpen only (control arm)", "2/8 banded. Interleaved against hold 31.", "key"),
    ("raw",    "RAW Bayer, AE saturated", "gain pegged 25600, frame flat 253 — unusable.", ""),
    ("g",      "RAW gain sweep", "Finding a usable manual exposure.", ""),
    ("e",      "RAW exposure sweep", "gain=8 exposure=200 gives a clean RAW frame.", ""),
    ("rb",     "RAW Bayer, pinned exposure", "0/8 banded — no anomaly before the colour pipeline.", "key"),
    ("cg",     "RGB888, pinned exposure", "7/8 banded at the same exposure (post-FLASH reset).", "key"),
    ("base",   "RGB888 baseline (post-reboot)", "0/6 banded — contradicts 'cg' under identical settings.", "bad"),
    ("nbf",    "BF disabled", "6/6 banded.", ""),
    ("nsh",    "Sharpen disabled", "5/6 banded.", ""),
    ("nccm",   "CCM disabled", "5/6 banded.", ""),
    ("ngam",   "Gamma disabled", "6/6 banded.", ""),
    ("blk_",   "First per-block probe", "Single samples — superseded.", "retracted"),
    ("off",    "Hub push disabled", "Early tap plumbing.", ""),
    ("hv",     "Hardened open_capture", "Format/geometry validation added.", ""),
    ("final",  "OUYY header rollout", "", ""),
    ("fin",    "OUYY header rollout", "", ""),
    ("ouyy",   "OUYY decode validation", "U/V order determined against live.", ""),
]


def group_of(tag: str) -> str:
    best = ""
    for pref, *_ in GROUPS:
        if tag.startswith(pref) and len(pref) > len(best):
            best = pref
    return best


def main() -> None:
    rows = []
    for b in sorted(glob.glob("2026*.bin")):
        base = b[:-4]
        j = base + ".json"
        meta = {}
        if os.path.exists(j):
            try:
                meta = json.load(open(j))
            except Exception:
                meta = {}
        cc = meta.get("fourcc", "")
        w = int(meta.get("width", W) or W)
        h = int(meta.get("height", H) or H)
        if not cc.strip():
            # No sidecar (a few frames predate a receiver fix): infer from size.
            # The three payload sizes at 1920x1296 are unambiguous.
            cc = {7464960: "RGB3", 3732480: "OUYY", 2488320: "BA81"}.get(
                os.path.getsize(b), "")
        stem = os.path.basename(base)
        tag = stem.split("_", 1)[1].rsplit("_", 1)[0] if "_" in stem else stem
        rgb = decode(b, cc, w, h)
        thumb = os.path.join(THUMBS, stem + ".jpg")
        m = None
        if rgb is not None and rgb.size:
            m = measure(rgb)
            if not os.path.exists(thumb):
                Image.fromarray(rgb).resize((320, max(1, int(320 * rgb.shape[0] / rgb.shape[1])))).save(thumb, quality=82)
        rows.append(dict(stem=stem, tag=tag, cc=cc, w=w, h=h,
                         size=os.path.getsize(b), thumb=thumb if m else None,
                         grp=group_of(tag), **(m or {})))
        print(".", end="", flush=True)
    print()

    by = {}
    for r in rows:
        by.setdefault(r["grp"], []).append(r)

    badge = {"key": ("KEY RESULT", "#1a7f37"), "bad": ("CONTRADICTS", "#b35900"),
             "retracted": ("RETRACTED", "#a40e26"), "ok": ("VERIFIED", "#1a7f37"), "": ("", "")}

    out = []
    A = out.append
    A(f"""<meta charset="utf-8"><title>ESP32-P4 ISP left-edge band — investigation report</title>
<style>
:root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --mut:#666; --line:#e2e2e2; --card:#fafafa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8e8; --bg:#161616; --mut:#9a9a9a; --line:#333; --card:#1e1e1e; }} }}
body {{ font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--fg); background:var(--bg);
        max-width:1180px; margin:0 auto; padding:28px 20px 80px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:36px 0 10px; padding-top:12px; border-top:1px solid var(--line); }}
h3 {{ font-size:16px; margin:22px 0 6px; }}
.sub {{ color:var(--mut); margin:0 0 22px; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }}
th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
th {{ font-weight:600; color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
.badge {{ display:inline-block; font-size:11px; font-weight:700; padding:2px 7px; border-radius:3px;
          color:#fff; letter-spacing:.03em; vertical-align:middle; margin-left:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(168px,1fr)); gap:10px; margin:12px 0 4px; }}
.cell {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px; font-size:11px; color:var(--mut); }}
.cell img {{ width:100%; display:block; border-radius:3px; }}
.cell .w {{ font-weight:700; }}
.band {{ color:#c0392b; }} .clean {{ color:#1a7f37; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 18px; margin:14px 0; }}
code {{ background:var(--card); padding:1px 5px; border-radius:3px; font-size:13px; }}
ul {{ margin:8px 0; padding-left:20px; }} li {{ margin:4px 0; }}
.scroll {{ overflow-x:auto; }}
</style>
<h1>ESP32-P4 ISP — left-edge band investigation</h1>
<p class="sub">All {len(rows)} captures collected on this host, decoded and measured.
Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Raw payloads remain in this directory as <code>.bin</code>
with a <code>.json</code> sidecar; thumbnails in <code>thumbs/</code>.</p>

<div class="card">
<b>Band metric.</b> Every frame is decoded to RGB, then the mean column-to-column luma gradient is
computed. The band width is the last column (scanning 0&ndash;299) whose gradient exceeds
5&times; the frame median. A clean frame scores <b>0</b>; banded frames here score 15&ndash;207&nbsp;px.
The same metric is applied to every format so conditions are comparable.
</div>

<h2>Where the investigation stands</h2>
<h3>Established</h3>
<ul>
<li>The P4's <code>V4L2_PIX_FMT_YUV420</code> is <b>not I420</b>. It is <code>OUYY_EVYY</code>:
each line is <i>w</i>/2 packed 3-byte (chroma, Y, Y) groups, chroma type alternating per line,
<b>no U/V planes</b>. Same byte count as I420, which is why every size check passed while the
layout differed. Decoded correctly, a capture matches the live encoder output to <b>1.0 RGB count</b>.</li>
<li>The band is <b>intermittent</b>, width 15&ndash;207&nbsp;px, and is <b>random colour speckle</b> &mdash;
not displaced scene content.</li>
</ul>
<h3>Ruled out</h3>
<ul>
<li><b>WB / CCM tuning</b> &mdash; band survives unity white balance.</li>
<li><b>Sensor digital crop</b> &mdash; <code>DIG_CROP_X_OFFSET</code> = 0x00C0 = 192, correctly centred.</li>
<li><b>Buffer overrun</b> &mdash; <code>QUERYBUF</code> length is exactly one frame.</li>
<li><b>A wrap of the right-hand image</b> &mdash; band vs right edge correlates +0.009 (same row) /
+0.023 (previous row) against +0.006 for an arbitrary strip.</li>
<li><b>The IPA's parameter updates</b> &mdash; freezing all five blocks still gives 6/8 banded.</li>
</ul>
<h3>Retracted</h3>
<ul>
<li><b>&ldquo;The IPA's sharpen causes it.&rdquo;</b> Claimed on five consecutive clean captures.
Against a ~50% fault that is a 1-in-32 coincidence. Disproven by the interleaved hold&nbsp;31 test.</li>
<li><b>&ldquo;RAW clean vs RGB banded localises it to the colour pipeline.&rdquo;</b> The statistics were
sound (0/9 vs 7/8) but the design was confounded: the RGB arm ran after a flash-reset and the RAW arm
did not. A later RGB run after a console reboot gave 0/6 under identical settings and scene.</li>
</ul>
<h3>Open</h3>
<ul>
<li>An <b>uncontrolled state variable</b> flips RGB band incidence between ~0% and ~87%.
Leading suspect: hard reset after flashing vs software <code>esp_restart</code>.</li>
<li><code>isp blk &lt;name&gt; 0</code> <b>induces</b> the band for every block tested, so it cannot
be used to bisect the pipeline.</li>
</ul>
""")

    A('<h2>Experiments</h2><div class="scroll"><table><tr><th>Condition</th><th>What it tested</th>'
      '<th class="n">Frames</th><th class="n">Banded</th><th class="n">Mean width</th></tr>')
    for pref, title, desc, kind in GROUPS:
        rs = [r for r in by.get(pref, []) if r.get("thumb")]
        if not rs:
            continue
        nb = sum(1 for r in rs if r["width"] > 0)
        lbl, col = badge[kind]
        bd = f'<span class="badge" style="background:{col}">{lbl}</span>' if lbl else ""
        A(f'<tr><td><b>{html.escape(title)}</b>{bd}</td><td>{html.escape(desc)}</td>'
          f'<td class="n">{len(rs)}</td><td class="n">{nb}</td>'
          f'<td class="n">{np.mean([r["width"] for r in rs]):.0f} px</td></tr>')
    A("</table></div>")

    A("<h2>Samples</h2>")
    for pref, title, desc, kind in GROUPS:
        rs = sorted([r for r in by.get(pref, []) if r.get("thumb")], key=lambda r: r["stem"])
        if not rs:
            continue
        lbl, col = badge[kind]
        bd = f'<span class="badge" style="background:{col}">{lbl}</span>' if lbl else ""
        nb = sum(1 for r in rs if r["width"] > 0)
        A(f'<h3>{html.escape(title)}{bd}</h3>')
        A(f'<p class="sub" style="margin:0 0 8px">{html.escape(desc)} &mdash; '
          f'{nb}/{len(rs)} banded.</p><div class="grid">')
        for r in rs:
            cls = "band" if r["width"] > 0 else "clean"
            txt = f'{r["width"]} px' if r["width"] > 0 else "clean"
            A(f'<div class="cell"><a href="{html.escape(r["stem"])}.bin"><img src="{html.escape(r["thumb"])}" '
              f'alt="{html.escape(r["stem"])}" loading="lazy"></a>'
              f'<div class="w {cls}">{txt}</div>'
              f'<div>{html.escape(r["tag"])} &middot; {html.escape(r["cc"])}</div>'
              f'<div>luma {r["luma"]:.0f} &middot; &Delta;B&minus;G {r["dbg"]:+.0f}</div></div>')
        A("</div>")

    A(f'<h2>All captures</h2><div class="scroll"><table>'
      '<tr><th>File</th><th>Format</th><th class="n">Bytes</th><th class="n">Width</th>'
      '<th class="n">Edge grad</th><th class="n">Median</th><th class="n">Luma</th></tr>')
    for r in sorted(rows, key=lambda r: r["stem"]):
        if not r.get("thumb"):
            A(f'<tr><td>{html.escape(r["stem"])}</td><td>{html.escape(r["cc"])}</td>'
              f'<td class="n">{r["size"]:,}</td><td colspan="4">not decodable</td></tr>')
            continue
        cls = "band" if r["width"] > 0 else "clean"
        A(f'<tr><td><a href="{html.escape(r["stem"])}.bin">{html.escape(r["stem"])}</a></td>'
          f'<td>{html.escape(r["cc"])}</td><td class="n">{r["size"]:,}</td>'
          f'<td class="n {cls}">{r["width"]}</td><td class="n">{r["edge"]:.2f}</td>'
          f'<td class="n">{r["med"]:.2f}</td><td class="n">{r["luma"]:.0f}</td></tr>')
    A("</table></div>")

    open("report.html", "w").write("\n".join(out))
    ok = sum(1 for r in rows if r.get("thumb"))
    print(f"report.html written — {ok}/{len(rows)} captures decoded")


if __name__ == "__main__":
    main()
