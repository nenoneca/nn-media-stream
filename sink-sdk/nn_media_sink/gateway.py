"""
Hub video gateway — opens a real-time channel to the video streaming service
and re-publishes the live camera video to end users over WebRTC.

  C6 ─enc H.264─▶ video_service (GStreamer) ──RTP/UDP──▶ [hub: this module]
                                                              └─ aiortc ─▶ browser (WebRTC)

Flow:
  1. POST /api/v1/cameras/{cam}/stream  {protocol:"webrtc"}
       → the hub picks a local UDP port, tells the video streaming service to
         append an RTP/UDP branch aimed at that port (the "append a GST element"
         step), and starts ingesting the RTP/H.264 with aiortc (PyAV).
       → returns {sid, offer_url, ...}.
  2. Browser opens GET /webrtc/{sid}, creates a recvonly offer, and POSTs it to
       POST /api/v1/cameras/{cam}/stream/{sid}/offer
       → the hub answers; WebRTC media (H.264, passthrough — no transcode) flows.
  3. DELETE /api/v1/cameras/{cam}/stream/{sid}  → tears down the PC + branch.

WebRTC is done entirely on the hub (aiortc), NOT in GStreamer; the service↔hub
hop is plain RTP/UDP.  Mount into the hub aiohttp app via setup_video_routes(),
or run standalone (python -m nn_media_sink.gateway).
"""
from __future__ import annotations
import argparse
import asyncio
import contextlib
import logging
import socket
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web, ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay

log = logging.getLogger("hub.video")

# One SDP per session describing the RTP/H.264 stream we receive from the service.
_SDP_TMPL = """v=0
o=- 0 0 IN IP4 {ip}
s=nn-camera
c=IN IP4 {ip}
t=0 0
m=video {port} RTP/AVP {pt}
a=rtpmap:{pt} H264/90000
a=fmtp:{pt} packetization-mode=1
"""


@dataclass
class StreamSession:
    sid: str
    cam: str
    udp_port: int
    branch_id: int
    sdp_path: str
    player: MediaPlayer
    relay: MediaRelay
    pcs: set = field(default_factory=set)


class VideoGateway:
    def __init__(self, service_url: str, recv_ip: str = "127.0.0.1"):
        self.service_url = service_url.rstrip("/")
        self.recv_ip = recv_ip
        self.sessions: dict[str, StreamSession] = {}

    @staticmethod
    def _free_udp_port() -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    async def open_stream(self, cam: str, pt: int = 96) -> StreamSession:
        port = self._free_udp_port()
        sdp = _SDP_TMPL.format(ip=self.recv_ip, port=port, pt=pt)
        f = tempfile.NamedTemporaryFile("w", suffix=".sdp", delete=False)
        f.write(sdp)
        f.close()

        # Tell the video streaming service to append an RTP/UDP branch to us.
        async with ClientSession() as cs:
            async with cs.post(f"{self.service_url}/branch",
                               json={"host": self.recv_ip, "port": port, "pt": pt}) as r:
                if r.status != 200:
                    raise web.HTTPBadGateway(text=f"service /branch failed: {r.status}")
                branch = await r.json()
        log.info("opened service branch %s → %s:%d", branch.get("id"), self.recv_ip, port)

        # Ingest the RTP/H.264 with aiortc.  decode=True transcodes for the
        # browser (works with any negotiated codec); H.264 passthrough is a
        # later optimization (needs forced H264 codec prefs on the answer).
        player = MediaPlayer(
            f.name, format="sdp", decode=True,
            options={"protocol_whitelist": "file,udp,rtp", "fflags": "nobuffer",
                     "max_delay": "500000"},
        )
        sid = uuid.uuid4().hex[:12]
        sess = StreamSession(sid=sid, cam=cam, udp_port=port, branch_id=branch.get("id", -1),
                             sdp_path=f.name, player=player, relay=MediaRelay())
        self.sessions[sid] = sess
        return sess

    async def answer(self, sid: str, offer: RTCSessionDescription) -> RTCSessionDescription:
        sess = self.sessions[sid]
        pc = RTCPeerConnection()
        sess.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def _on_state():
            log.info("[%s] pc %s", sid, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                sess.pcs.discard(pc)
                with contextlib.suppress(Exception):
                    await pc.close()

        # Subscribe a relayed copy of the source so multiple viewers can share it.
        pc.addTrack(sess.relay.subscribe(sess.player.video))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    async def close_stream(self, sid: str):
        sess = self.sessions.pop(sid, None)
        if not sess:
            return False
        for pc in list(sess.pcs):
            with contextlib.suppress(Exception):
                await pc.close()
        with contextlib.suppress(Exception):
            sess.player.video.stop()
        # Ask the service to drop the branch.
        with contextlib.suppress(Exception):
            async with ClientSession() as cs:
                await cs.delete(f"{self.service_url}/branch/{sess.branch_id}")
        with contextlib.suppress(Exception):
            Path(sess.sdp_path).unlink()
        return True


# ── HTTP routes (mountable into the hub app) ────────────────────────────────
def setup_video_routes(app: web.Application, gw: VideoGateway, prefix: str = "/api/v1"):
    async def open_stream(request):
        cam = request.match_info["cam"]
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        proto = (body or {}).get("protocol", "webrtc")
        if proto != "webrtc":
            return web.json_response({"error": f"unsupported protocol {proto!r}"}, status=400)
        sess = await gw.open_stream(cam)
        return web.json_response({
            "sid": sess.sid, "cam": cam, "protocol": "webrtc",
            "offer_url": f"{prefix}/cameras/{cam}/stream/{sess.sid}/offer",
            "view_url": f"/webrtc/{sess.sid}",
            "ingest": {"transport": "rtp/udp", "port": sess.udp_port, "branch": sess.branch_id},
        })

    async def offer(request):
        sid = request.match_info["sid"]
        if sid not in gw.sessions:
            return web.json_response({"error": "no such stream"}, status=404)
        params = await request.json()
        ans = await gw.answer(sid, RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
        return web.json_response({"sdp": ans.sdp, "type": ans.type})

    async def close_stream(request):
        ok = await gw.close_stream(request.match_info["sid"])
        return web.json_response({"closed": ok}, status=200 if ok else 404)

    async def view_page(request):
        sid = request.match_info["sid"]
        sess = gw.sessions.get(sid)
        if not sess:
            return web.Response(status=404, text="no such stream")
        return web.Response(content_type="text/html",
                            text=_VIEW_HTML.replace("__SID__", sid).replace("__CAM__", sess.cam))

    app.add_routes([
        web.post(prefix + "/cameras/{cam}/stream", open_stream),
        web.post(prefix + "/cameras/{cam}/stream/{sid}/offer", offer),
        web.delete(prefix + "/cameras/{cam}/stream/{sid}", close_stream),
        web.get("/webrtc/{sid}", view_page),
    ])


_VIEW_HTML = """<!doctype html><meta charset=utf-8><title>nn camera __CAM__</title>
<body style="background:#111;color:#ddd;font-family:sans-serif">
<h3>nn camera __CAM__ (WebRTC)</h3>
<video id=v autoplay playsinline muted style="max-width:90vw;border:1px solid #444"></video>
<pre id=log></pre>
<script>
const sid="__SID__", cam="__CAM__";
const log=m=>document.getElementById('log').textContent+=m+"\\n";
(async()=>{
  const pc=new RTCPeerConnection();
  pc.addTransceiver('video',{direction:'recvonly'});
  pc.ontrack=e=>{document.getElementById('v').srcObject=e.streams[0];log('track');};
  pc.onconnectionstatechange=()=>log('pc '+pc.connectionState);
  const offer=await pc.createOffer();
  await pc.setLocalDescription(offer);
  const r=await fetch(`/api/v1/cameras/${cam}/stream/${sid}/offer`,{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
  await pc.setRemoteDescription(await r.json());
  log('answered');
})().catch(e=>log('ERR '+e));
</script>
"""


# ── standalone runner (for testing without the full hub) ────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--service-url", default="http://127.0.0.1:8899")
    ap.add_argument("--recv-ip", default="127.0.0.1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gw = VideoGateway(args.service_url, args.recv_ip)
    app = web.Application()
    setup_video_routes(app, gw)

    async def on_cleanup(app):
        for sid in list(gw.sessions):
            await gw.close_stream(sid)
    app.on_cleanup.append(on_cleanup)

    print(f">> hub video gateway on http://0.0.0.0:{args.port}  (service {args.service_url})")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
