import asyncio, json
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

API = "http://127.0.0.1:8769"
CAM = "cam1"

async def main():
    async with aiohttp.ClientSession() as cs:
        # 1) open a stream channel via the hub API
        async with cs.post(f"{API}/api/v1/cameras/{CAM}/stream", json={"protocol": "webrtc"}) as r:
            info = await r.json()
        print("open_stream:", info)
        sid = info["sid"]; offer_url = API + info["offer_url"]

        # 2) WebRTC offer/answer (recvonly, like a browser)
        pc = RTCPeerConnection()
        got = asyncio.Event(); frames = [0]; codecs = []

        @pc.on("track")
        def on_track(track):
            print("track:", track.kind)
            async def pump():
                try:
                    while frames[0] < 30:
                        fr = await track.recv()
                        frames[0] += 1
                        if frames[0] == 1:
                            print("first frame:", type(fr).__name__, getattr(fr, "width", "?"), "x", getattr(fr, "height", "?"))
                        got.set()
                except Exception as e:
                    print("recv ended:", e)
            asyncio.ensure_future(pump())

        pc.addTransceiver("video", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        async with cs.post(offer_url, json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}) as r:
            answer = await r.json()
        # which codec did we negotiate?
        for line in answer["sdp"].splitlines():
            if line.startswith("a=rtpmap"):
                codecs.append(line.split(" ", 1)[1])
        print("answer codecs:", codecs)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        # 3) wait for frames
        try:
            await asyncio.wait_for(got.wait(), timeout=15)
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            print("TIMEOUT waiting for video frames")
        print(f"RESULT: decoded {frames[0]} video frames, pc={pc.connectionState}")

        # 4) teardown
        await pc.close()
        async with cs.delete(f"{API}/api/v1/cameras/{CAM}/stream/{sid}") as r:
            print("close:", await r.json())

asyncio.run(main())
