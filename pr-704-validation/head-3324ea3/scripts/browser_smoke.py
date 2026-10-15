# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Chromium WebRTC input, rejection, video, and reconnection smoke check."""

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "704" / "browser"
CHECKOUT = Path("<CHECKOUT>")
OUTPUT.mkdir(parents=True, exist_ok=True)
assert not (OUTPUT / "events.jsonl").exists(), "Use a fresh browser output directory"
environment = os.environ.copy()
environment.update(PYTHONPATH=f"{CHECKOUT / 'flashdreams'}:{CHECKOUT / 'integrations_v2/red_screen'}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
command = ["<VENV>/bin/python", str(ROOT / "browser_server.py")]
result = dict(head="3324ea30559b78586353a69704b5422c4f164f77", status="running", command=command, checks=[], console=[], http_errors=[], page_errors=[], protocol_injection=["keyboard", "mouse", "focus", "touch", "gamepad", "game_wheel", "xr_controller", "reset", "close"], injected_types=[], device_protocol_only=["gamepad", "game_wheel", "xr_controller"])


def events():
    """Read the server's append-only event log."""
    path = OUTPUT / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_event(kind, start, predicate=lambda event: True):
    """Require a newly delivered event satisfying a predicate."""
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if any(event["type"] == kind and predicate(event) for event in events()[start:]):
            return
        time.sleep(0.05)
    raise AssertionError(f"No new {kind} event: {events()[start:]}")


def video_state(page):
    """Return actual decoded video progress and mean pixel intensity."""
    return page.evaluate("""() => {
      const v=document.querySelector('video');const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
      const x=c.getContext('2d');x.drawImage(v,0,0);const a=x.getImageData(0,0,c.width,c.height).data;
      let r=0,g=0,b=0;for(let i=0;i<a.length;i+=4){r+=a[i];g+=a[i+1];b+=a[i+2];}
      return {time:v.currentTime,frames:v.getVideoPlaybackQuality().totalVideoFrames,width:v.videoWidth,height:v.videoHeight,r:r/(a.length/4),g:g/(a.length/4),b:b/(a.length/4)};
    }""")


def send(page, payload):
    """Send through the page's real WebRTC data channel."""
    page.evaluate("value => window.__channels.controls.send(value)", payload if isinstance(payload, str) else json.dumps(payload))


with (OUTPUT / "server.log").open("w") as server_log:
    server = subprocess.Popen(command, cwd=CHECKOUT, env=environment, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 90
        while not (OUTPUT / "url.txt").exists():
            assert server.poll() is None, "Server exited before opening"
            assert time.monotonic() < deadline, "Server startup timeout"
            time.sleep(0.1)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(Path.home()/".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"))
            result["browser_version"] = browser.version
            context = browser.new_context(viewport={"width": 960, "height": 540}, has_touch=True, record_video_dir=str(OUTPUT / "recordings"))
            context.add_init_script("""(() => {
              window.__channels={};window.__replies=[];window.__sent=[];
              const create=RTCPeerConnection.prototype.createDataChannel;
              RTCPeerConnection.prototype.createDataChannel=function(...args){
                const ch=create.apply(this,args);window.__channels[ch.label]=ch;
                const send=ch.send.bind(ch);ch.send=value=>{window.__sent.push(value);return send(value);};
                ch.addEventListener('message',e=>{try{window.__replies.push(JSON.parse(e.data));}catch{window.__replies.push(e.data);}});
                return ch;
              };
            })();""")
            page = context.new_page()
            page.on("console", lambda message: result["console"].append(dict(type=message.type, text=message.text, location=message.location)))
            page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
            page.on("response", lambda response: result["http_errors"].append(dict(url=response.url, status=response.status)) if response.status >= 400 else None)
            url = (OUTPUT / "url.txt").read_text().strip()
            page.goto(url)
            page.wait_for_function("window.__channels.controls?.readyState === 'open' && document.querySelector('video').videoWidth > 0", timeout=30000)
            before = video_state(page)
            page.wait_for_timeout(600)
            after = video_state(page)
            assert after["frames"] > before["frames"]
            result["checks"].append(dict(name="video-decoding", status="passed", before=before, after=after))

            start = len(events())
            page.keyboard.down("r")
            wait_event("keyboard", start, lambda e: e["fields"]["key"] == "r")
            page.wait_for_function("""() => {
                const v=document.querySelector('video');const c=document.createElement('canvas');c.width=1;c.height=1;
                const x=c.getContext('2d');x.drawImage(v,0,0,1,1);return x.getImageData(0,0,1,1).data[0]>180;
            }""", timeout=10000)
            red = video_state(page)
            assert red["r"] > 180 and red["g"] < 30 and red["b"] < 30, red
            start = len(events())
            page.keyboard.up("r")
            wait_event("keyboard", start)
            page.wait_for_function("""() => {
                const v=document.querySelector('video');const c=document.createElement('canvas');c.width=1;c.height=1;
                const x=c.getContext('2d');x.drawImage(v,0,0,1,1);return x.getImageData(0,0,1,1).data[0]<30;
            }""", timeout=10000)
            black = video_state(page)
            assert black["r"] < 30, black
            result["checks"].append(dict(name="keyboard-press-release-and-video", status="passed", red=red, black=black))

            start = len(events())
            page.mouse.move(480, 270)
            page.mouse.down()
            page.mouse.up()
            wait_event("mouse", start)
            start = len(events())
            page.touchscreen.tap(480, 270)
            wait_event("touch", start)
            start = len(events())
            page.evaluate("document.querySelector('video').blur()")
            wait_event("focus", start, lambda event: event["fields"]["focused"] is False)
            assert page.evaluate("document.activeElement !== document.querySelector('video')")
            page.locator("video").focus()
            wait_event("focus", start, lambda event: event["fields"]["focused"] is True)
            assert page.evaluate("document.activeElement === document.querySelector('video')")
            result["checks"].append(dict(name="mouse-touch-focus-events", status="passed", focus_method="actual element blur/focus with activeElement assertions"))

            for payload in [
                {"type":"keyboard","key":"q","pressed":False},
                {"type":"mouse","action":"wheel","x":0.5,"y":0.5,"wheel_y":-0.25},
                {"type":"focus","focused":True},
                {"type":"touch","action":"cancel","touch_id":99,"x":0.3,"y":0.4},
                {"type":"gamepad","action":"state","index":1,"axes":[-0.5,0.25],"buttons":[0,1],"pressed":[False,True]},
                {"type":"game_wheel","action":"state","steering":0.25,"throttle":0.8,"brake":0.1},
                {"type":"xr_controller","action":"state","handedness":"right","position":[1,2,3],"orientation":[0,0,0,1]},
                {"type":"reset"},
            ]:
                start = len(events())
                send(page, payload)
                wait_event(payload["type"], start)
                result["injected_types"].append(payload["type"])
            result["checks"].append(dict(name="device-and-reset-protocol-injection", status="passed"))

            malformed = ["{", json.dumps({"type":"mouse","action":"fly"}),
                         '{"type":"mouse","action":"move","x":' + "1" + "0"*400 + ',"y":0}',
                         '{"type":"close","extra":' + "["*10000 + "0" + "]"*10000 + "}"]
            for index, payload in enumerate(malformed):
                page.wait_for_timeout(150)
                start = len(events())
                errors = page.evaluate("window.__replies.filter(x=>x.type==='error').length")
                send(page, payload)
                page.wait_for_function("count => window.__replies.filter(x=>x.type==='error').length > count", arg=errors, timeout=8000)
                page.wait_for_timeout(150)
                assert len(events()) == start, (index, events()[start:])
                result["checks"].append(dict(name=f"malformed-{index}", status="passed", error=page.evaluate("window.__replies.filter(x=>x.type==='error').at(-1)")))
            start = len(events())
            page.keyboard.down("w")
            wait_event("keyboard", start, lambda event: event["fields"]["key"] == "w")
            page.keyboard.up("w")
            before = video_state(page)
            page.wait_for_timeout(500)
            after = video_state(page)
            assert after["frames"] > before["frames"]
            result["checks"].append(dict(name="input-and-video-survive-invalid-messages", status="passed", before=before, after=after))

            page.screenshot(path=str(OUTPUT / "before-refresh.png"))
            page.reload()
            page.wait_for_function("window.__channels.controls?.readyState === 'open' && document.querySelector('video').videoWidth > 0", timeout=30000)
            start = len(events())
            page.keyboard.press("w")
            wait_event("keyboard", start)
            before = video_state(page)
            page.wait_for_timeout(500)
            after = video_state(page)
            assert after["frames"] > before["frames"]
            result["checks"].append(dict(name="refresh-reconnect-without-server-restart", status="passed", before=before, after=after))
            page.screenshot(path=str(OUTPUT / "after-refresh.png"))
            start = len(events())
            send(page, {"type":"close"})
            wait_event("close", start)
            result["injected_types"].append("close")
            assert set(result["injected_types"]) == set(result["protocol_injection"])
            result["checks"].append(dict(name="nine-legal-protocol-types",status="passed",types=result["injected_types"]))
            server.wait(timeout=15)
            assert server.returncode == 0
            result["checks"].append(dict(name="explicit-close-stops-application", status="passed"))
            result["replies"] = page.evaluate("window.__replies")
            context.close()
            browser.close()
            assert result["page_errors"] == [], result["page_errors"]
            result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["failure"] = traceback.format_exc()
        print(result["failure"], flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=15)
        result["server_exit_code"] = server.returncode
        (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "passed" else 1)
