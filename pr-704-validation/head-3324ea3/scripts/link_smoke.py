# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Chromium check of server-side link regions: a click opens the link and is not delivered to the app."""

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "link"
CHECKOUT = Path(sys.argv[1])
OUTPUT.mkdir(parents=True, exist_ok=True)
assert not (OUTPUT / "events.jsonl").exists(), "Use a fresh link output directory"
hits: list[str] = []


def opened():
    """Count link-page requests; a popup also fetches /favicon.ico, which is not an extra open."""
    return sum(path == "/" for path in hits)



class Target(http.server.BaseHTTPRequestHandler):
    """Record every request made to the link destination."""

    def do_GET(self):
        hits.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<title>link target</title>link target")

    def log_message(self, *args):
        pass


target = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Target)
threading.Thread(target=target.serve_forever, daemon=True).start()
link_url = f"http://127.0.0.1:{target.server_port}/"
environment = os.environ.copy()
environment.update(PYTHONPATH=f"{CHECKOUT / 'flashdreams'}:{CHECKOUT / 'integrations_v2/red_screen'}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", LINK_URL=link_url)
command = ["<VENV>/bin/python", str(ROOT / "link_server.py")]
head = subprocess.run(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
result = dict(head=head, status="running", link_url=link_url, link_bounds=[0.1, 0.1, 0.4, 0.4], command=command, checks=[], console=[], page_errors=[])


def events(kind=None, **fields):
    path = OUTPUT / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if (kind is None or row["type"] == kind) and all(row["fields"].get(k) == v for k, v in fields.items())]


def wait_until(condition, what, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {what}: hits={hits}, events={events()[-6:]}")


def settle(page, ms=1200):
    page.wait_for_timeout(ms)


with (OUTPUT / "server.log").open("w") as server_log:
    server = subprocess.Popen(command, cwd=CHECKOUT, env=environment, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 90
        while not (OUTPUT / "url.txt").exists():
            assert server.poll() is None, "Server exited before opening"
            assert time.monotonic() < deadline, "Server startup timeout"
            time.sleep(0.1)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"))
            result["browser_version"] = browser.version
            context = browser.new_context(viewport={"width": 960, "height": 540}, has_touch=True)
            context.add_init_script("""(() => { window.__channels={}; const create=RTCPeerConnection.prototype.createDataChannel;
              RTCPeerConnection.prototype.createDataChannel=function(...a){const ch=create.apply(this,a);window.__channels[ch.label]=ch;return ch;}; })();""")
            page = context.new_page()
            page.on("console", lambda m: result["console"].append(dict(type=m.type, text=m.text)))
            page.on("pageerror", lambda e: result["page_errors"].append(str(e)))
            page.goto((OUTPUT / "url.txt").read_text().strip())
            page.wait_for_function("window.__channels.controls?.readyState === 'open' && document.querySelector('video').videoWidth > 0", timeout=30000)
            settle(page, 600)
            bounds = page.evaluate("""() => { const v=document.querySelector('video'), b=v.getBoundingClientRect();
              const s=Math.min(b.width/v.videoWidth,b.height/v.videoHeight), w=v.videoWidth*s, h=v.videoHeight*s;
              return {left:b.left+(b.width-w)/2, top:b.top+(b.height-h)/2, width:w, height:h}; }""")
            at = lambda fx, fy: (bounds["left"] + fx * bounds["width"], bounds["top"] + fy * bounds["height"])
            inside, outside = at(0.25, 0.25), at(0.75, 0.75)
            result["content_bounds_px"] = bounds

            def buttons():
                return len(events("mouse", action="button"))

            def touches():
                return len(events("touch"))

            # 1. Primary click inside the link opens it once and the app sees no button events.
            base_hits, base_buttons = opened(), buttons()
            page.mouse.move(*inside)
            page.mouse.down()
            page.mouse.up()
            wait_until(lambda: opened() == base_hits + 1, "link request after mouse click")
            settle(page)
            assert opened() == base_hits + 1, hits
            assert buttons() == base_buttons, events("mouse", action="button")
            result["checks"].append(dict(name="mouse-click-inside-opens-link-and-is-consumed", status="passed", hits=len(hits)))

            # 2. Click outside the link is an ordinary app event and opens nothing.
            base_hits, base_buttons = opened(), buttons()
            page.mouse.move(*outside)
            page.mouse.down()
            page.mouse.up()
            wait_until(lambda: buttons() == base_buttons + 2, "ordinary mouse button events")
            settle(page)
            assert opened() == base_hits, hits
            result["checks"].append(dict(name="mouse-click-outside-reaches-app-and-opens-nothing", status="passed"))

            # 3. Press inside, release outside: no link opens and the app sees no button events.
            base_hits, base_buttons = opened(), buttons()
            page.mouse.move(*inside)
            page.mouse.down()
            page.mouse.move(*outside)
            page.mouse.up()
            settle(page)
            assert opened() == base_hits, hits
            assert buttons() == base_buttons, events("mouse", action="button")
            result["checks"].append(dict(name="mouse-press-inside-release-outside-opens-nothing", status="passed"))

            # 4. Touch tap inside the link opens it once and the app sees no touch events.
            base_hits, base_touches = opened(), touches()
            page.touchscreen.tap(*inside)
            wait_until(lambda: opened() == base_hits + 1, "link request after touch tap")
            settle(page)
            assert opened() == base_hits + 1, hits
            assert touches() == base_touches, events("touch")
            result["checks"].append(dict(name="touch-tap-inside-opens-link-and-is-consumed", status="passed"))

            # 5. Touch tap outside reaches the app and opens nothing.
            base_hits, base_touches = opened(), touches()
            page.touchscreen.tap(*outside)
            wait_until(lambda: touches() >= base_touches + 2, "ordinary touch events")
            settle(page)
            assert opened() == base_hits, hits
            result["checks"].append(dict(name="touch-tap-outside-reaches-app-and-opens-nothing", status="passed"))

            # 6. Ordinary input still works afterwards, and close ends the app.
            base_keys = len(events("keyboard"))
            page.keyboard.press("w")
            wait_until(lambda: len(events("keyboard")) >= base_keys + 2, "keyboard events after link interactions")
            result["checks"].append(dict(name="keyboard-works-after-link-interactions", status="passed"))
            page.evaluate("v => window.__channels.controls.send(v)", json.dumps({"type": "close"}))
            server.wait(timeout=15)
            assert server.returncode == 0
            result["checks"].append(dict(name="explicit-close-stops-application", status="passed"))
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
        target.shutdown()
        result["server_exit_code"] = server.returncode
        result["target_requests"] = list(hits)
        (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(status=result["status"], checks=len(result["checks"]))))
raise SystemExit(0 if result["status"] == "passed" else 1)
