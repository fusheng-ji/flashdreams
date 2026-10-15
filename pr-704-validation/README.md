# Re-verification of NVIDIA/flashdreams#704

Two heads were verified: `77ecedd` (original merge head, results in `data/`) and `3324ea3` (after merging upstream main, results in [head-3324ea3/](head-3324ea3)). The sections below describe `77ecedd`; the new-head results are summarized first.

## After merging upstream main: head `3324ea3`

Upstream added clickable link regions to the browser-message handler that this PR extracts. The merge keeps `parse_browser_event` and adds `link_gesture()`, so the browser path was re-tested in real Chromium 153 (Playwright) with the same red-screen app.

**Original 12 checks, unchanged:** all pass ([result](head-3324ea3/browser/result.json), [recording](head-3324ea3/browser/recording.webm), [event log](head-3324ea3/browser/events.jsonl)). There were 0 JavaScript page errors and no HTTP errors; the console again shows only a favicon 404 and the four expected protocol-rejection warnings.

**New link checks (7):** one link region over the top-left of the content, served to a local HTTP target ([result](head-3324ea3/link/result.json), [events](head-3324ea3/link/events.jsonl)):

| Check | Result |
| --- | --- |
| Primary mouse click inside the link opens the link once; the app receives no button events | Passed |
| Mouse click outside the link reaches the app and opens nothing | Passed |
| Press inside, release outside: nothing opens and the app receives no button events | Passed |
| Touch tap inside the link opens it once; the app receives no touch events | Passed |
| Touch tap outside reaches the app and opens nothing | Passed |
| Keyboard input still works afterwards | Passed |
| Explicit `close` stops the app with exit code 0 | Passed |

The target saw exactly two page requests (one per opened link). Its `/favicon.ico` requests come from the opened tab, not from extra opens.

**Negative control:** with `link_gesture` temporarily disabled in a scratch copy, the first link check fails (no link opens and the click reaches the app), so these checks do detect broken link handling ([result](head-3324ea3/link-negative-control/result.json)).

Scripts are in [head-3324ea3/scripts](head-3324ea3/scripts). Unit-level and aiortc-peer link tests run in the CPU suite (567 CPU tests and the CUDA WebRTC test passed on this head; see the PR body).

---

# Results at head `77ecedd`

The checks were rerun on merge head `77ecedd338bd0b184571cf30186031b49be0025b`, with locked dependencies and Python 3.12.13. A real Chromium end-to-end smoke test was added.

## Real-browser smoke test

Setup:
- Chromium 153.0.8010.12, driven by Playwright.
- The red-screen demo app ran through the PR's WebRTC server on CPU.
- Results are in [result.json](data/browser/result.json).

| Check | Result |
| --- | --- |
| Video decodes and advances | Passed |
| Real keyboard press turns the frame red; release returns it to black | Passed |
| Mouse, touch and DOM focus changes (real `blur`/`focus`, `activeElement` asserted) are consumed | Passed |
| All nine protocol message types reach the server over the real data channel | Passed |
| Gamepad, game wheel and XR controller | Protocol only; no physical devices |
| Invalid JSON, invalid enum, non-finite number and over-deep nesting | Each returns an error reply and delivers no input event |
| Keyboard input and video still work after those invalid messages | Passed |
| Page refresh reconnects without restarting the server | Passed |
| Explicit `close` stops the app (exit code 0) | Passed |

Other outputs:
- **Page and console:** 0 JavaScript page errors and no HTTP errors. The console holds a favicon 404 and the four expected protocol-rejection warnings.
- **Recordings and logs:** a [recording](data/browser/recordings), [before](data/browser/before-refresh.png) and [after](data/browser/after-refresh.png) refresh screenshots, the [server event log](data/browser/events.jsonl) and the [server log](data/browser/server.log).
- **Scripts:** [smoke driver](scripts/browser_smoke.py) and [server wrapper](scripts/browser_server.py).
- **Earlier failed attempts:** two earlier probe attempts are kept locally but not published. They failed because the probe misused the event-log API and used fixed waits that were too short. The final run uses conditional waits, real focus and explicit injection of all nine types.

## Repeated CI-equivalent checks

| Check | Result |
| --- | --- |
| `pytest -m ci_cpu`: dependency policy, browser input, WebRTC client window | 113 passed ([log](data/cpu.log)) |
| `pytest -m ci_gpu`: WebRTC client window, CUDA | 1 passed ([log](data/cuda.log)) |
| `ruff check` / `ruff format --check` on the 4 changed files | Passed |
| `uv lock --check` | Passed |
| `uv pip check` | Fails with one conflict: `torch` requires `setuptools<82`, but 83.0.0 is installed. This conflict is pre-existing and already disclosed in the PR ([log](data/dependency-consistency.log)) |

Local paths are redacted as `<REPO>`, `<HOME>`, `<TMP>`, `<CHECKOUT>` and `<VENV>`.
