# Re-verification of NVIDIA/flashdreams#704 at head `77ecedd`

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
