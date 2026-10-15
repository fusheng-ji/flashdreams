# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Red-screen application with one server-side link region and test-only event logging."""

import dataclasses
import json
import os
from pathlib import Path

from red_screen.app import create_app
from flashdreams.api_v2.client_window import ClientWindowLink
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

ROOT = Path(__file__).resolve().parent / "link"
ROOT.mkdir(parents=True, exist_ok=True)
LINK = ClientWindowLink("Smoke link", os.environ["LINK_URL"], (0.1, 0.1, 0.4, 0.4))
window = WebRTCClientWindow()
receive = window.get_user_input_events


def record_events():
    """Record consumed events without changing the application's event stream."""
    events = receive()
    with (ROOT / "events.jsonl").open("a") as handle:
        for event in events.get_events():
            handle.write(json.dumps(dict(type=event.get_type_name(), fields=dataclasses.asdict(event)), default=lambda value: getattr(value, "value", int(value) if hasattr(value, "__int__") else str(value))) + "\n")
    return events


window.get_user_input_events = record_events

# Opening a session clears links, so register them after every open, as an app would.
open_server = window.server.open


def open_with_links(session_desc):
    open_server(session_desc)
    window.request_links((LINK,))


window.server.open = open_with_links
(ROOT / "url.txt").write_text(window.server.url)
try:
    ApplicationRunner(create_app(), window).run(
        SessionDesc(output_layout=VideoTensorLayout.bcthw, video_width=320, video_height=180,
                    frames_per_second_for_ui=30, frames_per_second_for_step=30),
        timeout_seconds=180,
    )
finally:
    window.close()
