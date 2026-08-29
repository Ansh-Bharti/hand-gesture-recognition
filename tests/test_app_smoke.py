"""Smoke tests for app.py using Streamlit's AppTest harness.

AppTest executes the real Streamlit script headlessly (no browser, no
WebSocket) and lets assertions run against the resulting page elements.

It cannot simulate streamlit-webrtc's actual WebRTC session plumbing (that
needs a real browser + JS runtime): webrtc_streamer() reliably fails inside
this harness with an internal AttributeError from streamlit-webrtc reaching
into Streamlit Runtime internals that AppTest doesn't fully mock. That
failure is used here deliberately — it exercises this app's own try/except
boundary around webrtc_streamer(), proving the app degrades to a clear
on-page message instead of crashing, exactly as it should for any other
unexpected webcam-component initialization failure.

Real webcam permission handling (browser prompts, actual video streaming)
still requires manual testing in a real browser — see the README's manual
test checklist. This file only covers what's deterministically testable
without one.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_app() -> AppTest:
    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.run()
    return at


def test_app_starts_without_an_unhandled_exception():
    at = _run_app()
    assert not at.exception


def test_app_shows_friendly_error_when_webcam_component_fails_to_init():
    at = _run_app()
    error_messages = [e.value.lower() for e in at.error]
    assert any("webcam component" in msg for msg in error_messages)


def test_app_shows_camera_inactive_status_when_stream_is_not_playing():
    at = _run_app()
    warning_messages = [w.value.lower() for w in at.warning]
    assert any("camera inactive" in msg for msg in warning_messages)


def test_app_renders_static_content_regardless_of_webcam_state():
    at = _run_app()
    subheaders = [s.value for s in at.subheader]
    assert "Webhook Configuration" in subheaders
    assert "Supported Gestures" in subheaders
    assert len(at.text_input) == 1
