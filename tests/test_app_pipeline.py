"""Integration-style tests for the frame-processing pipeline in app.py.

tests/test_app_smoke.py proves the page renders; it cannot exercise
_on_frame itself because streamlit-webrtc never actually invokes the
callback inside that harness (see its docstring). This file instead calls
_on_frame directly with a fake frame and a fake detector, proving detector
-> classifier -> debounce -> webhook-dispatch-decision are wired together
correctly in app.py — with no real camera, MediaPipe model, or network
call involved anywhere.
"""

import numpy as np
import pytest

import app as app_module
from services.webhook import WebhookResult

WRIST = (0.5, 0.95, 0.0)


def _thumbs_up_landmarks():
    """A minimal, valid 21-point pose that classify_gesture reads as thumbs_up."""
    landmarks = [(0.5, 0.7, 0.0)] * 21
    landmarks[0] = WRIST
    landmarks[9] = (0.5, 0.7, 0.0)  # middle MCP: defines hand_size with wrist
    landmarks[2] = (0.35, 0.75, 0.0)  # thumb MCP
    landmarks[4] = (0.05, 0.5, 0.0)  # thumb tip: far from palm and above wrist
    # All other fingers folded: tip near its own MCP.
    for base in (5, 13, 17):  # index, ring, pinky MCPs left at fold-like defaults
        landmarks[base] = (0.5, 0.7, 0.0)
        landmarks[base + 1] = (0.5, 0.55, 0.0)
        landmarks[base + 3] = (0.5, 0.68, 0.0)
    return landmarks


class _FakeFrame:
    """Minimal stand-in for av.VideoFrame, matching the interface _on_frame uses."""

    def __init__(self, array):
        self._array = array

    def to_ndarray(self, format):
        return self._array


class _FakeDetectionResult:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.detection_confidence = 0.9 if landmarks is not None else None

    @property
    def hand_present(self):
        return self.landmarks is not None


class _FakeDetector:
    """Returns the same canned landmarks (or None) on every detect() call."""

    def __init__(self, landmarks):
        self._landmarks = landmarks

    def detect(self, frame):
        return _FakeDetectionResult(self._landmarks)


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """app.py's module-level singletons must not leak state between tests."""
    app_module._gesture_state.reset()
    app_module._webhook_config.url = ""
    app_module._webhook_config.last_result = None
    app_module._webhook_config.last_gesture = None
    app_module._live_status.hand_present = False
    app_module._live_status.detection_confidence = None
    yield
    app_module._gesture_state.reset()


def _blank_frame():
    return _FakeFrame(np.zeros((100, 100, 3), dtype=np.uint8))


def _feed_frames(n):
    for _ in range(n):
        app_module._on_frame(_blank_frame())


def test_on_frame_confirms_gesture_after_enough_consistent_frames(monkeypatch):
    monkeypatch.setattr(app_module, "get_detector", lambda: _FakeDetector(_thumbs_up_landmarks()))

    _feed_frames(app_module._gesture_state._confirm_frames)

    assert app_module._gesture_state.confirmed_gesture == "thumbs_up"


def test_on_frame_dispatches_webhook_when_configured(monkeypatch):
    monkeypatch.setattr(app_module, "get_detector", lambda: _FakeDetector(_thumbs_up_landmarks()))
    fake_result = WebhookResult(success=True, status_code=200, error=None, sent_at=None)
    monkeypatch.setattr(
        app_module,
        "send_webhook_async",
        lambda url, gesture, ts, on_result=None: on_result(fake_result),
    )
    app_module._webhook_config.set_url("https://example.com/hook")

    _feed_frames(app_module._gesture_state._confirm_frames)

    _, last_gesture, last_result = app_module._webhook_config.snapshot()
    assert last_gesture == "thumbs_up"
    assert last_result.success is True


def test_on_frame_skips_webhook_when_not_configured(monkeypatch):
    monkeypatch.setattr(app_module, "get_detector", lambda: _FakeDetector(_thumbs_up_landmarks()))
    calls = []
    monkeypatch.setattr(app_module, "send_webhook_async", lambda *a, **k: calls.append((a, k)))

    _feed_frames(app_module._gesture_state._confirm_frames)

    assert calls == []


def test_on_frame_records_hand_absent_status(monkeypatch):
    monkeypatch.setattr(app_module, "get_detector", lambda: _FakeDetector(None))

    app_module._on_frame(_blank_frame())

    hand_present, confidence = app_module._live_status.snapshot()
    assert hand_present is False
    assert confidence is None


def test_on_frame_never_raises_when_detector_unavailable(monkeypatch):
    monkeypatch.setattr(app_module, "get_detector", lambda: None)

    result = app_module._on_frame(_blank_frame())

    assert result is not None
