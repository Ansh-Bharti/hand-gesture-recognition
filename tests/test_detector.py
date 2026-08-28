"""Deterministic tests for gesture/detector.py.

These tests avoid depending on the real MediaPipe model, network access, or
a physical camera: HandDetector's internal `_landmarker` is replaced with a
lightweight fake so the wrapping/parsing logic can be verified in isolation,
per the assignment's guidance to isolate deterministic logic from the
webcam/model layer.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gesture import detector as detector_module
from gesture.detector import HandDetector, ModelUnavailableError, draw_landmarks


def _make_detector_with_fake_landmarker(fake_landmarker):
    instance = HandDetector.__new__(HandDetector)
    instance._landmarker = fake_landmarker
    return instance


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeCategory:
    def __init__(self, category_name, score):
        self.category_name = category_name
        self.score = score


def _fake_result(landmarks_2d=None, handedness=None):
    return SimpleNamespace(
        hand_landmarks=landmarks_2d or [],
        handedness=handedness or [],
    )


def test_detect_parses_landmarks_and_handedness():
    landmarks = [_FakeLandmark(i / 21, i / 21, 0.0) for i in range(21)]
    result = _fake_result(
        landmarks_2d=[landmarks],
        handedness=[[_FakeCategory("Right", 0.97)]],
    )
    fake_landmarker = SimpleNamespace(detect=lambda mp_image: result)
    d = _make_detector_with_fake_landmarker(fake_landmarker)

    detection = d.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert detection.hand_present
    assert len(detection.landmarks) == 21
    assert detection.handedness == "Right"
    assert detection.detection_confidence == pytest.approx(0.97)


def test_detect_reports_no_hand_when_nothing_found():
    fake_landmarker = SimpleNamespace(detect=lambda mp_image: _fake_result())
    d = _make_detector_with_fake_landmarker(fake_landmarker)

    detection = d.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert not detection.hand_present
    assert detection.landmarks is None
    assert detection.handedness is None
    assert detection.detection_confidence is None


def test_detect_never_raises_on_internal_error():
    def _boom(mp_image):
        raise RuntimeError("simulated mediapipe failure")

    fake_landmarker = SimpleNamespace(detect=_boom)
    d = _make_detector_with_fake_landmarker(fake_landmarker)

    detection = d.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert not detection.hand_present


def test_ensure_model_downloaded_uses_cache_without_network(monkeypatch, tmp_path):
    cached = tmp_path / "hand_landmarker.task"
    cached.write_bytes(b"fake-model-bytes")
    monkeypatch.setattr(detector_module, "MODEL_PATH", cached)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("network should not be hit when a cached model exists")

    monkeypatch.setattr(detector_module, "urlopen", _fail_if_called)

    result_path = detector_module.ensure_model_downloaded()

    assert result_path == cached


def test_ensure_model_downloaded_raises_clear_error_on_network_failure(monkeypatch, tmp_path):
    missing = tmp_path / "hand_landmarker.task"
    monkeypatch.setattr(detector_module, "MODEL_PATH", missing)
    monkeypatch.setattr(detector_module, "MODEL_DIR", tmp_path)

    def _raise_network_error(*args, **kwargs):
        raise OSError("simulated network failure")

    monkeypatch.setattr(detector_module, "urlopen", _raise_network_error)

    with pytest.raises(ModelUnavailableError):
        detector_module.ensure_model_downloaded()


def test_draw_landmarks_returns_same_shape_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = [(i / 21, i / 21, 0.0) for i in range(21)]

    annotated = draw_landmarks(frame, landmarks)

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)  # something was actually drawn
