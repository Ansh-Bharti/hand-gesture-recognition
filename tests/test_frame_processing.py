"""Deterministic tests for the per-frame video callback in app.py.

The webcam UI itself requires a browser and cannot be exercised in an
automated test; the frame-transform logic it delegates to is pure and is
tested here in isolation.
"""

import av
import numpy as np

from app import _on_frame


def _make_frame(width: int = 4, height: int = 2) -> av.VideoFrame:
    # Distinct per-column pixel values so mirroring is easy to detect.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        img[:, x, :] = x
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def test_on_frame_mirrors_horizontally():
    frame = _make_frame(width=4, height=2)
    result = _on_frame(frame)

    original = frame.to_ndarray(format="bgr24")
    mirrored = result.to_ndarray(format="bgr24")

    assert mirrored.shape == original.shape
    assert np.array_equal(mirrored, original[:, ::-1, :])


def test_on_frame_never_raises_on_bad_input():
    class BrokenFrame:
        def to_ndarray(self, format):
            raise RuntimeError("simulated decode failure")

    broken = BrokenFrame()
    result = _on_frame(broken)  # must not raise

    assert result is broken
