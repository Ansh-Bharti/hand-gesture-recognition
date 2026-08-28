"""Hand landmark detection using MediaPipe's Tasks API (HandLandmarker).

Wraps a single-hand HandLandmarker instance and exposes a small,
framework-agnostic result type so the rest of the pipeline (gesture
classification, debounce, UI) never has to touch MediaPipe types directly.
See DECISIONS.md (DEC-003, DEC-004) for why MediaPipe and the Tasks API
were selected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarksConnections

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

Landmark = Tuple[float, float, float]


class ModelUnavailableError(RuntimeError):
    """Raised when the hand-landmarker model asset could not be obtained."""


def ensure_model_downloaded(timeout: float = 30.0) -> Path:
    """Return a local path to the hand-landmarker model, downloading it if needed.

    The model is not bundled in the mediapipe pip package (see DEC-004), so
    it is fetched once from a fixed, versioned URL and cached under
    ``models/``. In Docker this is run at build time so the resulting image
    is self-contained and needs no network access at container run time.
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Hand landmarker model not found locally; downloading from %s", MODEL_URL)
    tmp_path = MODEL_PATH.with_suffix(".tmp")
    try:
        with urlopen(MODEL_URL, timeout=timeout) as response:
            data = response.read()
        tmp_path.write_bytes(data)
        tmp_path.replace(MODEL_PATH)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise ModelUnavailableError(
            "Could not download the hand-landmarker model. Check network "
            "connectivity, or pre-fetch it at build time."
        ) from exc

    logger.info("Hand landmarker model downloaded to %s", MODEL_PATH)
    return MODEL_PATH


@dataclass
class HandDetectionResult:
    """Result of running hand detection on a single frame."""

    landmarks: Optional[List[Landmark]]
    """21 (x, y, z) points in normalized [0, 1] image coordinates, or None."""

    handedness: Optional[str]
    """"Left" or "Right" as seen by the camera, or None if no hand."""

    detection_confidence: Optional[float]
    """MediaPipe's own hand-presence confidence score (not a gesture-classification score)."""

    @property
    def hand_present(self) -> bool:
        return self.landmarks is not None


class HandDetector:
    """Detects a single hand and its 21-point landmark set per frame."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        resolved_path = model_path or ensure_model_downloaded()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info("HandLandmarker initialized (num_hands=1)")

    def detect(self, frame_bgr: np.ndarray) -> HandDetectionResult:
        """Run detection on a single BGR frame (as produced by OpenCV/av).

        Never raises: an unexpected MediaPipe or runtime error is logged and
        reported as a "no hand" result so one bad frame cannot crash the
        video pipeline.
        """
        try:
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
        except Exception:
            logger.exception("Unexpected error during hand detection")
            return HandDetectionResult(None, None, None)

        if not result.hand_landmarks:
            return HandDetectionResult(None, None, None)

        landmarks = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]

        handedness_label = None
        confidence = None
        if result.handedness and result.handedness[0]:
            top = result.handedness[0][0]
            handedness_label = top.category_name
            confidence = top.score

        return HandDetectionResult(landmarks, handedness_label, confidence)

    def close(self) -> None:
        self._landmarker.close()


def draw_landmarks(frame_bgr: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
    """Draw the hand skeleton on a copy of the frame, for visual feedback."""
    import cv2  # local import: this is a UI-facing helper, not core detection logic

    h, w = frame_bgr.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    out = frame_bgr.copy()

    for connection in HAND_CONNECTIONS:
        cv2.line(out, points[connection.start], points[connection.end], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(out, (x, y), 4, (0, 120, 255), -1)

    return out
