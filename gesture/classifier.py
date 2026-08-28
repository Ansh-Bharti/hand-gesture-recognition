"""Rule-based gesture classification from hand landmark geometry.

Deliberately decoupled from MediaPipe and the detector: this module takes a
plain list of 21 (x, y, z) normalized landmark points (as produced by
gesture.detector.HandDetector, or any equivalent source) and returns a
gesture label using only relative distances and angles between landmarks.
Nothing here touches pixels, absolute coordinates, or camera state, which
is what makes it independently testable with synthetic landmark data (see
tests/test_classifier.py) and robust to hand position/scale in the frame.

There is no numeric "confidence" produced by this module: classification is
deterministic boolean geometry, not a learned model, so a fabricated
confidence score would be misleading. Hand *detection* confidence (a real
score from MediaPipe) lives on HandDetectionResult, one layer up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

Landmark = Tuple[float, float, float]

# MediaPipe's 21-point hand landmark indices.
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Angle (degrees) at a finger's PIP joint above which it is considered
# straight/extended. A fully folded finger measures close to 0 degrees
# with this same formula; a fully straight one measures close to 180.
_EXTENDED_ANGLE_THRESHOLD_DEG = 150.0

# Thumb extension and OK-sign "pinch" distances are compared against the
# hand's own scale (wrist-to-middle-MCP distance) rather than fixed
# fractions of the frame, so classification stays valid regardless of how
# close the hand is to the camera.
_THUMB_EXTENDED_RATIO = 1.15
_OK_PINCH_RATIO = 0.55
_THUMB_UP_VERTICAL_RATIO = 0.35

OPEN_PALM = "open_palm"
FIST = "fist"
THUMBS_UP = "thumbs_up"
PEACE = "peace"
OK_SIGN = "ok_sign"
POINTING = "pointing"

# Single source of truth for supported gestures, used by the UI and README
# generation alike so the documented list can never drift from the code.
SUPPORTED_GESTURES = [
    {"id": OPEN_PALM, "label": "Open Palm", "emoji": "🖐️"},
    {"id": FIST, "label": "Fist", "emoji": "✊"},
    {"id": THUMBS_UP, "label": "Thumbs Up", "emoji": "👍"},
    {"id": PEACE, "label": "Peace / Victory", "emoji": "✌️"},
    {"id": OK_SIGN, "label": "OK Sign", "emoji": "👌"},
    {"id": POINTING, "label": "Pointing", "emoji": "☝️"},
]


@dataclass
class FingerStates:
    """Extension state of each finger, the intermediate step before classification."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


def _distance(a: Landmark, b: Landmark) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Angle ABC (at vertex b) in degrees, using 2D (x, y) coordinates only.

    z is intentionally ignored: MediaPipe's z estimate is far noisier than
    x/y, and 2D angles are enough to tell a straight finger from a bent one.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _finger_extended(landmarks: List[Landmark], mcp: int, pip: int, tip: int) -> bool:
    angle = _angle_deg(landmarks[mcp], landmarks[pip], landmarks[tip])
    return angle > _EXTENDED_ANGLE_THRESHOLD_DEG


def compute_finger_states(landmarks: List[Landmark]) -> FingerStates:
    """Determine which fingers are extended from landmark geometry alone.

    Non-thumb fingers: extended if the joint angle at the PIP is close to
    straight (>150 degrees); folded fingers bend sharply and measure a
    much smaller angle. This is scale- and rotation-tolerant because it
    only compares angles between a finger's own three joints.

    Thumb: the thumb's range of motion doesn't bend the same way (it
    abducts sideways from the palm rather than curling like the other
    fingers), so a distance-based rule is used instead: it is "extended"
    when its tip sits meaningfully farther from the base of the pinky
    (i.e. away from the palm) than the thumb's own base joint does,
    normalized by the hand's own scale.
    """
    hand_size = _distance(landmarks[WRIST], landmarks[MIDDLE_MCP]) or 1e-6

    thumb_tip_dist = _distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
    thumb_base_dist = _distance(landmarks[THUMB_MCP], landmarks[PINKY_MCP])
    thumb = thumb_tip_dist > thumb_base_dist * _THUMB_EXTENDED_RATIO

    return FingerStates(
        thumb=thumb,
        index=_finger_extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP),
        middle=_finger_extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        ring=_finger_extended(landmarks, RING_MCP, RING_PIP, RING_TIP),
        pinky=_finger_extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP),
    )


def _is_ok_pinch(landmarks: List[Landmark], hand_size: float) -> bool:
    return _distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) < hand_size * _OK_PINCH_RATIO


def _thumb_points_up(landmarks: List[Landmark], hand_size: float) -> bool:
    # Image y grows downward, so "up" means a smaller y than the wrist.
    return landmarks[THUMB_TIP][1] < landmarks[WRIST][1] - hand_size * _THUMB_UP_VERTICAL_RATIO


def classify_gesture(landmarks: List[Landmark]) -> Optional[str]:
    """Classify a single hand's 21 landmarks into one of SUPPORTED_GESTURES.

    Returns None when the hand geometry does not confidently match any
    supported gesture (an intermediate hand shape, mid-transition, etc.) -
    this is a legitimate "unrecognized gesture" outcome, distinct from "no
    hand detected" which is handled one layer up by HandDetectionResult.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    hand_size = _distance(landmarks[WRIST], landmarks[MIDDLE_MCP]) or 1e-6
    fingers = compute_finger_states(landmarks)

    # OK sign is checked before the general extension pattern: a pinched
    # thumb+index reads as "folded" under the angle/distance tests above
    # (both tips are pulled in toward each other, not extended outward),
    # so it must be special-cased rather than falling out of that logic.
    if _is_ok_pinch(landmarks, hand_size) and fingers.middle and fingers.ring and fingers.pinky:
        return OK_SIGN

    if fingers.thumb and not any((fingers.index, fingers.middle, fingers.ring, fingers.pinky)):
        if _thumb_points_up(landmarks, hand_size):
            return THUMBS_UP
        return None  # thumb out to the side is not a supported gesture

    if all(fingers.as_tuple()):
        return OPEN_PALM

    if not any(fingers.as_tuple()):
        return FIST

    if fingers.index and fingers.middle and not fingers.ring and not fingers.pinky:
        return PEACE

    if fingers.index and not any((fingers.thumb, fingers.middle, fingers.ring, fingers.pinky)):
        return POINTING

    return None
