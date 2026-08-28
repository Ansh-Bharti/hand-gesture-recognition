"""Deterministic tests for gesture/classifier.py using synthetic landmarks.

No camera, model, or MediaPipe dependency is needed here: canonical hand
poses are constructed by hand as plain (x, y, z) coordinate lists, which is
exactly what the assignment asks for ("the classifier should be
independently testable").
"""

import pytest

from gesture.classifier import (
    FIST,
    OK_SIGN,
    OPEN_PALM,
    PEACE,
    POINTING,
    THUMBS_UP,
    classify_gesture,
    compute_finger_states,
)

WRIST = (0.5, 0.95, 0.0)
# (mcp_x, extended) pairs for the four non-thumb fingers, left to right.
_FINGER_X = {"index": 0.42, "middle": 0.5, "ring": 0.58, "pinky": 0.66}
_MCP_Y = 0.7
_PIP_Y = 0.55
_TIP_Y_EXTENDED = 0.25
_TIP_Y_FOLDED = 0.68

THUMB_MCP = (0.35, 0.75, 0.0)
PINKY_MCP = (_FINGER_X["pinky"], _MCP_Y, 0.0)


def _finger_landmarks(x: float, extended: bool):
    mcp = (x, _MCP_Y, 0.0)
    pip = (x, _PIP_Y, 0.0)
    tip_y = _TIP_Y_EXTENDED if extended else _TIP_Y_FOLDED
    tip = (x, tip_y, 0.0)
    dip = ((pip[0] + tip[0]) / 2, (pip[1] + tip[1]) / 2, 0.0)
    return mcp, pip, dip, tip


def build_landmarks(
    thumb_extended: bool = False,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    ok_pinch: bool = False,
):
    """Build a full 21-point landmark list for a canonical hand pose."""
    landmarks = [None] * 21
    landmarks[0] = WRIST

    if thumb_extended:
        thumb_tip = (0.05, 0.5, 0.0)  # far from the palm -> "extended"
    else:
        thumb_tip = (0.55, 0.72, 0.0)  # tucked near the palm -> "folded"
    thumb_ip = (
        (THUMB_MCP[0] + thumb_tip[0]) / 2,
        (THUMB_MCP[1] + thumb_tip[1]) / 2,
        0.0,
    )
    landmarks[1] = THUMB_MCP  # CMC (unused by classifier, kept plausible)
    landmarks[2] = THUMB_MCP
    landmarks[3] = thumb_ip
    landmarks[4] = thumb_tip

    finger_specs = [
        ("index", 5, index_extended),
        ("middle", 9, middle_extended),
        ("ring", 13, ring_extended),
        ("pinky", 17, pinky_extended),
    ]
    for name, base_index, extended in finger_specs:
        mcp, pip, dip, tip = _finger_landmarks(_FINGER_X[name], extended)
        landmarks[base_index] = mcp
        landmarks[base_index + 1] = pip
        landmarks[base_index + 2] = dip
        landmarks[base_index + 3] = tip

    if ok_pinch:
        # Pull thumb and index tips together, close to each other.
        pinch_point = (0.46, 0.5, 0.0)
        landmarks[4] = pinch_point
        landmarks[8] = (pinch_point[0] + 0.02, pinch_point[1], 0.0)

    assert all(lm is not None for lm in landmarks)
    return landmarks


def test_open_palm():
    landmarks = build_landmarks(
        thumb_extended=True,
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
    )
    assert classify_gesture(landmarks) == OPEN_PALM


def test_fist():
    landmarks = build_landmarks()  # everything folded (all defaults False)
    assert classify_gesture(landmarks) == FIST


def test_thumbs_up():
    landmarks = build_landmarks(thumb_extended=True)
    assert classify_gesture(landmarks) == THUMBS_UP


def test_peace_sign():
    landmarks = build_landmarks(index_extended=True, middle_extended=True)
    assert classify_gesture(landmarks) == PEACE


def test_pointing():
    landmarks = build_landmarks(index_extended=True)
    assert classify_gesture(landmarks) == POINTING


def test_ok_sign():
    landmarks = build_landmarks(
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
        ok_pinch=True,
    )
    assert classify_gesture(landmarks) == OK_SIGN


def test_ambiguous_pose_returns_none():
    # Only the ring finger extended matches no supported gesture.
    landmarks = build_landmarks(ring_extended=True)
    assert classify_gesture(landmarks) is None


def test_thumb_out_to_the_side_is_not_thumbs_up():
    landmarks = list(build_landmarks(thumb_extended=True))
    landmarks[4] = (0.05, 0.93, 0.0)  # far from palm but level with the wrist, not "up"
    assert classify_gesture(landmarks) is None


def test_compute_finger_states_matches_pose():
    landmarks = build_landmarks(index_extended=True, middle_extended=True)
    states = compute_finger_states(landmarks)
    assert states.as_tuple() == (False, True, True, False, False)


def test_classify_gesture_rejects_wrong_landmark_count():
    with pytest.raises(ValueError):
        classify_gesture([(0.0, 0.0, 0.0)] * 5)
