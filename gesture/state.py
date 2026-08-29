"""Gesture debounce and event-confirmation state machine.

Turns a noisy, per-frame stream of classifier output (a gesture id, or None
for "no hand" / "no recognized gesture") into confirmed, de-duplicated
gesture *events* — the piece of the pipeline responsible for the
assignment's core anti-spam requirement: holding a gesture must fire the
webhook once, not once per frame.

Kept fully independent of Streamlit, MediaPipe, and HTTP so it can be
unit-tested as plain Python (see tests/test_state.py).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _read_confirm_frames_from_env(default: int = 8) -> int:
    """Read GESTURE_CONFIRM_FRAMES from the environment, tolerating bad input.

    This value comes from user-editable configuration (.env), so a typo or
    invalid value must degrade to a safe default with a logged warning,
    not crash the whole application at import time.
    """
    raw = os.getenv("GESTURE_CONFIRM_FRAMES")
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < 1:
            raise ValueError("must be >= 1")
        return value
    except ValueError:
        logger.warning(
            "Invalid GESTURE_CONFIRM_FRAMES=%r in environment; using default of %d",
            raw,
            default,
        )
        return default


DEFAULT_CONFIRM_FRAMES = _read_confirm_frames_from_env()


@dataclass(frozen=True)
class GestureEvent:
    """A confirmed, de-duplicated gesture occurrence."""

    gesture: str
    timestamp: datetime  # UTC, set at confirmation time


class GestureStateMachine:
    """Debounces a raw per-frame gesture stream into confirmed events.

    Algorithm: a candidate label must be seen for `confirm_frames`
    consecutive updates before it is "confirmed." Once confirmed, the same
    label produces no further events until either a different label (or
    None, meaning the hand/gesture disappeared) is itself confirmed for
    `confirm_frames` frames in a row — at which point the confirmed state
    changes and, if the new label is a real gesture, a new event fires.

    Treating None (no hand / unrecognized pose) as just another candidate
    value — rather than a special case — is what makes "gesture holds
    don't repeat" and "gesture can retrigger after the hand reappears"
    fall out of one mechanism instead of two.

    Thread-safe: `update()` is called from the video-processing thread
    while the Streamlit UI thread reads `confirmed_gesture` concurrently.
    """

    def __init__(self, confirm_frames: int = DEFAULT_CONFIRM_FRAMES) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self._confirm_frames = confirm_frames
        self._candidate_label: Optional[str] = None
        self._candidate_count: int = 0
        self._confirmed_label: Optional[str] = None
        self._lock = threading.Lock()

    def update(self, raw_label: Optional[str]) -> Optional[GestureEvent]:
        """Feed one frame's classifier output in; returns a new event, or None."""
        with self._lock:
            if raw_label == self._candidate_label:
                self._candidate_count += 1
            else:
                self._candidate_label = raw_label
                self._candidate_count = 1

            if self._candidate_count < self._confirm_frames:
                return None

            if raw_label == self._confirmed_label:
                return None  # already confirmed and held; this is the debounce

            self._confirmed_label = raw_label
            if raw_label is None:
                return None  # confirmed "empty" state is a reset, not an event

            return GestureEvent(gesture=raw_label, timestamp=datetime.now(timezone.utc))

    @property
    def confirmed_gesture(self) -> Optional[str]:
        with self._lock:
            return self._confirmed_label

    @property
    def candidate_progress(self) -> tuple[Optional[str], int, int]:
        """(candidate_label, frames_seen, frames_required) — for UI feedback."""
        with self._lock:
            return self._candidate_label, self._candidate_count, self._confirm_frames

    def reset(self) -> None:
        with self._lock:
            self._candidate_label = None
            self._candidate_count = 0
            self._confirmed_label = None
