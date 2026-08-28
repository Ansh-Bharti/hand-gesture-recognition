"""Deterministic tests for gesture/state.py (debounce / event state machine)."""

import pytest

from gesture.state import GestureEvent, GestureStateMachine


def _feed(machine, labels):
    """Feed a sequence of raw labels; return the list of events produced."""
    events = []
    for label in labels:
        event = machine.update(label)
        if event is not None:
            events.append(event)
    return events


def test_holding_a_gesture_fires_exactly_one_event():
    machine = GestureStateMachine(confirm_frames=3)
    # 3 frames to confirm, then hold for many more frames (simulating 30fps x 5s).
    events = _feed(machine, ["thumbs_up"] * 3 + ["thumbs_up"] * 50)
    assert len(events) == 1
    assert events[0].gesture == "thumbs_up"
    assert isinstance(events[0], GestureEvent)


def test_gesture_change_fires_a_new_event():
    machine = GestureStateMachine(confirm_frames=3)
    events = _feed(
        machine,
        ["thumbs_up"] * 3 + ["peace"] * 3,
    )
    assert [e.gesture for e in events] == ["thumbs_up", "peace"]


def test_hand_disappearing_and_reappearing_allows_retrigger():
    machine = GestureStateMachine(confirm_frames=3)
    events = _feed(
        machine,
        ["fist"] * 3  # confirm fist -> event 1
        + [None] * 3  # hand disappears, confirmed state resets
        + ["fist"] * 3,  # same gesture reappears -> event 2
    )
    assert [e.gesture for e in events] == ["fist", "fist"]


def test_no_hand_state_never_itself_produces_an_event():
    machine = GestureStateMachine(confirm_frames=3)
    events = _feed(machine, [None] * 10)
    assert events == []
    assert machine.confirmed_gesture is None


def test_flickering_candidate_never_reaches_confirmation():
    machine = GestureStateMachine(confirm_frames=5)
    # Alternates every frame, so candidate_count never accumulates past 1.
    events = _feed(machine, ["peace", "fist"] * 10)
    assert events == []


def test_unstable_transition_does_not_fire_prematurely():
    machine = GestureStateMachine(confirm_frames=3)
    # Two frames of thumbs_up (not enough to confirm), then switches away.
    events = _feed(machine, ["thumbs_up", "thumbs_up", "peace", "peace", "peace"])
    assert [e.gesture for e in events] == ["peace"]


def test_confirmed_gesture_property_reflects_current_hold():
    machine = GestureStateMachine(confirm_frames=2)
    assert machine.confirmed_gesture is None
    _feed(machine, ["ok_sign", "ok_sign"])
    assert machine.confirmed_gesture == "ok_sign"


def test_candidate_progress_reports_frames_seen():
    machine = GestureStateMachine(confirm_frames=5)
    machine.update("open_palm")
    machine.update("open_palm")
    label, seen, required = machine.candidate_progress
    assert label == "open_palm"
    assert seen == 2
    assert required == 5


def test_reset_clears_all_state():
    machine = GestureStateMachine(confirm_frames=2)
    _feed(machine, ["fist", "fist"])
    assert machine.confirmed_gesture == "fist"

    machine.reset()

    assert machine.confirmed_gesture is None
    label, seen, _ = machine.candidate_progress
    assert label is None and seen == 0


def test_rejects_invalid_confirm_frames():
    with pytest.raises(ValueError):
        GestureStateMachine(confirm_frames=0)
