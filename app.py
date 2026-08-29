"""Streamlit entry point for the Real-Time Hand Gesture Detector.

Webcam capture is implemented with streamlit-webrtc: the browser captures
the camera feed via WebRTC and streams frames to the `_on_frame` callback
below, which runs on a background worker thread managed by streamlit-webrtc
(not the Streamlit script-rerun thread). See DECISIONS.md (DEC-002) for why
this approach was chosen over cv2.VideoCapture(0) or st.camera_input.
"""

import logging
import os
import threading
import time

# Loaded before any project module: gesture/state.py and services/webhook.py
# read GESTURE_CONFIRM_FRAMES / WEBHOOK_TIMEOUT_SECONDS from the environment
# at import time, so .env must be applied to os.environ first or those
# values would silently fall back to their defaults even when a real .env
# file is present.
from dotenv import load_dotenv

load_dotenv()

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from gesture.classifier import SUPPORTED_GESTURES, classify_gesture
from gesture.detector import HandDetector, ModelUnavailableError, draw_landmarks
from gesture.state import GestureStateMachine
from services.webhook import WebhookResult, send_webhook_async
from utils.validation import validate_webhook_url


def _resolve_log_level(default: str = "INFO") -> int:
    raw = os.getenv("LOG_LEVEL", default).upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        logging.warning("Invalid LOG_LEVEL=%r in environment; using %s", raw, default)
        return getattr(logging, default)
    return level


logging.basicConfig(
    level=_resolve_log_level(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gesture_app")

GESTURE_DISPLAY = {g["id"]: f"{g['emoji']} {g['label']}" for g in SUPPORTED_GESTURES}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Status polling interval for the "Detection Status" panel below. Fast
# enough to feel live, slow enough not to busy-loop the script thread.
_STATUS_POLL_INTERVAL_SECONDS = 0.2

_detector_instance: HandDetector | None = None
_detector_init_failed = False
_detector_lock = threading.Lock()

# One gesture state machine per process: this is a single-operator local
# app (one webcam, one user), so a module-level singleton is the simplest
# option that satisfies the requirement without introducing per-session
# plumbing that nothing here needs.
_gesture_state = GestureStateMachine()


class _LiveStatus:
    """Thread-safe holder for the latest per-frame detection status.

    Written from the video-processing (worker) thread in `_on_frame`,
    read from the main Streamlit script thread in the status-panel loop
    in `main()`. A plain module-level variable would be read/written
    across threads without any ordering guarantee; this wraps both sides
    in one lock so a reader never observes a half-updated pair of values.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.hand_present = False
        self.detection_confidence: float | None = None

    def update(self, hand_present: bool, confidence: float | None) -> None:
        with self._lock:
            self.hand_present = hand_present
            self.detection_confidence = confidence

    def snapshot(self) -> tuple[bool, float | None]:
        with self._lock:
            return self.hand_present, self.detection_confidence


_live_status = _LiveStatus()


class _WebhookConfig:
    """Thread-safe holder bridging the webhook URL input to the worker thread.

    The URL text input lives in the main Streamlit script thread; the
    gesture event that needs it fires from the video-processing worker
    thread. Streamlit's own st.session_state is not a safe cross-thread
    channel, so the current value is mirrored into this plain,
    lock-protected object every script rerun and read from there instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.url = ""
        self.last_result: WebhookResult | None = None
        self.last_gesture: str | None = None

    def set_url(self, url: str) -> None:
        with self._lock:
            self.url = url

    def get_url(self) -> str:
        with self._lock:
            return self.url

    def record_result(self, gesture: str, result: WebhookResult) -> None:
        with self._lock:
            self.last_gesture = gesture
            self.last_result = result

    def snapshot(self) -> tuple[str, str | None, WebhookResult | None]:
        with self._lock:
            return self.url, self.last_gesture, self.last_result


_webhook_config = _WebhookConfig()


def get_detector() -> HandDetector | None:
    """Build the HandDetector once per process and reuse it on every call.

    Streamlit re-executes this script top-to-bottom on every UI interaction
    (e.g. typing in the webhook field), and the video callback fires ~30
    times per second — so the detector is memoized in a plain module-level
    singleton rather than reconstructed per call. A simple st.cache_resource
    decorator was considered but rejected: it does not cache exceptions, so
    a persistent model-download failure would retry a full network fetch on
    every single video frame. Here, a failed load is remembered and returns
    None immediately on every subsequent call instead of retrying.
    """
    global _detector_instance, _detector_init_failed

    if _detector_instance is not None:
        return _detector_instance
    if _detector_init_failed:
        return None

    with _detector_lock:
        if _detector_instance is not None:
            return _detector_instance
        if _detector_init_failed:
            return None
        try:
            _detector_instance = HandDetector()
        except ModelUnavailableError:
            logger.exception("Hand detector unavailable; disabling hand detection")
            _detector_init_failed = True
            return None
        except Exception:
            # Anything else (e.g. a corrupted cached model file, an
            # unexpected MediaPipe initialization error) is still a
            # detector-unavailable condition from the app's point of view,
            # not a reason to crash the whole page on first load.
            logger.exception("Unexpected error initializing the hand detector; disabling hand detection")
            _detector_init_failed = True
            return None

    return _detector_instance


def _on_frame(frame: av.VideoFrame) -> av.VideoFrame:
    """Per-frame callback invoked by streamlit-webrtc on its worker thread.

    Must never raise: an unhandled exception here would kill the WebRTC
    worker thread and silently freeze the video feed. Any processing error
    is logged and the original frame is passed through unchanged so the
    live feed keeps running even if a downstream step misbehaves.
    """
    try:
        img = frame.to_ndarray(format="bgr24")
        # mirror, so the feed matches the user's motion; made contiguous
        # since cv2 drawing calls below reject the negative-stride view.
        img = np.ascontiguousarray(img[:, ::-1, :])

        label = "No hand detected"
        raw_gesture = None
        detector = get_detector()
        if detector is not None:
            detection = detector.detect(img)
            _live_status.update(detection.hand_present, detection.detection_confidence)
            if detection.hand_present:
                img = draw_landmarks(img, detection.landmarks)
                raw_gesture = classify_gesture(detection.landmarks)
                label = GESTURE_DISPLAY.get(raw_gesture, "Gesture not recognized") if raw_gesture else "Gesture not recognized"

        event = _gesture_state.update(raw_gesture)
        if event is not None:
            logger.info("Gesture confirmed: %s", event.gesture)
            webhook_url = _webhook_config.get_url()
            if webhook_url.strip():
                validation_error = validate_webhook_url(webhook_url)
                if validation_error:
                    logger.warning("Webhook not sent for %s: %s", event.gesture, validation_error)
                    _webhook_config.record_result(
                        event.gesture,
                        WebhookResult(
                            success=False,
                            status_code=None,
                            error=validation_error,
                            sent_at=event.timestamp,
                        ),
                    )
                else:
                    logger.info("Dispatching webhook for gesture: %s", event.gesture)
                    send_webhook_async(
                        webhook_url,
                        event.gesture,
                        event.timestamp,
                        on_result=lambda result, g=event.gesture: _webhook_config.record_result(g, result),
                    )
            else:
                logger.info("Gesture confirmed (%s) but no webhook URL configured", event.gesture)

        cv2.putText(
            img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception:
        logger.exception("Unexpected error while processing a video frame")
        return frame


def _render_status(container, ctx) -> None:
    """Render the live Detection Status panel.

    streamlit-webrtc's frame callback runs on a background thread, so
    Streamlit widgets can't be updated directly from inside `_on_frame`.
    Instead, following streamlit-webrtc's own recommended pattern, the
    main script thread polls the thread-safe shared state on a short
    interval and redraws a single placeholder in place for as long as the
    stream is active.
    """
    placeholder = container.empty()

    if ctx is None or not ctx.state.playing:
        placeholder.warning(
            "Camera inactive. Click **START** above and allow camera "
            "access when your browser prompts you."
        )
        return

    logger.info("Camera stream started")
    while ctx.state.playing:
        hand_present, confidence = _live_status.snapshot()
        confirmed = _gesture_state.confirmed_gesture
        webhook_url, last_gesture, last_result = _webhook_config.snapshot()

        with placeholder.container():
            st.success("Camera active")
            if hand_present:
                conf_text = f" ({confidence:.0%} detection confidence)" if confidence is not None else ""
                st.success(f"Hand detected{conf_text}")
            else:
                st.info("No hand detected")

            if confirmed is not None:
                st.metric("Current Gesture", GESTURE_DISPLAY.get(confirmed, confirmed))
            else:
                st.metric("Current Gesture", "—")

            if not webhook_url.strip():
                st.info("Webhook: not configured")
            elif last_result is None:
                st.info("Webhook: configured, no events sent yet")
            elif last_result.success:
                st.success(
                    f"Last webhook: Success — {GESTURE_DISPLAY.get(last_gesture, last_gesture)} "
                    f"(HTTP {last_result.status_code})"
                )
            else:
                st.error(
                    f"Last webhook: Failed — {GESTURE_DISPLAY.get(last_gesture, last_gesture)} "
                    f"({last_result.error})"
                )

        time.sleep(_STATUS_POLL_INTERVAL_SECONDS)

    logger.info("Camera stream stopped")


def main() -> None:
    logger.info("Application starting")
    st.set_page_config(page_title="Hand Gesture Detector", page_icon=":wave:", layout="wide")
    st.title("Real-Time Hand Gesture Detector")

    with st.spinner("Loading hand detection model..."):
        detector_ready = get_detector() is not None
    if not detector_ready:
        st.error(
            "Hand detection model could not be loaded (likely no network "
            "access on first run). The webcam feed will still work, but "
            "gestures cannot be detected until this is resolved."
        )

    st.subheader("Webhook Configuration")
    webhook_url = st.text_input(
        "Webhook URL",
        key="webhook_url_input",
        placeholder="https://example.com/webhook",
        help=(
            "Called once per confirmed gesture with a JSON POST: "
            '{"event": "gesture_detected", "gesture": "...", "timestamp": "..."}'
        ),
    )
    _webhook_config.set_url(webhook_url)
    if webhook_url:
        validation_error = validate_webhook_url(webhook_url)
        if validation_error:
            st.error(validation_error)
        else:
            st.caption("✓ URL looks valid.")
    st.divider()

    st.subheader("Supported Gestures")
    st.write(" | ".join(f"{g['emoji']} {g['label']}" for g in SUPPORTED_GESTURES))
    st.divider()

    # Everything above this point is static and must render immediately.
    # _render_status below loops for as long as the camera is active, so
    # anything placed after it in the script would never appear until the
    # user stops the stream — it is deliberately the last thing in main().
    video_col, info_col = st.columns([2, 1])

    with video_col:
        st.subheader("Webcam Feed")
        try:
            ctx = webrtc_streamer(
                key="gesture-detector",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=_on_frame,
                async_processing=True,
            )
        except Exception:
            logger.exception("Failed to initialize the webcam component")
            ctx = None
            st.error(
                "Could not initialize the webcam component. "
                "Reload the page and try again."
            )
        st.caption(
            "Camera access is requested by your browser, not this server — "
            "if the prompt does not appear, check your browser's site "
            "permissions for the camera."
        )

    with info_col:
        st.subheader("Detection Status")
        _render_status(info_col, ctx)


if __name__ == "__main__":
    main()
