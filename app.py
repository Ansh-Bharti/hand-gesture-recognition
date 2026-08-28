"""Streamlit entry point for the Real-Time Hand Gesture Detector.

Webcam capture is implemented with streamlit-webrtc: the browser captures
the camera feed via WebRTC and streams frames to the `_on_frame` callback
below, which runs on a background worker thread managed by streamlit-webrtc
(not the Streamlit script-rerun thread). See DECISIONS.md (DEC-002) for why
this approach was chosen over cv2.VideoCapture(0) or st.camera_input.
"""

import logging

import av
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gesture_app")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def _on_frame(frame: av.VideoFrame) -> av.VideoFrame:
    """Per-frame callback invoked by streamlit-webrtc on its worker thread.

    Must never raise: an unhandled exception here would kill the WebRTC
    worker thread and silently freeze the video feed. Any processing error
    is logged and the original frame is passed through unchanged so the
    live feed keeps running even if a downstream step misbehaves.
    """
    try:
        img = frame.to_ndarray(format="bgr24")
        img = img[:, ::-1, :]  # mirror, so the feed matches the user's motion
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception:
        logger.exception("Unexpected error while processing a video frame")
        return frame


def main() -> None:
    st.set_page_config(page_title="Hand Gesture Detector", page_icon=":wave:", layout="wide")
    st.title("Real-Time Hand Gesture Detector")

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

    with info_col:
        st.subheader("Detection Status")
        if ctx is not None and ctx.state.playing:
            st.success("Camera active")
        else:
            st.warning(
                "Camera inactive. Click **START** above and allow camera "
                "access when your browser prompts you."
            )

    st.divider()
    st.caption(
        "Camera access is requested by your browser, not this server — "
        "if the prompt does not appear, check your browser's site "
        "permissions for the camera."
    )


if __name__ == "__main__":
    main()
