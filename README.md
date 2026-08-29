# Real-Time Hand Gesture Detector

A browser-based Streamlit application that uses a laptop webcam to detect hand
gestures in real time and notifies an external webhook when a gesture is
confirmed.

> **Project status — Phase 6 (webhook integration).**
> Each confirmed gesture now fires a JSON `POST` to a user-supplied webhook URL:
> `{"event": "gesture_detected", "gesture": "...", "timestamp": "..."}`. The URL
> is validated in the UI as you type; the request goes out on a background
> daemon thread (fire-and-forget) so a slow endpoint never stalls the video;
> success/failure is shown in the status panel and logged (host only, never the
> full URL). All 48 tests pass. The running decision log in
> [`DECISIONS.md`](DECISIONS.md) is the source of truth for *why* each choice
> was made.

## Overview

The finished application:

1. Captures the webcam feed **in the browser** via WebRTC and streams frames to
   the server (`streamlit-webrtc`), so the exact same code path works locally
   and inside a container.
2. Extracts 21 hand landmarks per frame with **MediaPipe Hands** (Tasks API,
   CPU, no GPU required).
3. Classifies the landmark geometry into one of a small fixed set of gestures
   using **deterministic geometric rules** — no trained model, no training data.
4. **Debounces** the noisy per-frame stream through a state machine so that
   holding a gesture fires exactly one event, not one per frame.
5. **POSTs a JSON payload to a configurable webhook URL** on each confirmed
   gesture, fire-and-forget, without ever stalling the video feed.

## Architecture

```
browser webcam ──WebRTC──▶ _on_frame() callback (worker thread)
                                │
                                ▼
                     gesture/detector.py      MediaPipe HandLandmarker → 21 (x,y,z) landmarks
                                │
                                ▼
                     gesture/classifier.py    pure geometry → gesture id | None
                                │
                                ▼
                     gesture/state.py         N-frame confirmation → GestureEvent
                                │
                                ▼
                     services/webhook.py      background thread → HTTP POST
```

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit entry point, WebRTC wiring, UI, per-frame callback |
| `gesture/detector.py` | MediaPipe hand + landmark detection, model download |
| `gesture/classifier.py` | Landmark-geometry → gesture label (framework-agnostic, unit-tested with synthetic points) |
| `gesture/state.py` | Debounce / event-confirmation state machine |
| `services/webhook.py` | Webhook payload build + async dispatch + error handling |
| `utils/validation.py` | Webhook URL validation |
| `tests/` | `pytest` suite for the classifier, state machine, and webhook layers |

Each non-UI module is deliberately decoupled from Streamlit and MediaPipe so it
can be tested as plain Python.

## Supported Gestures

The target gesture vocabulary (implemented in Phase 4):

| Gesture | Rule (informal) |
| --- | --- |
| 🖐️ Open Palm | all five fingers extended |
| ✊ Fist | no fingers extended |
| 👍 Thumbs Up | only the thumb extended, pointing up |
| ✌️ Peace / Victory | index + middle extended, ring + pinky folded |
| 👌 OK Sign | thumb and index tips pinched, other three fingers extended |
| ☝️ Pointing | only the index finger extended |

Any other hand shape classifies as "unrecognized" (distinct from "no hand
detected").

## Requirements

- **Python 3.10** (MediaPipe / `streamlit-webrtc` ship reliable wheels for
  3.9–3.11; see [`DECISIONS.md`](DECISIONS.md) DEC-001)
- A webcam
- A modern browser (camera access requires a secure context — `localhost` or
  HTTPS)
- Docker (optional, for the containerized run)
- Network access on first run, to download the ~7.8 MB MediaPipe
  hand-landmarker model (cached locally afterward)

## Local Setup

```bash
# from the repository root
py -3.10 -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # runtime + pytest
```

Optionally copy the environment template and edit values:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `GESTURE_CONFIRM_FRAMES` | `8` | consecutive matching frames required to confirm a gesture |
| `WEBHOOK_TIMEOUT_SECONDS` | `5` | HTTP timeout for outbound webhook POSTs |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## Running Locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>) and allow
camera access when the browser prompts.

## Running the Tests

```bash
pytest
```

## Docker Setup

_Placeholder until the Docker phase._ The intended usage:

```bash
docker build -t hand-gesture-detector .
docker run --rm -p 8501:8501 hand-gesture-detector
```

The webcam is captured in the browser, **not** read from a device inside the
container, so no `--device /dev/video0` mapping is needed (see
[`DECISIONS.md`](DECISIONS.md) DEC-002). The model file is fetched at image
build time so the container needs no network access at run time.

## Webhook Configuration

Enter an `http(s)` URL in the app's **Webhook URL** field. On each confirmed
gesture the app sends:

```http
POST <your-url>
Content-Type: application/json

{
  "event": "gesture_detected",
  "gesture": "thumbs_up",
  "timestamp": "2026-08-28T12:34:56Z"
}
```

Delivery is fire-and-forget on a background thread. Failures (bad URL,
unreachable endpoint, timeout, non-2xx response) are surfaced in the UI and
logs; there is no automatic retry.

## Error Handling

- The per-frame callback never raises — any processing error is logged and the
  original frame is passed through so the live feed keeps running.
- A failed model download is remembered: the app reports it once at startup
  instead of retrying the fetch on every frame.
- Malformed `.env` values fall back to their defaults with a logged warning
  rather than crashing at import.
- Webhook errors are captured and reported, never propagated into the video
  pipeline.

## Design Decisions

See [`DECISIONS.md`](DECISIONS.md) for the full log with alternatives and
trade-offs. In brief:

| ID | Decision |
| --- | --- |
| DEC-001 | Target Python 3.10 |
| DEC-002 | Browser-side WebRTC capture (`streamlit-webrtc`), not `cv2.VideoCapture` |
| DEC-003 | MediaPipe Hands for 21-point landmarks (no custom/trained detector) |
| DEC-004 | MediaPipe **Tasks API** `HandLandmarker` in `IMAGE` mode |

Later phases add decisions on the detector lifecycle, rule-based
classification, not fabricating a classifier confidence score, the debounce
state machine, and the webhook dispatch model.

## Assumptions

- One hand in frame at a time (`num_hands=1`).
- A single local operator — one webcam, one browser session. State is a
  process-level singleton, not per-session.
- Pointing the webhook at `localhost` is a legitimate use case; no SSRF-style
  blocking of private/loopback addresses is applied.

## Limitations

- Gesture thresholds are hand-tuned constants; unusual hand shapes, camera
  angles, or lighting that skews MediaPipe's landmark estimates may need them
  adjusted.
- No confidence score for the gesture label itself — classification is boolean
  geometry. MediaPipe's hand-*detection* confidence is a separate, real value.
- Webhook delivery has no retry or queue; a down endpoint means that one event
  is reported as failed and dropped.
- There is a small deliberate lag (≈ `GESTURE_CONFIRM_FRAMES` frames) between a
  hand leaving the frame and the gesture being treated as cleared.

## Future Improvements

- Webhook retry with backoff and a small delivery queue.
- Multi-hand support.
- On-screen debounce progress indicator.
- Configurable gesture set / thresholds via the UI.
- CI workflow running the test suite on push.

## Roadmap

| Phase | Scope | State in this snapshot |
| --- | --- | --- |
| 1 | Project scaffold, deps, test harness, config, decision log | ✅ done |
| 2 | Webcam interface (`streamlit-webrtc`) | ✅ done |
| 3 | Hand landmark detection (MediaPipe) | ✅ done |
| 4 | Rule-based gesture classification | ✅ done |
| 5 | Debounce / event state machine | ✅ done |
| 6 | Webhook integration + URL validation | ✅ done |
| 7 | Error handling, logging, config audit | pending |
| 8 | Automated test suite | pending |
| 9 | Docker packaging | placeholder |
