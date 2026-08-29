# Real-Time Hand Gesture Detector

A browser-based Streamlit application that uses a laptop webcam to detect hand
gestures in real time and notifies an external webhook when a gesture is
confirmed.

> **Status: complete.** All core functional requirements are implemented and
> covered by 65 automated tests. The app runs locally (`streamlit run app.py`)
> and as a Docker container. It was built in numbered phases with an entry in
> [`DECISIONS.md`](DECISIONS.md) for every significant choice; the
> [Development History](#development-history) table maps phases to commits.

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
| `tests/` | 65 `pytest` tests — unit (`test_classifier`, `test_state`, `test_webhook`, `test_detector`, `test_frame_processing`) and integration (`test_app_smoke` renders the page headlessly, `test_app_pipeline` drives `_on_frame` end to end) |

Each non-UI module is deliberately decoupled from Streamlit and MediaPipe so it
can be tested as plain Python — no camera, model, or network needed in the test
suite.

## Supported Gestures

Six gestures, recognised from landmark geometry alone (no trained model):

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

```bash
docker build -t hand-gesture-detector .
docker run --rm -p 8501:8501 hand-gesture-detector
```

Then open <http://localhost:8501>.

Details:

- Base image `python:3.10-slim`. The `Dockerfile` installs the OS libraries
  MediaPipe/OpenCV need at runtime (`libgl1`, `libglib2.0-0`, …) — the slim
  image doesn't ship them, and without them `import cv2` fails at container
  start.
- `requirements.txt` is copied and installed **before** the rest of the source,
  so editing code doesn't re-run the slow `pip install`.
- The MediaPipe model is downloaded **at build time**
  (`ensure_model_downloaded()`), so the running container needs no outbound
  network. `models/` is in `.dockerignore` so the host's copy is never sent
  into the build.
- A `HEALTHCHECK` polls Streamlit's `/_stcore/health`.
- The webcam is captured **in the browser**, not read from a device inside the
  container, so there is no `--device /dev/video0` mapping (see
  [`DECISIONS.md`](DECISIONS.md) DEC-002 and DEC-012). Because browsers only
  grant camera access on `localhost` or HTTPS, a remote container deployment
  needs a TLS terminator in front of it.
- Config: pass env vars with `-e`, e.g.
  `docker run --rm -p 8501:8501 -e GESTURE_CONFIRM_FRAMES=5 hand-gesture-detector`.

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
| DEC-001 | Target Python 3.10 (dependency wheel availability) |
| DEC-002 | Browser-side WebRTC capture (`streamlit-webrtc`), not `cv2.VideoCapture` or `st.camera_input` |
| DEC-003 | MediaPipe Hands for 21-point landmarks — no custom/trained detector |
| DEC-004 | MediaPipe **Tasks API** `HandLandmarker` in `IMAGE` mode; model asset fetched, not bundled |
| DEC-005 | Detector as a module singleton with a lock + failure flag, not `st.cache_resource` |
| DEC-006 | Rule-based geometric classification, not a trained classifier |
| DEC-007 | `classify_gesture()` returns a label or `None` — never a fabricated confidence score |
| DEC-008 | One `GestureStateMachine` for debounce; `None` (no hand) treated as an ordinary candidate value |
| DEC-009 | Webhook dispatched on a background daemon thread — not async, not a task queue |
| DEC-010 | Webhook URL validation is syntactic only — no SSRF/private-range blocking (single-user local tool) |
| DEC-011 | `load_dotenv()` before the modules that read env vars; bad config values fall back with a warning, never crash |
| DEC-012 | Single-stage `python:3.10-slim` image; model pre-fetched at build time; no webcam device mapping |

## Assumptions

- **One hand in frame at a time** (`num_hands=1`).
- **A single local operator** — one webcam, one browser session. Shared state is
  a process-level singleton, not per-session; the app is not designed for
  concurrent users.
- **A Chromium-based browser or Firefox**, served over `localhost` or HTTPS —
  browsers only grant camera access in a secure context.
- **Outbound network on first run** to fetch the ~7.8 MB MediaPipe model (or the
  Docker image, which bakes it in at build time). No network needed afterward.
- Pointing the webhook at `localhost` / a LAN address is a legitimate use case,
  so no SSRF-style blocking of private/loopback addresses is applied (DEC-010).

## Limitations

- **Thresholds are hand-tuned constants** (extension angle 150°, pinch and
  thumb-distance ratios). Unusual hand shapes, steep camera angles, or lighting that skews
  MediaPipe's landmark estimates may misclassify or fall through to
  "unrecognized"; the constants are named at the top of `gesture/classifier.py`
  for tuning.
- **No confidence score for the gesture label** — classification is boolean
  geometry, not a model. MediaPipe's hand-*detection* confidence is a separate,
  real value shown in the status panel (DEC-007).
- **Webhook delivery has no retry or queue** — a down endpoint means that one
  event is reported as failed and dropped (DEC-009).
- **Debounce lag** — there is a deliberate ~`GESTURE_CONFIRM_FRAMES`-frame delay
  (≈ ⅓ s at 25 fps) between a hand leaving the frame and the gesture being
  treated as cleared, so a single dropped detection frame doesn't break a hold
  (DEC-008).
- **Gesture set is fixed in code** — adding a gesture means adding a rule to
  `classify_gesture()` and an entry to `SUPPORTED_GESTURES`, not configuration.
- **The automated tests cannot exercise real WebRTC** (no browser); live camera
  behaviour is covered by the [manual checklist](#demonstrating--manual-test-checklist).
- **A remote container deployment needs HTTPS** in front of it for the browser
  to allow the camera; plain `http://<server-ip>:8501` will not get camera
  permission.

## Future Improvements

- Webhook retry with backoff and a small delivery queue.
- Multi-hand support.
- On-screen debounce progress indicator (frames-to-confirm meter).
- Configurable gesture set / thresholds via the UI.
- CI workflow running the test suite on every push.
- HMAC-signed webhook payloads so receivers can verify the sender.

## Requirement → Implementation

| Assignment requirement | Where it's met |
| --- | --- |
| Access & process the live webcam feed | `webrtc_streamer` + `_on_frame()` in `app.py` (browser capture, DEC-002) |
| Display the live feed in the UI | "Webcam Feed" panel; frames returned from `_on_frame()` with a mirror + landmark overlay |
| Detect gestures in real time | `gesture/detector.py` (MediaPipe, per frame) → `gesture/classifier.py` |
| Support ≥ 5 gestures | **6** — see [Supported Gestures](#supported-gestures) / `SUPPORTED_GESTURES` |
| Visually display the detected gesture | Text drawn on the video + "Current Gesture" metric in the status panel |
| UI input for a webhook URL | `st.text_input` in "Webhook Configuration", with live validation feedback |
| Send gesture info to the webhook | `services/webhook.py` `send_webhook_async()` — JSON `POST` per confirmed gesture |
| Handle no hand detected | `HandDetectionResult.hand_present` false → "No hand detected"; state machine treats `None` as a valid state |
| Handle invalid webhook URL | `utils/validation.py` — checked live in the UI and again before every send |
| Handle webhook failure | `send_webhook()` never raises; failure returned as `WebhookResult` and shown in the UI + logs |
| Handle camera access issues | `try/except` around `webrtc_streamer()`; `get_detector()` catches every startup error; `_on_frame()` never raises |
| Prevent repeated triggering | `gesture/state.py` `GestureStateMachine` — N-frame confirmation, one event per hold (DEC-008) |
| Python / Streamlit / Git / Docker | Python 3.10; Streamlit UI; phase-by-phase Git history; `Dockerfile` + `.dockerignore` |
| No hardcoded config / secrets | `.env` + `.env.example` + `load_dotenv()`; env values validated with fallbacks (DEC-011); `.env` git-ignored |
| Error handling & logging | Covered above + `logging` throughout (lifecycle, warnings, errors) |
| `.gitignore` excludes generated files / venv / secrets | `.venv/`, `.env`, `__pycache__/`, `*.task` model, `.pytest_cache/` |
| Docker: same functionality as local | Same code; `ENTRYPOINT` runs the identical `streamlit run app.py` |
| README covers purpose, local run, Docker, webhook, gestures, limitations | This document |

## Demonstrating / Manual Test Checklist

The automated suite (`pytest`, 65 tests) covers every module and the
`_on_frame` pipeline without a camera. The items below need a real browser +
webcam and are the manual acceptance pass:

- [ ] `streamlit run app.py` → page loads at `http://localhost:8501`, model spinner clears
- [ ] Click **START**, allow the camera → mirrored live video appears, status shows **Camera active**
- [ ] Hold a hand up → green 21-point skeleton tracks it; status shows **Hand detected (NN% detection confidence)**
- [ ] Each of the six gestures → its name shows on the video and as **Current Gesture**
- [ ] An in-between hand shape → **Gesture not recognized**
- [ ] Move the hand out of frame → **No hand detected**
- [ ] Hold one gesture steady for several seconds → the terminal logs `Gesture confirmed: <name>` **once**, not repeatedly
- [ ] Drop the hand, remake the same gesture → it confirms and logs again
- [ ] Paste a `https://webhook.site` URL → make gestures → each confirmed gesture arrives there as the JSON payload; status shows **Last webhook: Success (HTTP 200)**
- [ ] Paste `not-a-url` → inline **error** message, no request sent
- [ ] Paste a valid-looking but dead URL (e.g. `https://127.0.0.1:9/x`) → status shows **Last webhook: Failed (...)**, app keeps running
- [ ] Deny the camera permission / have no webcam → friendly on-page message, no crash
- [ ] `docker build -t hand-gesture-detector .` then `docker run --rm -p 8501:8501 hand-gesture-detector` → same behaviour at `http://localhost:8501`; `docker ps` shows `(healthy)` after ~30 s

## Development History

Built in numbered phases, one commit each, with a `DECISIONS.md` entry for
every significant choice.

| Phase | Scope | Commit |
| --- | --- | --- |
| 1 | Project scaffold, deps, test harness, config, decision log | `91129e3`, `60615d4` |
| 2 | Webcam interface (`streamlit-webrtc`) | `4a8ae68` |
| 3 | Hand landmark detection (MediaPipe) | `a988ad4` |
| 4 | Rule-based gesture classification (6 gestures) | `a685263` |
| 5 | Debounce / event state machine + live gesture readout | `50f4c64` |
| 6 | Webhook integration + URL validation | `ca7a722` |
| 7 | Error handling, logging, `.env` config audit | `a5d629c` |
| 8 | Test-suite consolidation (integration tests) | `299f5eb` |
| 9 | Docker packaging | `429ed4a` |
| 10 | Documentation finalisation (this pass) | `5e19f5b`+ |

_Commit hashes are from the current history; see `git log` for the authoritative list._
