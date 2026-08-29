# Decision Log

## DEC-001 — Python Runtime Version

### Decision
Target Python 3.10 for the virtual environment and Docker image, rather than the system default (3.14).

### Alternatives Considered
- Python 3.14 (system default on the dev machine)
- Python 3.13
- Python 3.10
- Python 3.9

### Reason
MediaPipe and streamlit-webrtc both ship prebuilt wheels for 3.9–3.11 reliably; 3.14 is too new for MediaPipe's prebuilt wheels at the time of writing, which would force a from-source build (slow, fragile, not reproducible). 3.10 is a safe, well-supported middle ground for every dependency in this project.

### Trade-off
The project cannot use the newest CPython language features. Irrelevant for this assignment's scope.

### Date
2026-08-27

---

## DEC-002 — Webcam Capture Architecture

### Decision
Use `streamlit-webrtc` to capture the webcam feed in the browser via WebRTC and stream frames to a server-side callback, instead of `cv2.VideoCapture(0)` or `st.camera_input`.

### Alternatives Considered
- `cv2.VideoCapture(0)` directly in the Streamlit process
- `st.camera_input` (single-shot snapshot widget)
- `streamlit-webrtc` (browser-mediated WebRTC stream)

### Reason
The assignment explicitly warns that a webcam device inside Docker (`/dev/video0`) does not represent the user's actual browser webcam. `cv2.VideoCapture(0)` only works when the Streamlit *server process* has direct OS-level access to the camera — true for local `streamlit run` on a laptop, but false the moment the app is containerized or accessed from a different machine. `st.camera_input` only captures on-demand single frames, not a continuous real-time stream. `streamlit-webrtc` captures video in the browser (which always has access to the user's camera, with permission) and streams frames over the network to the server for processing — this is architecturally identical whether the server is local or in a container, so the same code path works in both environments.

### Trade-off
Adds a heavier dependency chain (`aiortc`, `av`/PyAV) and requires STUN/ICE negotiation, which is more complex than a plain `cv2.VideoCapture` loop. WebRTC also requires a secure context (`localhost` or HTTPS) for browser camera permissions, which must be documented as a deployment constraint.

### Date
2026-08-27

---

## DEC-003 — Hand Landmark Detection

### Decision
Use MediaPipe Hands for hand detection and 21-point landmark extraction.

### Alternatives Considered
- MediaPipe Hands
- OpenCV-only contour/skin-color detection
- YOLO-based hand detector
- Custom-trained CNN

### Reason
The assignment needs a small, fixed gesture vocabulary recognized reliably on CPU with no dedicated GPU. MediaPipe Hands provides stable, well-tested 21-point landmarks per hand out of the box, runs comfortably on CPU in real time, and requires no training data or model-building effort. Contour/skin-color detection is highly sensitive to lighting and background and cannot produce landmark geometry. YOLO-hand and a custom CNN both require building or sourcing a labeled training pipeline, which is exactly the "unnecessary ML infrastructure" the assignment tells us to avoid for a fixed gesture set.

### Trade-off
Gesture classification becomes fully dependent on landmark geometry quality; MediaPipe's hand detection confidence is a real score, but there is no equivalent confidence score for our own rule-based gesture step (documented explicitly, not invented).

### Date
2026-08-27

---

## DEC-004 — MediaPipe API Surface: Tasks API vs. Legacy `solutions.hands`

### Decision
Use MediaPipe's newer unified **Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`) instead of the older `mp.solutions.hands.Hands` API, running in `IMAGE` mode (stateless, one call per frame).

### Alternatives Considered
- Legacy `mp.solutions.hands.Hands` (bundled model, no separate asset file)
- Tasks API `HandLandmarker` in `IMAGE` mode
- Tasks API `HandLandmarker` in `LIVE_STREAM` mode (async callback-based)

### Reason
Verified by directly importing `mediapipe==1.0.1` in this environment: the `mp.solutions` namespace no longer exists in current MediaPipe releases — only `mediapipe.tasks` is available. This is not optional; the legacy API is gone from the package we installed. The Tasks API requires an explicit `.task` model asset (downloaded once from Google's model store, ~7.8 MB, confirmed reachable and functional against a real CPU inference call in this environment). `IMAGE` mode was chosen over `LIVE_STREAM` because our own frame callback already delivers frames one at a time synchronously — `IMAGE` mode's per-call, stateless semantics match that directly, whereas `LIVE_STREAM` mode adds its own async result callback and internal timestamp bookkeeping that would duplicate work our architecture already does elsewhere (the debounce state machine).

### Trade-off
The model file is not bundled in the pip package, so the app needs a one-time network fetch to obtain it (cached locally afterward). This is handled by a small downloader in `gesture/detector.py` and pre-fetched at Docker build time so the built image is self-contained and does not need network access at container run time.

### Date
2026-08-27

---

## DEC-005 — Detector Lifecycle: Module Singleton vs. `st.cache_resource`

### Decision
Build the `HandDetector` once and keep it in a plain module-level singleton in `app.py` (`get_detector()`), guarded by a `threading.Lock`, with an explicit "init failed" flag. Do not use Streamlit's `st.cache_resource`.

### Alternatives Considered
- Construct a new `HandDetector` inside `_on_frame` every frame
- `@st.cache_resource` on a `get_detector()` factory
- Module-level singleton with an explicit failure flag (chosen)

### Reason
Streamlit re-runs the whole script on every UI interaction and the video callback fires ~30 times a second, so the detector must be created once and reused. Both a singleton and `st.cache_resource` achieve that; the deciding factor is failure handling. `st.cache_resource` does not cache exceptions, so if the model load fails (e.g. no network on first run) every call retries the full download — at ~30 calls/second. The module singleton records the failure once, then returns `None` immediately on every later call, and the app surfaces one clear error at startup. The lock guards only the construction path, not the fast-path reads.

### Trade-off
Slightly more code than a one-line decorator, and thread-safety is now our responsibility because the video callback runs on a different thread from the main Streamlit script.

### Date
2026-08-28

---

## DEC-006 — Gesture Classification Strategy

### Decision
Classify gestures with deterministic, rule-based geometry over the 21 landmarks (finger-extension angles, thumb/pinch distances, a directional check for thumbs-up), instead of training a classifier model.

### Alternatives Considered
- Rule-based geometry over MediaPipe landmarks
- A small trained classifier (e.g. an SVM or shallow MLP) on landmark features
- End-to-end image classification (CNN on raw frames)

### Reason
The assignment fixes a small, known gesture vocabulary (6 gestures here: Open Palm, Fist, Thumbs Up, Peace, OK Sign, Pointing). For a fixed vocabulary like this, hand-written geometric rules are fully explainable (each gesture maps to a specific, statable condition on finger states), require no training data or labeling effort, and are trivially unit-testable with synthetic landmark coordinates rather than recorded video. A trained classifier would need a labeled dataset the assignment does not provide and would trade explainability for marginal accuracy gains that aren't needed at this vocabulary size.

### Trade-off
Thresholds (e.g. the 150° extension-angle cutoff, the thumb/pinch distance ratios) are hand-tuned constants rather than learned, so they may need adjustment for unusual hand shapes, camera angles, or lighting that skews MediaPipe's own landmark estimates. This is called out explicitly in the README's Limitations section rather than presented as universally robust.

### Date
2026-08-28

---

## DEC-007 — No Fabricated Gesture-Classification Confidence

### Decision
`classify_gesture()` returns a label or `None`; it never returns a numeric confidence score.

### Alternatives Considered
- Return a synthetic confidence derived from, e.g., how far past the angle threshold a finger's measurement is
- Return label-only, with hand-*detection* confidence (from MediaPipe) reported separately

### Reason
The assignment explicitly warns against inventing ML confidence values for a rule-based classifier. There is a real confidence score for hand *detection* (`HandDetectionResult.detection_confidence`, sourced directly from MediaPipe's handedness classification), and the two are kept clearly distinct so the UI and webhook payload never imply a model-derived certainty that doesn't exist for the gesture label itself.

### Trade-off
None of substance — this is a correctness/honesty decision, not a capability trade-off.

### Date
2026-08-28

---

## DEC-008 — Debounce Strategy: Unified State Machine, "No Hand" as a Valid State

### Decision
Implement gesture debounce as a single `GestureStateMachine` (`gesture/state.py`) that requires N consecutive matching classifier outputs to "confirm" a state change, where `None` (no hand / unrecognized pose) is treated as just another candidate value rather than a special case.

### Alternatives Considered
- Separate boolean flags/counters scattered across the frame callback ("already_fired", "frames_since_change", etc.)
- A dedicated state machine class, with `None` handled as an ordinary candidate value
- A dedicated state machine class, with `None` handled as a special "reset now" signal (immediate reset on any no-hand frame, no debounce on the reset path)

### Reason
Treating `None` identically to every other gesture label means one mechanism produces all the required behaviors: holding a gesture emits exactly one event, switching to a new gesture (or to "no gesture") requires the same N-frame confirmation as any other transition (so a single dropped frame of detection doesn't spuriously reset the whole hold), and the same gesture reappearing after the hand was confirmed absent for N frames is free to fire again. Scattering ad-hoc counters directly in the video callback (the initial mental draft) was rejected because the debounce logic would not be unit-testable independently of MediaPipe/Streamlit, and the assignment explicitly calls for "a clean state machine or dedicated component."

### Trade-off
Requiring N consecutive `None` frames before resetting (rather than resetting instantly) means there is a small, deliberate lag between the hand actually leaving the frame and the UI/webhook treating the gesture as "gone." At the default `GESTURE_CONFIRM_FRAMES=8` and a typical 20-30fps stream, that's roughly a quarter-to-a-third of a second — judged an acceptable trade for not being oversensitive to single-frame detection dropouts.

### Date
2026-08-28

---

## DEC-009 — Webhook Dispatch: Background Thread, Not Async/Queue

### Decision
Dispatch each confirmed gesture's webhook POST on a plain daemon `threading.Thread` (`send_webhook_async` in `services/webhook.py`), fire-and-forget, rather than using `asyncio`, a task queue (Celery/RQ), or calling `requests.post` synchronously inline.

### Alternatives Considered
- Synchronous call inline in the video callback
- `asyncio` + an async HTTP client (`aiohttp`/`httpx`)
- A task queue (Celery, RQ) backed by Redis or similar
- A background thread per event (chosen)

### Reason
The video callback must return quickly to keep the live feed smooth; blocking it on a network call (up to the configured timeout, default 5s) for a slow or unresponsive webhook endpoint would visibly stall the video. By the time an event reaches the webhook layer it has already been debounced (Phase 5) — one event per confirmed gesture, not one per frame — so the dispatch rate is inherently low (at most a few per second in pathological rapid-gesture-switching use, realistically much less). At that rate, a plain background thread per event has no meaningful resource cost, and is far simpler to write, test, and explain than introducing an async runtime or a message broker into what is otherwise a synchronous Streamlit script. Streamlit itself does not run an asyncio event loop for user script code, so `asyncio` would have needed its own thread anyway.

### Trade-off
No retry-on-failure and no delivery queue/backpressure: if the endpoint is down, that single event's webhook is simply reported as failed (visible in the UI and logs) and the app moves on. Acceptable for this assignment's scope; documented as a Future Improvement rather than silently assumed away.

### Date
2026-08-28

---

## DEC-010 — Webhook URL Validation Scope (no SSRF hardening)

### Decision
`validate_webhook_url()` checks only that the URL is well-formed and uses an `http`/`https` scheme with a host. It deliberately does **not** block private, loopback, or link-local addresses.

### Alternatives Considered
- Syntactic validation only (chosen)
- Additionally resolve the host and reject RFC1918 / loopback / metadata IPs (SSRF hardening)

### Reason
This is a single-user local tool: the person typing the webhook URL is the same person running the app, and pointing it at `http://localhost:9000` or a LAN address is a normal, expected way to test with a local receiver. SSRF protections exist to stop an *untrusted* caller from reaching internal services; there is no untrusted caller here. Adding DNS resolution to validation would also make a cheap synchronous check slow and network-dependent.

### Trade-off
If this app were ever exposed as a multi-tenant service, this validation would be insufficient and would need address-range blocking added. Called out here and in the README Limitations section rather than left implicit.

### Date
2026-08-28

---

## DEC-011 — Configuration Loading Order and Env-Var Validation

### Decision
Call `load_dotenv()` at the very top of `app.py`, before importing `gesture.state` or `services.webhook` (both of which read env vars at module-import time), and validate every env-derived config value at read time with a safe fallback + logged warning instead of letting a parse error crash at import.

### Alternatives Considered
- Call `load_dotenv()` wherever convenient in `app.py`
- Read env vars lazily (inside functions) instead of at module import time
- Read env vars at import time with a fallback, called before other imports (chosen)

### Reason
Caught during a deliberate Phase 7 audit, not by accident: `python-dotenv` was listed in `requirements.txt` and documented in `.env.example` since Phase 1, but `load_dotenv()` was never actually called anywhere — a real `.env` file would have had zero effect. Fixing the call alone was not sufficient: `gesture/state.py` and `services/webhook.py` compute their default constants (`DEFAULT_CONFIRM_FRAMES`, `DEFAULT_TIMEOUT_SECONDS`) once, at module-import time, so `load_dotenv()` must run *before* those modules are imported, or the values would already have been read (and missed) by the time `.env` was applied. Separately, a malformed value (e.g. `GESTURE_CONFIRM_FRAMES=abc`) previously crashed at import with an unhandled `ValueError` — a corrupted single config field taking down the whole app on startup is a worse failure mode than falling back to the built-in default with a logged warning.

### Trade-off
Reading configuration once at import time (rather than per-call) means a `.env` change requires restarting the app to take effect — acceptable for a local single-user tool where restart is cheap and expected.

### Date
2026-08-28
