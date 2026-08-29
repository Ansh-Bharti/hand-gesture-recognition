FROM python:3.10-slim

# System libraries required at runtime by OpenCV (pulled in transitively by
# mediapipe) and MediaPipe's own native components. python:3.10-slim is
# Debian-based and does not include these by default; without them,
# `import cv2` / `import mediapipe` fail at container start with missing
# shared-library errors (libGL.so.1 etc.), not a Python-level exception.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installed before copying the rest of the source so this (slow) layer is
# only rebuilt when dependencies actually change, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-fetch the MediaPipe hand-landmarker model at build time (see
# DECISIONS.md DEC-004) so the resulting image is self-contained and does
# not need network access at container run time.
RUN python -c "from gesture.detector import ensure_model_downloaded; ensure_model_downloaded()"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
