# Base image for JetPack 6
FROM ultralytics/ultralytics:latest-jetson-jetpack6

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ---------------------------------------------------------------------------
# 1. Install System Dependencies (Added Qt/XCB libs for GUI support)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gir1.2-aravis-0.8 \
    libaravis-dev \
    aravis-tools \
    python3-gi \
    python3-gi-cairo \
    gobject-introspection \
    libgirepository1.0-dev \
    libopencv-dev \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    # --- GUI SUPPORT LIBS START ---
    libgl1 \
    libqt5gui5 \
    libqt5widgets5 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libgtk-3-0 \
    # --- GUI SUPPORT LIBS END ---
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2 & 3. Python Dependencies (Merged)
#    - Uninstall numpy first to clear artifacts
#    - Install pinned numpy AND all other packages in one go
#    - CHANGED: "opencv-python-headless" -> "opencv-python" for imshow support
#    - Added boxmot for advanced tracking with ReID
# ---------------------------------------------------------------------------
RUN pip3 install --upgrade pip && \
    pip3 uninstall -y numpy || true && \
    pip3 install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python" \
    "pyyaml" \
    "supervision" \
    "tensorrt" \
    "pandas" \
    "boxmot" \
    --extra-index-url https://pypi.nvidia.com

# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------
RUN python3 - <<'PY'
import numpy, torch
print(f"NumPy Version: {numpy.__version__}")
print(f"Torch Version: {torch.__version__}")
if not numpy.__version__.startswith("1."):
    raise SystemExit("ERROR: NumPy 2.x detected. Build aborted to protect torch compatibility.")
PY

# ---------------------------------------------------------------------------
# 5. Copy Project Files
# ---------------------------------------------------------------------------
COPY . /app

# ---------------------------------------------------------------------------
# 6. Run Inference
# ---------------------------------------------------------------------------
CMD ["python3", "main_inference.py"]