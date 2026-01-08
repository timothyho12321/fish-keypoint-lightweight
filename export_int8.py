from ultralytics import YOLO

# 1. Load your custom trained model
model = YOLO("models/yolov8_hik_side_fish_pose_11s_250102_2026-01-02-keypoint-trial-0001.pt")

# 2. Export to TensorRT with INT8 Quantization
# 'data': Points to your dataset yaml so it can find images for calibration
# 'workspace': GPU memory in GB to allocate for the build process (4 is usually safe for Orin)
model.export(
    format="engine",
    device=0,
    int8=True,
    data="data/2026-01-02-keypoint-trial-0001.v4i.yolov8/data.yaml",  # <--- CRITICAL: Must point to your data.yaml
    batch=1,       # Optimized for single-stream inference
    workspace=4    # 4GB workspace for building engine
)