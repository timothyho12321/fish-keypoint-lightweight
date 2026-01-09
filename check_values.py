import cv2
import yaml
import time
import math
import queue
import threading
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import gi

# Validates Aravis import
try:
    gi.require_version('Aravis', '0.8')
    from gi.repository import Aravis
except ValueError:
    print("Error: Aravis not found. Ensure gi and Aravis are installed.")
    exit()

# ==========================================
#      USER CALIBRATION SETTINGS
# ==========================================
NUM_HEALTHY_FISH = 3
NUM_SICK_FISH    = 1
DURATION_MINUTES = 1.0  # How long to record for calibration
SAVE_CSV_NAME    = "calibration_results.csv"

# Camera Settings
FPS_LIMIT = 30.0

# Keypoint Indices
IDX_DORSAL = 1
IDX_CENTER = 3
IDX_BOTTOM = 4
IDX_SNOUT  = 0
IDX_TAIL   = 2

# Detection Thresholds (from main_inference.py)
MAX_ALLOWED_TILT = 50.0
RATIO_OPEN_WATER = 0.35
RATIO_BOTTOM_ZONE = 0.4
BOTTOM_ZONE_LIMIT = 0.55
# ==========================================

class AravisCaptureThread:
    def __init__(self, ip_address, name="Cam"):
        self.ip = ip_address
        self.name = name
        self.stop_event = threading.Event()
        self.image_queue = queue.Queue(maxsize=1) 
        self.cam = None
        self.stream = None
        self.width = None
        self.height = None
        
    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join()

    def run(self):
        print(f"[{self.name}] Connecting to Aravis Camera at {self.ip}...")
        try:
            self.cam = Aravis.Camera.new(self.ip)
        except Exception as e:
            print(f"[{self.name}] Failed to find camera: {e}")
            return

        self.cam.set_string("AcquisitionMode", "Continuous")
        try:
            self.cam.set_float("AcquisitionFrameRate", FPS_LIMIT)
        except:
            pass 
        
        self.width = self.cam.get_integer("Width")
        self.height = self.cam.get_integer("Height")
        payload = self.cam.get_payload()

        self.stream = self.cam.create_stream(None, None)
        for _ in range(5): 
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        self.cam.start_acquisition()
        print(f"[{self.name}] Started. Res: {self.width}x{self.height}")

        while not self.stop_event.is_set():
            buffer = self.stream.timeout_pop_buffer(1000000) 
            if buffer:
                if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                    data = buffer.get_data()
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                    
                    if self.image_queue.full():
                        try:
                            self.image_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.image_queue.put(frame_bgr)
                self.stream.push_buffer(buffer)
        self.cam.stop_acquisition()

def get_fish_status(kpts_xy, frame_height):
    """
    Returns tuple: (Status_String, Color_Object)
    """
    snout  = kpts_xy[0]
    dorsal = kpts_xy[IDX_DORSAL]
    tail   = kpts_xy[2]
    center = kpts_xy[IDX_CENTER]
    bottom = kpts_xy[IDX_BOTTOM]

    # 1. MISSING POINT CHECK
    if dorsal[0] == 0 or bottom[0] == 0:
        if snout[0] != 0 and tail[0] != 0:
            return "SICK (Occluded)", sv.Color(255, 165, 0)
        return "Unknown", sv.Color(128, 128, 128)

    # 2. CALCULATE DIMENSIONS
    height = math.hypot(dorsal[0] - bottom[0], dorsal[1] - bottom[1])
    length = math.hypot(snout[0] - tail[0], snout[1] - tail[1])

    if length == 0: return "Unknown", sv.Color(128, 128, 128)

    # 3. RATIO CHECK (ZONE BASED)
    ratio = height / length
    
    # Determine Y-position (Use Center Keypoint if available, else average dorsal/bottom)
    y_pos = center[1] if center[0] != 0 else (dorsal[1] + bottom[1]) / 2
    
    # Check if fish is in the "Bottom Zone" (last 15% of screen)
    if y_pos > (frame_height * BOTTOM_ZONE_LIMIT):
        # STRICT MODE: Fish on bottom must be very upright
        limit = RATIO_BOTTOM_ZONE
        zone_label = "BTM FLAT"
    else:
        # LENIENT MODE: Fish swimming freely
        limit = RATIO_OPEN_WATER
        zone_label = "OPEN FLAT"

    if ratio < limit:
        # Returns SICK with Ratio and Zone info
        return f"SICK ({zone_label} {ratio:.2f})", sv.Color(255, 165, 0)

    # 4. TILT CHECK
    dx = dorsal[0] - bottom[0]
    dy = dorsal[1] - bottom[1] 
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    deviation = abs(angle_deg - (-90))
    if deviation > 180: deviation = 360 - deviation

    if deviation > MAX_ALLOWED_TILT:
        return f"SICK (Tilt {int(deviation)})", sv.Color(255, 165, 0)
    
    return f"HEALTHY", sv.Color.GREEN

def calculate_metrics(kpts_xy):
    """
    Returns (tilt_degrees, body_ratio, y_center)
    Returns (None, None, None) if points missing
    """
    snout  = kpts_xy[IDX_SNOUT]
    dorsal = kpts_xy[IDX_DORSAL]
    tail   = kpts_xy[IDX_TAIL]
    bottom = kpts_xy[IDX_BOTTOM]

    if dorsal[0] == 0 or bottom[0] == 0 or snout[0] == 0 or tail[0] == 0:
        return None, None, None

    # 1. Calculate Tilt
    dx = dorsal[0] - bottom[0]
    dy = dorsal[1] - bottom[1] 
    angle_deg = math.degrees(math.atan2(dy, dx))
    deviation = abs(angle_deg - (-90))
    if deviation > 180: deviation = 360 - deviation

    # 2. Calculate Ratio
    height = math.hypot(dorsal[0] - bottom[0], dorsal[1] - bottom[1])
    length = math.hypot(snout[0] - tail[0], snout[1] - tail[1])
    
    if length == 0: return None, None, None
    ratio = height / length

    # 3. Y Center (approximate depth)
    y_center = (dorsal[1] + bottom[1]) / 2

    return deviation, ratio, y_center

if __name__ == "__main__":
    # 1. Config Loading
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        exit()

    MODEL_PATH = "models/yolov8_hik_side_fish_pose_11s_250102_2026-01-02-keypoint-trial-0001.engine"
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task='pose')

    # 2. Camera Start
    side_camera_id = config.get('side_source', 'No ID Found')
    cam = AravisCaptureThread(side_camera_id)
    cam.start()

    # 3. Visualization Setup (same as main_inference.py)
    fish_edges = [(0, 1), (1, 2), (2, 4), (4, 0)] 
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=1, edges=fish_edges)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=4)

    box_annotator_healthy = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    label_annotator_healthy = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.5, text_thickness=1)

    box_annotator_sick = sv.BoxAnnotator(color=sv.Color(255, 165, 0), thickness=2)
    label_annotator_sick = sv.LabelAnnotator(color=sv.Color(255, 165, 0), text_scale=0.5, text_thickness=1)

    box_annotator_unknown = sv.BoxAnnotator(color=sv.Color(128, 128, 128), thickness=2)
    label_annotator_unknown = sv.LabelAnnotator(color=sv.Color(128, 128, 128), text_scale=0.5, text_thickness=1)

    # 4. Data Storage
    collected_data = [] 

    print("---------------------------------------------------------")
    print(f" STARTING CALIBRATION for {DURATION_MINUTES} Minutes")
    print(f" Setup: {NUM_HEALTHY_FISH} Healthy Fish, {NUM_SICK_FISH} Sick Fish")
    print(" Press 'q' to stop early.")
    print("---------------------------------------------------------")
    
    start_time = time.time()
    end_time = start_time + (DURATION_MINUTES * 60)

    try:
        while time.time() < end_time:
            if cam.image_queue.empty():
                time.sleep(0.005)
                continue
                
            image = cam.image_queue.get()

            # Inference
            results = model.predict(source=image, conf=0.45, iou=0.7, imgsz=640, verbose=False, max_det=30)
            result = results[0]

            detections = sv.Detections.from_ultralytics(result)
            key_points = sv.KeyPoints.from_ultralytics(result)
            
            healthy_count = 0
            sick_count = 0
            
            healthy_indices = []
            healthy_labels = []
            sick_indices = []
            sick_labels = []
            unknown_indices = []
            unknown_labels = []

            # Draw keypoints first
            image = edge_annotator.annotate(scene=image, key_points=key_points)
            image = vertex_annotator.annotate(scene=image, key_points=key_points)

            if len(key_points.xy) > 0:
                for i in range(len(detections)):
                    kpts = key_points.xy[i]
                    
                    # Get fish status using main_inference.py logic
                    status_text, _ = get_fish_status(kpts, cam.height)
                    
                    if "HEALTHY" in status_text:
                        healthy_count += 1
                        healthy_indices.append(i)
                        healthy_labels.append(status_text)
                        
                        # Also collect data for calibration
                        tilt, ratio, y_pos = calculate_metrics(kpts)
                        if tilt is not None:
                            collected_data.append({
                                'tilt': tilt,
                                'ratio': ratio,
                                'y_pos': y_pos
                            })
                    elif "SICK" in status_text:
                        sick_count += 1
                        sick_indices.append(i)
                        sick_labels.append(status_text)
                        
                        # Also collect data for calibration
                        tilt, ratio, y_pos = calculate_metrics(kpts)
                        if tilt is not None:
                            collected_data.append({
                                'tilt': tilt,
                                'ratio': ratio,
                                'y_pos': y_pos
                            })
                    else:
                        unknown_indices.append(i)
                        unknown_labels.append(status_text)
            
            # Annotate each category separately
            if healthy_indices:
                det_h = detections[healthy_indices]
                image = box_annotator_healthy.annotate(scene=image, detections=det_h)
                image = label_annotator_healthy.annotate(scene=image, detections=det_h, labels=healthy_labels)

            if sick_indices:
                det_s = detections[sick_indices]
                image = box_annotator_sick.annotate(scene=image, detections=det_s)
                image = label_annotator_sick.annotate(scene=image, detections=det_s, labels=sick_labels)

            if unknown_indices:
                det_u = detections[unknown_indices]
                image = box_annotator_unknown.annotate(scene=image, detections=det_u)
                image = label_annotator_unknown.annotate(scene=image, detections=det_u, labels=unknown_labels)

            # Draw Timer and Counts (after classification)
            remaining = int(end_time - time.time())
            cv2.putText(image, f"Calibration: {remaining}s | Healthy: {healthy_count} | Sick: {sick_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Calibration Mode", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    # ==========================================
    #      STATISTICAL CALCULATION
    # ==========================================
    if len(collected_data) < 10:
        print("Not enough data collected!")
        exit()

    df = pd.DataFrame(collected_data)
    print("\nProcessing Data...")

    # --- SEPARATION LOGIC ---
    total_fish = NUM_HEALTHY_FISH + NUM_SICK_FISH
    healthy_fraction = NUM_HEALTHY_FISH / total_fish
    
    # Calculate the index to split the data
    split_index = int(len(df) * healthy_fraction)
    
    # Sort data by Ratio (High Ratio = Healthy, Low Ratio = Sick)
    df_sorted = df.sort_values(by='ratio', ascending=False)
    
    healthy_group = df_sorted.iloc[:split_index]
    sick_group = df_sorted.iloc[split_index:]

    # --- 1. CALCULATE TILT LIMIT ---
    # We take the 98th percentile of the healthy group to exclude outliers
    max_healthy_tilt = healthy_group['tilt'].quantile(0.98)
    rec_max_tilt = min(max_healthy_tilt + 5.0, 60.0) 

    # --- 2. CALCULATE RATIO_OPEN_WATER ---
    # Bottom 2% of healthy group (conservative)
    min_healthy_ratio = healthy_group['ratio'].quantile(0.02)
    rec_open_water = max(min_healthy_ratio - 0.02, 0.1)

    # --- 3. CALCULATE RATIO_BOTTOM_ZONE ---
    # Median of Healthy minus 1 StdDev
    healthy_mean = healthy_group['ratio'].mean()
    healthy_std  = healthy_group['ratio'].std()
    
    if healthy_std < 0.02: healthy_std = 0.02
        
    rec_bottom_zone = healthy_mean - (1.0 * healthy_std)
    
    if rec_bottom_zone < rec_open_water:
        rec_bottom_zone = rec_open_water + 0.05

    # --- OUTPUT RESULTS ---
    print("\n" + "="*50)
    print("       CALIBRATION RESULTS       ")
    print("="*50)
    print(f"Total Frames Analyzed: {len(df)}")
    print(f"Healthy Samples: {len(healthy_group)} | Sick Samples: {len(sick_group)}")
    print("-" * 30)
    print("OBSERVED STATS (Means):")
    print(f"Healthy -> Avg Tilt: {healthy_group['tilt'].mean():.1f}°, Avg Ratio: {healthy_group['ratio'].mean():.2f}")
    if not sick_group.empty:
        print(f"Sick    -> Avg Tilt: {sick_group['tilt'].mean():.1f}°, Avg Ratio: {sick_group['ratio'].mean():.2f}")
    print("-" * 30)
    print("RECOMMENDED SETTINGS (Paste these into main_inference.py):")
    print(f"MAX_ALLOWED_TILT  = {rec_max_tilt:.1f}")
    print(f"RATIO_OPEN_WATER  = {rec_open_water:.2f}")
    print(f"RATIO_BOTTOM_ZONE = {rec_bottom_zone:.2f}")
    print("="*50)

    # Save to CSV
    output_df = pd.DataFrame([{
        "PARAMETER": "MAX_ALLOWED_TILT", "VALUE": round(rec_max_tilt, 1)
    }, {
        "PARAMETER": "RATIO_OPEN_WATER", "VALUE": round(rec_open_water, 2)
    }, {
        "PARAMETER": "RATIO_BOTTOM_ZONE", "VALUE": round(rec_bottom_zone, 2)
    }])
    
    output_df.to_csv(SAVE_CSV_NAME, index=False)
    print(f"Saved to {SAVE_CSV_NAME}")