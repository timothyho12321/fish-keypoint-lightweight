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

    # 4. Data Storage (separate for healthy and sick)
    healthy_data = []
    sick_data = [] 

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
                        
                        # Collect data for healthy fish
                        tilt, ratio, y_pos = calculate_metrics(kpts)
                        if tilt is not None:
                            healthy_data.append({
                                'tilt': tilt,
                                'ratio': ratio,
                                'y_pos': y_pos
                            })
                    elif "SICK" in status_text:
                        sick_count += 1
                        sick_indices.append(i)
                        sick_labels.append(status_text)
                        
                        # Collect data for sick fish
                        tilt, ratio, y_pos = calculate_metrics(kpts)
                        if tilt is not None:
                            sick_data.append({
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
    if len(healthy_data) < 5 and len(sick_data) < 5:
        print("Not enough data collected for both healthy and sick fish!")
        exit()

    print("\nProcessing Data...")
    print("="*60)
    print("       CALIBRATION RESULTS (Based on Current Thresholds)")
    print("="*60)
    print(f"Current Settings Used for Classification:")
    print(f"  MAX_ALLOWED_TILT  = {MAX_ALLOWED_TILT}")
    print(f"  RATIO_OPEN_WATER  = {RATIO_OPEN_WATER}")
    print(f"  RATIO_BOTTOM_ZONE = {RATIO_BOTTOM_ZONE}")
    print(f"  BOTTOM_ZONE_LIMIT = {BOTTOM_ZONE_LIMIT}")
    print("-" * 60)

    # Convert to DataFrames
    df_healthy = pd.DataFrame(healthy_data) if len(healthy_data) > 0 else pd.DataFrame()
    df_sick = pd.DataFrame(sick_data) if len(sick_data) > 0 else pd.DataFrame()

    print(f"Healthy Samples Collected: {len(df_healthy)}")
    print(f"Sick Samples Collected:    {len(df_sick)}")
    print("-" * 60)

    # --- HEALTHY FISH STATISTICS ---
    if not df_healthy.empty:
        print("\nHEALTHY FISH BEHAVIOR:")
        avg_tilt_healthy = df_healthy['tilt'].mean()
        std_tilt_healthy = df_healthy['tilt'].std()
        avg_ratio_healthy = df_healthy['ratio'].mean()
        std_ratio_healthy = df_healthy['ratio'].std()
        
        print(f"  Avg Tilt:  {avg_tilt_healthy:.1f}° (±{std_tilt_healthy:.1f}°)")
        print(f"  Max Tilt:  {df_healthy['tilt'].max():.1f}°")
        print(f"  Avg Ratio: {avg_ratio_healthy:.3f} (±{std_ratio_healthy:.3f})")
        print(f"  Min Ratio: {df_healthy['ratio'].min():.3f}")
    else:
        print("\nHEALTHY FISH BEHAVIOR: No data collected")
        avg_tilt_healthy = None
        avg_ratio_healthy = None

    # --- SICK FISH STATISTICS ---
    if not df_sick.empty:
        print("\nSICK FISH BEHAVIOR:")
        avg_tilt_sick = df_sick['tilt'].mean()
        std_tilt_sick = df_sick['tilt'].std()
        avg_ratio_sick = df_sick['ratio'].mean()
        std_ratio_sick = df_sick['ratio'].std()
        
        print(f"  Avg Tilt:  {avg_tilt_sick:.1f}° (±{std_tilt_sick:.1f}°)")
        print(f"  Max Tilt:  {df_sick['tilt'].max():.1f}°")
        print(f"  Avg Ratio: {avg_ratio_sick:.3f} (±{std_ratio_sick:.3f})")
        print(f"  Max Ratio: {df_sick['ratio'].max():.3f}")
    else:
        print("\nSICK FISH BEHAVIOR: No data collected")
        avg_tilt_sick = None
        avg_ratio_sick = None

    # --- CALCULATE RECOMMENDED THRESHOLDS ---
    print("\n" + "="*60)
    print("       RECOMMENDED FINE-TUNED THRESHOLDS")
    print("="*60)

    # 1. MAX_ALLOWED_TILT: Set slightly above healthy average
    if avg_tilt_healthy is not None:
        rec_max_tilt = min(avg_tilt_healthy + 10.0, 70.0)  # Add 10° buffer
    else:
        rec_max_tilt = MAX_ALLOWED_TILT
    
    # 2. RATIO_OPEN_WATER: Set slightly below healthy minimum
    if avg_ratio_healthy is not None:
        min_healthy_ratio = df_healthy['ratio'].quantile(0.05)  # Bottom 5% of healthy
        rec_open_water = max(min_healthy_ratio - 0.03, 0.20)
    else:
        rec_open_water = RATIO_OPEN_WATER

    # 3. RATIO_BOTTOM_ZONE: Set more strictly (1 std dev below healthy mean)
    if avg_ratio_healthy is not None:
        rec_bottom_zone = max(avg_ratio_healthy - std_ratio_healthy, rec_open_water + 0.05)
    else:
        rec_bottom_zone = RATIO_BOTTOM_ZONE

    print(f"MAX_ALLOWED_TILT  = {rec_max_tilt:.1f}  (Healthy avg: {avg_tilt_healthy:.1f}°)" if avg_tilt_healthy else f"MAX_ALLOWED_TILT  = {rec_max_tilt:.1f}  (No change)")
    print(f"RATIO_OPEN_WATER  = {rec_open_water:.2f}  (Healthy min: {df_healthy['ratio'].min():.3f})" if not df_healthy.empty else f"RATIO_OPEN_WATER  = {rec_open_water:.2f}  (No change)")
    print(f"RATIO_BOTTOM_ZONE = {rec_bottom_zone:.2f}  (Healthy avg: {avg_ratio_healthy:.3f})" if avg_ratio_healthy else f"RATIO_BOTTOM_ZONE = {rec_bottom_zone:.2f}  (No change)")
    print("="*60)

    # Save to CSV
    output_df = pd.DataFrame([{
        "PARAMETER": "MAX_ALLOWED_TILT", 
        "VALUE": round(rec_max_tilt, 1),
        "HEALTHY_AVG": round(avg_tilt_healthy, 1) if avg_tilt_healthy else "N/A",
        "SICK_AVG": round(avg_tilt_sick, 1) if avg_tilt_sick else "N/A"
    }, {
        "PARAMETER": "RATIO_OPEN_WATER", 
        "VALUE": round(rec_open_water, 3),
        "HEALTHY_AVG": round(avg_ratio_healthy, 3) if avg_ratio_healthy else "N/A",
        "SICK_AVG": round(avg_ratio_sick, 3) if avg_ratio_sick else "N/A"
    }, {
        "PARAMETER": "RATIO_BOTTOM_ZONE", 
        "VALUE": round(rec_bottom_zone, 3),
        "HEALTHY_AVG": round(avg_ratio_healthy, 3) if avg_ratio_healthy else "N/A",
        "SICK_AVG": round(avg_ratio_sick, 3) if avg_ratio_sick else "N/A"
    }])
    
    output_df.to_csv(SAVE_CSV_NAME, index=False)
    print(f"Saved to {SAVE_CSV_NAME}")