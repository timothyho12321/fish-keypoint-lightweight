import cv2
import yaml
import time
import math
import queue
import threading
import numpy as np
import supervision as sv
from ultralytics import YOLO
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis

# --- 1. USER CONFIGURATION ---
MAX_ALLOWED_TILT = 50.0 
# Note: Ultralytics handles conf/iou inside the predict method
CONFIDENCE_THRESHOLD = 0.45 
IOU_THRESHOLD = 0.7

# Keypoint Definitions (Tiger Barbs)
KEYPOINT_NAMES = ["S", "D", "T", "C", "B"]
IDX_DORSAL = 1  
IDX_BELLY  = 4  

# --- 2. CAMERA CLASS (Optimized for Threading) ---
class AravisCaptureThread:
    def __init__(self, ip_address, name="Cam"):
        self.ip = ip_address
        self.name = name
        self.stop_event = threading.Event()
        # Use a small queue to ensure we always process the latest frame (drop old ones)
        self.image_queue = queue.Queue(maxsize=1) 
        self.cam = None
        self.stream = None
        
        # Settings
        self.width = None
        self.height = None
        self.fps_limit = 30.0 # Match your camera capabilities
        
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

        # Basic Setup
        self.cam.set_string("AcquisitionMode", "Continuous")
        try:
            self.cam.set_float("AcquisitionFrameRate", self.fps_limit)
        except:
            pass # Some cameras don't support setting FPS directly this way
        
        # Auto-detect resolution
        self.width = self.cam.get_integer("Width")
        self.height = self.cam.get_integer("Height")
        payload = self.cam.get_payload()

        # Stream Setup
        self.stream = self.cam.create_stream(None, None)
        # Push enough buffers to handle jitter, but not too many to increase latency
        for _ in range(5): 
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        self.cam.start_acquisition()
        print(f"[{self.name}] Started. Res: {self.width}x{self.height}")

        self.start_time = time.time()
        self.frame_count = 0

        while not self.stop_event.is_set():
            buffer = self.stream.timeout_pop_buffer(1000000) # 1 sec timeout
            if buffer:
                if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                    data = buffer.get_data()
                    # Efficient reshaping
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width)
                    
                    # Convert to BGR (CPU bound, but unavoidable without GStreamer)
                    # Use BAYER_RG2BGR or BG2BGR depending on your specific sensor pattern
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                    
                    # Non-blocking put: if queue is full, remove old item and put new one
                    if self.image_queue.full():
                        try:
                            self.image_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.image_queue.put(frame_bgr)

                    # Monitor Camera FPS
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed >= 30.0:
                        fps = self.frame_count / elapsed
                        print(f"[{self.name}] Camera FPS: {fps:.2f}")
                        self.frame_count = 0
                        self.start_time = time.time()
                
                self.stream.push_buffer(buffer)

        self.cam.stop_acquisition()

# --- 3. HELPER FUNCTION: TILT CALCULATION ---
def get_fish_tilt(kpts_xy):
    # kpts_xy is typically shape (N, 2) where N is num keypoints
    if len(kpts_xy) <= IDX_BELLY: return None

    dorsal = kpts_xy[IDX_DORSAL]
    belly  = kpts_xy[IDX_BELLY]

    # Check for (0,0) which indicates non-visible keypoint
    if dorsal[0] == 0 or belly[0] == 0: return None

    dx = dorsal[0] - belly[0]
    dy = dorsal[1] - belly[1] 
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    # Calculate deviation from vertical (-90 degrees)
    deviation = abs(angle_deg - (-90))
    if deviation > 180: deviation = 360 - deviation
    return deviation

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load Config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        exit()

    # --- MODEL LOADING STRATEGY ---
    # 1. Update this path to where your .pt file actually is!
    # Roboflow API strings won't work here. You need the physical file.
    # MODEL_PATH = "best.pt"
    # MODEL_PATH = "models/yolov8_hik_side_fish_pose_11s_250102_2026-01-02-keypoint-trial-0001.pt"
    MODEL_PATH = "models/yolov8_hik_side_fish_pose_11s_250102_2026-01-02-keypoint-trial-0001.engine"

    print(f"Loading model: {MODEL_PATH}")
    # model = YOLO(MODEL_PATH)
    model = YOLO(MODEL_PATH, task='pose')

    # CHECK FOR TENSORRT EXPORT (Huge Speedup)
    # If you haven't exported yet, the code below runs .pt (PyTorch)
    # If you provide a .engine file, Ultralytics uses TensorRT automatically.

    # B. Start Camera
    side_camera_id = config.get('side_source', 'No ID Found')
    cam = AravisCaptureThread(side_camera_id)
    cam.start()

    # C. Visualization Annotators
    # Define edges for Tiger Barb: Snout-Dorsal, Dorsal-Tail, Tail-Belly, Belly-Snout (Example)
    fish_edges = [(0, 1), (1, 2), (2, 4), (4, 0)] 
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=1, edges=fish_edges)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=4)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    print("Starting native inference loop. Press 'q' to exit.")
    
    inference_count = 0
    inference_start_time = time.time()

    try:
        while True:
            if cam.image_queue.empty():
                time.sleep(0.005) # Brief sleep to yield CPU
                continue
                
            image = cam.image_queue.get()

            # --- NATIVE INFERENCE ---
            # 1. verbose=False: Stops printing to console (saves time)
            # 2. half=True: Uses FP16 (Massive speedup on Jetson)
            # 3. device=0: Ensures GPU usage
            # 4. stream=True: generator memory efficiency
            # results = model.predict(
            #     source=image,
            #     conf=CONFIDENCE_THRESHOLD,
            #     iou=IOU_THRESHOLD,
            #     half=True,
            #     # device=0,
            #     verbose=False,
            #     max_det=30 # Limit max detections if you only expect 20 fish
            # )
            results = model.predict(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=640,    # <-- ADDED (Critical: Engine expects fixed 640x640 input)
                # device=0,
                verbose=False,
                max_det=30 # Limit max detections if you only expect 20 fish
            )
            
            result = results[0] # We are only processing 1 frame

            # --- SUPERVISION CONVERSION ---
            # Native Ultralytics output -> Supervision Format
            detections = sv.Detections.from_ultralytics(result)
            key_points = sv.KeyPoints.from_ultralytics(result)

            # --- ANNOTATION & LOGIC ---
            # Draw boxes first
            # We need to manually filter or categorize boxes based on tilt logic
            
            healthy_count = 0
            sick_count = 0
            
            # Prepare labels
            labels = []

            # Annotate Keypoints
            image = edge_annotator.annotate(scene=image, key_points=key_points)
            image = vertex_annotator.annotate(scene=image, key_points=key_points)

            if len(key_points.xy) > 0:
                for i in range(len(detections)):
                    # Get Keypoints for this specific detection
                    # supervision keypoints are (N, K, 2)
                    kpts = key_points.xy[i] 
                    
                    tilt = get_fish_tilt(kpts)
                    
                    if tilt is None:
                        labels.append("Unknown")
                        continue

                    if tilt <= MAX_ALLOWED_TILT:
                        status = "HEALTHY"
                        healthy_count += 1
                        # Hacky color change for box: Supervision BoxAnnotator doesn't support 
                        # per-box dynamic coloring easily without creating separate Detection objects.
                        # For speed/simplicity, we label clearly.
                    else:
                        status = "SICK"
                        sick_count += 1

                    labels.append(f"{status} {int(tilt)}deg")
            
            # Draw Boxes with Labels
            # 1. Draw Boxes (No labels argument here anymore)
            image = box_annotator.annotate(scene=image, detections=detections)

            # 2. Draw Labels (Use the new annotator here)
            image = label_annotator.annotate(scene=image, detections=detections, labels=labels)
            # --- STATS DISPLAY ---
            cv2.putText(image, f"Healthy: {healthy_count} | Sick: {sick_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Fish Monitor (Native)", image)

            # Monitor Inference FPS
            inference_count += 1
            elapsed = time.time() - inference_start_time
            if elapsed >= 30.0:
                fps = inference_count / elapsed
                print(f"[Main] Inference FPS: {fps:.2f}")
                inference_count = 0
                inference_start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping camera...")
        cam.stop()
        cv2.destroyAllWindows()