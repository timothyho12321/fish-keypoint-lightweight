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
CONFIDENCE_THRESHOLD = 0.45 
IOU_THRESHOLD = 0.7

# Keypoint Definitions (Tiger Barbs)
KEYPOINT_NAMES = ["S", "D", "T", "C", "B"]
IDX_DORSAL = 1  
IDX_BOTTOM = 4   # <--- RENAMED from IDX_BELLY

# ... (Camera Class omitted for brevity as it is unchanged) ...
class AravisCaptureThread:
    # ... (Same as previous code) ...
    def __init__(self, ip_address, name="Cam"):
        self.ip = ip_address
        self.name = name
        self.stop_event = threading.Event()
        self.image_queue = queue.Queue(maxsize=1) 
        self.cam = None
        self.stream = None
        self.width = None
        self.height = None
        self.fps_limit = 30.0 
        
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
            self.cam.set_float("AcquisitionFrameRate", self.fps_limit)
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

        self.start_time = time.time()
        self.frame_count = 0

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
    if len(kpts_xy) <= IDX_BOTTOM: return None # <--- UPDATED CHECK

    dorsal = kpts_xy[IDX_DORSAL]
    bottom = kpts_xy[IDX_BOTTOM] # <--- RENAMED from belly

    # Check for (0,0) which indicates non-visible keypoint
    if dorsal[0] == 0 or bottom[0] == 0: return None

    # Calculate difference
    dx = dorsal[0] - bottom[0]
    dy = dorsal[1] - bottom[1] 
    
    # Math remains the same: Calculate angle of the line connecting them
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    # Calculate deviation from vertical (-90 degrees)
    deviation = abs(angle_deg - (-90))
    if deviation > 180: deviation = 360 - deviation
    return deviation

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        exit()

    MODEL_PATH = "models/yolov8_hik_side_fish_pose_11s_250102_2026-01-02-keypoint-trial-0001.engine"

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task='pose')

    side_camera_id = config.get('side_source', 'No ID Found')
    cam = AravisCaptureThread(side_camera_id)
    cam.start()

    # Visualization Annotators
    fish_edges = [(0, 1), (1, 2), (2, 4), (4, 0)] 
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=1, edges=fish_edges)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=4)

    box_annotator_healthy = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    label_annotator_healthy = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.5, text_thickness=1)

    box_annotator_sick = sv.BoxAnnotator(color=sv.Color(255, 165, 0), thickness=2)
    label_annotator_sick = sv.LabelAnnotator(color=sv.Color(255, 165, 0), text_scale=0.5, text_thickness=1)

    box_annotator_unknown = sv.BoxAnnotator(color=sv.Color(128, 128, 128), thickness=2)
    label_annotator_unknown = sv.LabelAnnotator(color=sv.Color(128, 128, 128), text_scale=0.5, text_thickness=1)

    print("Starting native inference loop. Press 'q' to exit.")
    
    inference_count = 0
    inference_start_time = time.time()

    try:
        while True:
            if cam.image_queue.empty():
                time.sleep(0.005) 
                continue
                
            image = cam.image_queue.get()

            results = model.predict(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=640,    
                verbose=False,
                max_det=30 
            )
            
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

            image = edge_annotator.annotate(scene=image, key_points=key_points)
            image = vertex_annotator.annotate(scene=image, key_points=key_points)

            if len(key_points.xy) > 0:
                for i in range(len(detections)):
                    kpts = key_points.xy[i] 
                    
                    tilt = get_fish_tilt(kpts)
                    
                    if tilt is None:
                        unknown_indices.append(i)
                        unknown_labels.append("Unknown")
                        continue

                    if tilt <= MAX_ALLOWED_TILT:
                        healthy_count += 1
                        healthy_indices.append(i)
                        healthy_labels.append(f"HEALTHY {int(tilt)}deg")
                    else:
                        sick_count += 1
                        sick_indices.append(i)
                        sick_labels.append(f"SICK {int(tilt)}deg")
            
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

            cv2.putText(image, f"Healthy: {healthy_count} | Sick: {sick_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Fish Monitor (Native)", image)

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