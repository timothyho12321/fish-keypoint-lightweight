import cv2
import yaml
import time
import math
import queue
import threading
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis

# --- ObjectStatusAnalyzer Integration ---
@dataclass
class ObjectSnapshot:
    """Temporary snapshot of an object at a specific time"""
    timestamp: float
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    area: float
    width: float
    height: float
    orientation: float
    confidence: float
    is_vertical: bool = False
    fish_id: Optional[int] = None  # Track fish ID
    health_status: str = "UNKNOWN"  # HEALTHY, SICK, DEAD

class FishStatusAnalyzer:
    """Analyzer to track fish over time and differentiate sick vs dead"""
    def __init__(self, min_confidence=0.3, sick_duration=30.0, dead_duration=300.0):
        """
        Args:
            min_confidence: Minimum detection confidence threshold
            sick_duration: Seconds a fish must be unhealthy to be classified as sick (default 30s)
            dead_duration: Seconds a fish must be unhealthy with no movement to be dead (default 5 minutes)
        """
        self.min_confidence = min_confidence
        self.SICK_DURATION = sick_duration  # User configurable
        self.DEAD_DURATION = dead_duration  # User configurable
        
        # Update cycle (seconds)
        self.UPDATE_INTERVAL = 15.0  # Update stable values every 15 seconds
        self.ANALYSIS_WINDOW = 30.0  # Analyze last 30 seconds of data
        
        # Spatial clustering
        self.CLUSTER_DISTANCE = 50.0  # Pixels - fish closer than this are considered same fish
        
        # Track individual fish history
        self.fish_history = {}  # fish_id -> deque of snapshots
        self.fish_unhealthy_start = {}  # fish_id -> timestamp when became unhealthy
        self.fish_last_movement = {}  # fish_id -> timestamp of last significant movement
        
        # Data storage
        self.frame_buffer = deque(maxlen=900)  # 30 seconds at 30fps
        self.frame_count = 0
        
        # Stable values
        self.stable_values = {
            'total': 0,
            'active': 0,
            'sick': 0,
            'dead': 0,
            'last_update': 0,
            'next_update': 0
        }
        
        # Current window data
        self.current_window_data = {
            'start_time': time.time(),
            'frame_count': 0,
            'snapshots': []
        }
        
        self.start_time = time.time()
    
    def validate_detection(self, detection: Dict) -> bool:
        """Validate detection"""
        confidence = detection.get('confidence', 0)
        if confidence < self.min_confidence:
            return False
        
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True
    
    def create_snapshot(self, detection: Dict) -> ObjectSnapshot:
        """Create snapshot from detection"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        orientation = 0 if width >= height else 90
        is_vertical = orientation > 45
        
        return ObjectSnapshot(
            timestamp=time.time(),
            center=(center_x, center_y),
            bbox=bbox,
            area=area,
            width=width,
            height=height,
            orientation=orientation,
            confidence=detection.get('confidence', 0.3),
            is_vertical=is_vertical,
            fish_id=detection.get('fish_id'),
            health_status=detection.get('health_status', 'UNKNOWN')
        )
    
    def update_fish_history(self, snapshot: ObjectSnapshot):
        """Update individual fish tracking history"""
        if snapshot.fish_id is None:
            return
        
        fish_id = snapshot.fish_id
        current_time = snapshot.timestamp
        
        # Initialize history if new fish
        if fish_id not in self.fish_history:
            self.fish_history[fish_id] = deque(maxlen=900)  # 30s history
            self.fish_unhealthy_start[fish_id] = None
            self.fish_last_movement[fish_id] = current_time
        
        # Add snapshot to history
        self.fish_history[fish_id].append(snapshot)
        
        # Track unhealthy duration
        if snapshot.health_status in ['SICK', 'DEAD']:
            if self.fish_unhealthy_start[fish_id] is None:
                self.fish_unhealthy_start[fish_id] = current_time
        else:
            # Reset if healthy
            self.fish_unhealthy_start[fish_id] = None
            self.fish_last_movement[fish_id] = current_time
        
        # Check for movement (compare with recent history)
        if len(self.fish_history[fish_id]) >= 2:
            recent_snapshots = list(self.fish_history[fish_id])[-30:]  # Last 1 second at 30fps
            if len(recent_snapshots) >= 2:
                # Calculate movement
                positions = [s.center for s in recent_snapshots]
                max_distance = 0
                for i in range(len(positions) - 1):
                    dist = math.sqrt(
                        (positions[i+1][0] - positions[i][0])**2 +
                        (positions[i+1][1] - positions[i][1])**2
                    )
                    max_distance = max(max_distance, dist)
                
                # If significant movement detected (>5 pixels in 1 second)
                if max_distance > 5.0:
                    self.fish_last_movement[fish_id] = current_time
    
    def classify_fish_status(self, fish_id: int) -> str:
        """Classify fish as HEALTHY, SICK, or DEAD based on history"""
        if fish_id not in self.fish_history or not self.fish_history[fish_id]:
            return "UNKNOWN"
        
        current_time = time.time()
        recent_snapshot = self.fish_history[fish_id][-1]
        
        # If currently detected as healthy, return healthy
        if recent_snapshot.health_status == "HEALTHY":
            return "HEALTHY"
        
        # Check how long the fish has been unhealthy
        unhealthy_start = self.fish_unhealthy_start.get(fish_id)
        if unhealthy_start is None:
            return "HEALTHY"
        
        unhealthy_duration = current_time - unhealthy_start
        
        # Check last movement time
        last_movement = self.fish_last_movement.get(fish_id, current_time)
        time_since_movement = current_time - last_movement
        
        # DEAD: Unhealthy for long duration AND no movement
        if unhealthy_duration >= self.DEAD_DURATION and time_since_movement >= self.DEAD_DURATION:
            return "DEAD"
        
        # SICK: Unhealthy for shorter duration OR has some movement
        if unhealthy_duration >= self.SICK_DURATION:
            return "SICK"
        
        # Still in grace period
        return "HEALTHY"
    
    def process_frame(self, current_detections: List[Dict]) -> Dict:
        """
        Process a frame - called EVERY frame
        
        Args:
            current_detections: List of detection dictionaries with bbox, confidence, fish_id, health_status
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Create snapshots
        snapshots = []
        for det in current_detections:
            if self.validate_detection(det):
                snapshot = self.create_snapshot(det)
                snapshots.append(snapshot)
                self.update_fish_history(snapshot)
        
        # Classify each fish based on history
        active_count = 0
        sick_count = 0
        dead_count = 0
        
        for snapshot in snapshots:
            if snapshot.fish_id is not None:
                status = self.classify_fish_status(snapshot.fish_id)
                if status == "HEALTHY":
                    active_count += 1
                elif status == "SICK":
                    sick_count += 1
                elif status == "DEAD":
                    dead_count += 1
            else:
                # No ID, use current frame status
                if snapshot.health_status == "HEALTHY":
                    active_count += 1
                else:
                    sick_count += 1
        
        # Store frame data
        frame_data = {
            'frame': self.frame_count,
            'timestamp': current_time,
            'snapshots': snapshots,
            'status': {
                'active': active_count,
                'sick': sick_count,
                'dead': dead_count,
                'total': active_count + sick_count + dead_count
            }
        }
        
        self.frame_buffer.append(frame_data)
        self.current_window_data['snapshots'].extend(snapshots)
        self.current_window_data['frame_count'] += 1
        
        # Check if 15 seconds have passed
        time_in_window = current_time - self.current_window_data['start_time']
        
        if time_in_window >= self.UPDATE_INTERVAL:
            self.update_stable_values()
            
            # Reset window
            self.current_window_data = {
                'start_time': current_time,
                'frame_count': 0,
                'snapshots': []
            }
        
        return frame_data
    
    def update_stable_values(self):
        """Update stable values using recent data"""
        current_time = time.time()
        
        # Get data from last 30 seconds
        window_start = current_time - self.ANALYSIS_WINDOW
        recent_frames = [f for f in self.frame_buffer if f['timestamp'] >= window_start]
        
        if not recent_frames:
            return
        
        # Calculate median counts (robust to outliers)
        active_counts = [f['status']['active'] for f in recent_frames]
        sick_counts = [f['status']['sick'] for f in recent_frames]
        dead_counts = [f['status']['dead'] for f in recent_frames]
        total_counts = [f['status']['total'] for f in recent_frames]
        
        # Use median for stability
        stable_active = int(np.median(active_counts)) if active_counts else 0
        stable_sick = int(np.median(sick_counts)) if sick_counts else 0
        stable_dead = int(np.median(dead_counts)) if dead_counts else 0
        stable_total = int(np.median(total_counts)) if total_counts else 0
        
        # Ensure consistency
        if stable_active + stable_sick + stable_dead != stable_total:
            stable_total = stable_active + stable_sick + stable_dead
        
        # Smooth with previous values (60% previous, 40% new)
        smoothing = 0.6
        if self.stable_values['total'] > 0:
            stable_active = int(self.stable_values['active'] * smoothing + stable_active * (1 - smoothing))
            stable_sick = int(self.stable_values['sick'] * smoothing + stable_sick * (1 - smoothing))
            stable_dead = int(self.stable_values['dead'] * smoothing + stable_dead * (1 - smoothing))
            stable_total = stable_active + stable_sick + stable_dead
        
        # Update stable values
        self.stable_values['total'] = max(0, stable_total)
        self.stable_values['active'] = max(0, stable_active)
        self.stable_values['sick'] = max(0, stable_sick)
        self.stable_values['dead'] = max(0, stable_dead)
        self.stable_values['last_update'] = current_time
        self.stable_values['next_update'] = current_time + self.UPDATE_INTERVAL
        
        print(f"[{time.strftime('%H:%M:%S')}] Stable values updated: "
              f"Total={stable_total}, Active={stable_active}, Sick={stable_sick}, Dead={stable_dead}")
    
    def get_stable_counts(self) -> Dict:
        """Get current stable counts"""
        return {
            'total': self.stable_values['total'],
            'active': self.stable_values['active'],
            'sick': self.stable_values['sick'],
            'dead': self.stable_values['dead']
        }

# --- 1. USER CONFIGURATION ---
MAX_ALLOWED_TILT = 60.0#50.0
CONFIDENCE_THRESHOLD = 0.45 
IOU_THRESHOLD = 0.7

# RATIO THRESHOLDS
RATIO_OPEN_WATER = 0.35  # Lenient for swimming fish
RATIO_BOTTOM_ZONE = 0.4 # Strict for fish on floor
BOTTOM_ZONE_LIMIT = 0.45 # Bottom 15% of screen is "Danger Zone"

# VISUAL DEBUGGING
SHOW_BOTTOM_ZONE_LINE = True  # Set to False to hide the line

# TRACKING CONFIGURATION
MAX_FISH_COUNT = 20  # Maximum number of fish to track

# Keypoint Definitions (Tiger Barbs)
KEYPOINT_NAMES = ["S", "D", "T", "C", "B"] 
IDX_DORSAL = 1  
IDX_CENTER = 3   
IDX_BOTTOM = 4   

# --- 2. CAMERA CLASS ---
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

# --- 3. HELPER FUNCTION: GET STATUS (ZONE AWARE) ---
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
        limit = RATIO_BOTTOM_ZONE # 0.50
        zone_label = "BTM FLAT"
    else:
        # LENIENT MODE: Fish swimming freely
        limit = RATIO_OPEN_WATER # 0.35
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

    # Initialize Fish Status Analyzer
    fish_analyzer = FishStatusAnalyzer(
        min_confidence=CONFIDENCE_THRESHOLD,
        sick_duration=30.0,  # 30 seconds to classify as sick
        dead_duration=300.0  # 5 minutes to classify as dead
    )

    # ID Management: Map tracker IDs to fixed fish IDs (1-20)
    tracker_to_fish_id = {}  # Maps tracker_id -> fish_id (1-20)
    fish_id_pool = set(range(1, MAX_FISH_COUNT + 1))  # Available IDs
    active_fish_ids = {}  # fish_id -> last_seen_frame
    frame_counter = 0
    ID_TIMEOUT = 90  # Frames before ID is released back to pool

    try:
        while True:
            if cam.image_queue.empty():
                time.sleep(0.005) 
                continue
                
            image = cam.image_queue.get()

            # Original predict method (commented out)
            # results = model.predict(
            #     source=image,
            #     conf=CONFIDENCE_THRESHOLD,
            #     iou=IOU_THRESHOLD,
            #     imgsz=640,    
            #     verbose=False,
            #     max_det=30 
            # )
            
            # Using tracking with BoTSORT
            results = model.track(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=640,
                verbose=False,
                # max_det=20,
                # tracker="botsort.yaml",
                tracker="bytetrack.yaml",
                persist=True
            )
            
            result = results[0] 

            detections = sv.Detections.from_ultralytics(result)
            key_points = sv.KeyPoints.from_ultralytics(result)
            
            frame_counter += 1
            
            # Release IDs that haven't been seen for ID_TIMEOUT frames
            ids_to_release = [fish_id for fish_id, last_frame in active_fish_ids.items() 
                            if frame_counter - last_frame > ID_TIMEOUT]
            for fish_id in ids_to_release:
                fish_id_pool.add(fish_id)
                del active_fish_ids[fish_id]
                # Remove from tracker mapping
                tracker_to_fish_id = {k: v for k, v in tracker_to_fish_id.items() if v != fish_id}
            
            healthy_count = 0
            sick_count = 0
            
            healthy_indices = []
            healthy_labels = []
            sick_indices = []
            sick_labels = []
            unknown_indices = []
            unknown_labels = []
            
            # Prepare detections for analyzer
            detections_for_analyzer = []

            image = edge_annotator.annotate(scene=image, key_points=key_points)
            image = vertex_annotator.annotate(scene=image, key_points=key_points)

            if len(key_points.xy) > 0:
                for i in range(len(detections)):
                    kpts = key_points.xy[i] 
                    
                    # Get tracker ID from detections
                    tracker_id = None
                    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                        if i < len(detections.tracker_id):
                            tracker_id = int(detections.tracker_id[i])
                    
                    # Assign fish ID (1-20)
                    fish_id = None
                    if tracker_id is not None:
                        if tracker_id in tracker_to_fish_id:
                            # Existing mapping
                            fish_id = tracker_to_fish_id[tracker_id]
                        elif fish_id_pool:
                            # Assign new ID from pool
                            fish_id = min(fish_id_pool)
                            fish_id_pool.remove(fish_id)
                            tracker_to_fish_id[tracker_id] = fish_id
                    elif fish_id_pool:
                        # No tracker ID, assign from pool
                        fish_id = min(fish_id_pool)
                        fish_id_pool.remove(fish_id)
                        if tracker_id is not None:
                            tracker_to_fish_id[tracker_id] = fish_id
                    
                    # Update last seen
                    if fish_id is not None:
                        active_fish_ids[fish_id] = frame_counter
                    
                    # --- PASS FRAME HEIGHT TO FUNCTION ---
                    status_text, _ = get_fish_status(kpts, cam.height)
                    
                    # Add ID to status text
                    if fish_id is not None:
                        status_text = f"ID{fish_id} {status_text}"
                    
                    # Prepare detection data for analyzer
                    bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                    det_dict = {
                        'bbox': bbox.tolist(),
                        'confidence': float(detections.confidence[i]) if detections.confidence is not None else 0.5,
                        'fish_id': fish_id,
                        'health_status': 'HEALTHY' if 'HEALTHY' in status_text else 'SICK'
                    }
                    detections_for_analyzer.append(det_dict)
                    
                    if "HEALTHY" in status_text:
                        healthy_count += 1
                        healthy_indices.append(i)
                        healthy_labels.append(status_text)
                    elif "SICK" in status_text:
                        sick_count += 1
                        sick_indices.append(i)
                        sick_labels.append(status_text)
                    else:
                        unknown_indices.append(i)
                        unknown_labels.append(status_text)
            
            # Process frame with analyzer
            fish_analyzer.process_frame(detections_for_analyzer)
            
            # Get stable counts
            stable_counts = fish_analyzer.get_stable_counts()
            
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

            # --- DRAW ZONE LINE ---
            if SHOW_BOTTOM_ZONE_LINE and cam.height is not None:
                # Calculate Y-coordinate for the line
                line_y = int(cam.height * BOTTOM_ZONE_LIMIT)
                
                # Draw Line (Light Grey)
                # BGR Color: (211, 211, 211)
                cv2.line(image, (0, line_y), (cam.width, line_y), (211, 211, 211), 2)
                
                # Optional: Add small text label for the line
                cv2.putText(image, "BOTTOM ZONE", (10, line_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (211, 211, 211), 1)

            # --- DISPLAY CURRENT FRAME COUNTS (LEFT SIDE) ---
            cv2.putText(image, f"Current Frame:", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Healthy: {healthy_count} | Sick: {sick_count}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # --- DISPLAY STABLE COUNTS (RIGHT SIDE) ---
            # Position on right side of screen
            right_x = cam.width - 350 if cam.width else 950
            
            cv2.putText(image, f"Stable Counts (15s):", (right_x, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Total: {stable_counts['total']}", (right_x, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Active: {stable_counts['active']}", (right_x, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Sick: {stable_counts['sick']}", (right_x, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(image, f"Dead: {stable_counts['dead']}", (right_x, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
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