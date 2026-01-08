import sys
import time
import os
import gi
import cv2
import numpy as np
import yaml
from datetime import datetime

# Ensure Aravis is available
try:
    gi.require_version('Aravis', '0.8')
    from gi.repository import Aravis
except ValueError:
    print("Error: Aravis-0.8 not found. Please install gir1.2-aravis-0.8")
    sys.exit(1)

# ==========================================
#  USER CONFIGURATION
# ==========================================
CONFIG_PATH = 'config.yaml'
SAVE_FOLDER = "." 

RECORD_TIME_DEFAULT = 10  # in minutes
# ==========================================
#  HELPER FUNCTIONS
# ==========================================
def get_gst_pipeline(filename, width, height, fps):
    # Hardware accelerated encoding pipeline for Jetson Orin
    # Use h264 encoding
    return (
        f"appsrc ! video/x-raw, format=BGR ! "
        f"queue ! videoconvert ! video/x-raw,format=BGRx ! "
        f"nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc bitrate=8000000 ! h264parse ! mp4mux ! "
        f"filesink location={filename}"
    )

def main():
    # 1. Load Config
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters matching main_inference.py logic
    # Assuming the config has 'side_source' at the top level or within a 'video' block.
    # We prioritize the structure found in main_inference.py context (top level).
    camera_id = config.get('side_source')
    if not camera_id:
        # Fallback to check 'video' dict if it exists (legacy config support)
        if 'video' in config:
            camera_id = config['video'].get('side_source')
    
    if not camera_id:
        print("[ERROR] 'side_source' not found in config.yaml")
        return

    # Recording Duration
    record_minutes = config.get('record_minutes', RECORD_TIME_DEFAULT)
    if 'video' in config:
        record_minutes = config['video'].get('record_minutes', record_minutes)
    
    duration_limit = record_minutes * 60

    print(f"[INFO] Connecting to Aravis Camera: {camera_id}...")
    try:
        camera = Aravis.Camera.new(camera_id)
    except Exception as e:
        print(f"[ERROR] Failed to find camera: {e}")
        return

    # --- CAMERA SETUP (Matching main_inference.py) ---
    print(f"[INFO] Applying Settings from main_inference.py logic...")
    
    camera.set_string("AcquisitionMode", "Continuous")
    
    fps_limit = 30.0
    try:
        camera.set_float("AcquisitionFrameRate", fps_limit)
        print(f"  - FPS set to: {fps_limit}")
    except:
        pass # Some cameras don't support setting FPS directly this way

    # Auto-detect resolution
    width = camera.get_integer("Width")
    height = camera.get_integer("Height")
    print(f"  - Resolution: {width}x{height}")
    
    payload = camera.get_payload()

    # Stream Setup
    stream = camera.create_stream(None, None)
    # Push enough buffers to handle jitter, but not too many to increase latency
    for _ in range(5): 
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    print("[INFO] Starting acquisition...")
    camera.start_acquisition()

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cam_name = camera_id.split('-')[-1] if '-' in camera_id else camera_id
    filename = os.path.join(SAVE_FOLDER, f"recording_{cam_name}_{timestamp}.mp4")

    # Variables for recording loop
    writer = None
    frames_recorded = 0
    start_time = time.time()
    
    # FPS Calculation for Writer
    frame_buffer = []
    FPS_CALC_FRAMES = 60
    WARMUP_FRAMES = 20
    warmup_counter = 0
    first_frame_time = 0
    
    print(f"[INFO] Recording started. Duration: {record_minutes} minutes.")
    print(f"[INFO] Press 'q' to stop early.")

    try:
        while True:
            # 1 sec timeout
            buffer = stream.timeout_pop_buffer(1000000) 
            
            if buffer:
                if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                    data = buffer.get_data()
                    
                    # Efficient reshaping (from main_inference.py)
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
                    
                    # Convert to BGR (from main_inference.py)
                    # Note: main_inference.py uses COLOR_BAYER_BG2BGR. 
                    image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

                    # --- RECORDING LOGIC ---
                    
                    # Initialize writer on first frame (with FPS auto-detection)
                    if writer is None:
                        # Warmup phase
                        if warmup_counter < WARMUP_FRAMES:
                            warmup_counter += 1
                            frame_buffer.append(image)
                        else:
                            current_time = time.time()
                            if first_frame_time == 0:
                                first_frame_time = current_time
                            
                            frame_buffer.append(image)
                            
                            frames_to_measure = len(frame_buffer) - WARMUP_FRAMES
                            if frames_to_measure >= FPS_CALC_FRAMES:
                                duration = current_time - first_frame_time
                                if duration > 0:
                                    actual_fps = (frames_to_measure - 1) / duration
                                else:
                                    actual_fps = fps_limit
                                
                                print(f"[INFO] Measured Capture FPS: {actual_fps:.2f}")
                                
                                # Setup Video Writer
                                # Try GStreamer first (Jetson)
                                gst_pipeline = get_gst_pipeline(filename, width, height, actual_fps)
                                try:
                                    writer = cv2.VideoWriter(
                                        gst_pipeline,
                                        cv2.CAP_GSTREAMER,
                                        0,
                                        actual_fps,
                                        (width, height)
                                    )
                                except:
                                    writer = None

                                if not writer or not writer.isOpened():
                                    print("[INFO] Fallback to standard VideoWriter (mp4v)")
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    writer = cv2.VideoWriter(filename, fourcc, actual_fps, (width, height))
                                
                                # Write buffered frames
                                for f in frame_buffer:
                                    writer.write(f)
                                frame_buffer = [] 
                    else:
                        writer.write(image)

                    # Display (Optional - can be disabled for performance)
                    # Downscale for display if 4K
                    display_h = 600
                    scale = display_h / height
                    display_w = int(width * scale)
                    preview = cv2.resize(image, (display_w, display_h))
                    
                    elapsed = time.time() - start_time
                    cv2.putText(preview, f"REC: {int(elapsed)}s / {int(duration_limit)}s", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Recorder", preview)
                    
                    frames_recorded += 1
                
                stream.push_buffer(buffer)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if (time.time() - start_time) > duration_limit:
                print(f"[INFO] Time limit reached.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("Stopping camera...")
        camera.stop_acquisition()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Saved: {filename}")

if __name__ == "__main__":
    main()
