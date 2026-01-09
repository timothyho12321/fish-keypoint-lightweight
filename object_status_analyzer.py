# object_status_analyzer.py
import numpy as np
from collections import deque, defaultdict
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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

class ObjectStatusAnalyzer:
    def __init__(self, min_confidence=0.3):
        """
        Object analyzer with 15-second update cycle:
        
        process_frame(detections, camera_type) → Called every frame, collects data per camera
        get_display_data() → Returns current stable values for the active camera
        Actual updates happen only every 15 seconds
        
        Args:
            min_confidence: Minimum detection confidence threshold
        """
        self.min_confidence = min_confidence
        
        # Update cycle (seconds)
        self.UPDATE_INTERVAL = 15.0  # Update stable values every 15 seconds
        self.ANALYSIS_WINDOW = 30.0  # Analyze last 30 seconds of data
        
        # Spatial clustering
        self.CLUSTER_DISTANCE = 40.0
        
        # Track which camera is currently active
        self.current_camera_type = 'top'
        
        # Data storage per camera type
        self.camera_data = {
            'top': self._init_camera_data(),
            'side': self._init_camera_data()
        }
        
        # Performance tracking
        self.start_time = time.time()
    
    def _init_camera_data(self):
        """Initialize data structure for a single camera"""
        return {
            'frame_buffer': deque(maxlen=300),
            'all_detections_history': deque(maxlen=1000),
            'cluster_counts_history': deque(maxlen=200),
            'current_frame_data': None,
            'stable_values': {
                'total': 0,
                'active': 0,
                'sick': 0,
                'dead': 0,
                'last_update': 0,
                'next_update': 0
            },
            'window_statistics': {
                'min_count': float('inf'),
                'max_count': 0,
                'mode_count': 0,
                'median_count': 0,
                'mean_count': 0,
                'count_std': 0
            },
            'frame_count': 0,
            'last_15s_window_end': time.time(),
            'windows_processed': 0,
            'current_window_data': {
                'start_time': time.time(),
                'frame_count': 0,
                'detection_counts': [],
                'cluster_counts': [],
                'snapshots': [],
                'status_data': []
            }
        }
    
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
            is_vertical=is_vertical
        )
    
    def cluster_detections(self, snapshots: List[ObjectSnapshot]) -> Dict:
        """Cluster detections for current frame"""
        if not snapshots:
            return {}
        
        clusters = {}
        used_indices = set()
        
        for i, snapshot in enumerate(snapshots):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster_members = [snapshot]
            used_indices.add(i)
            
            # Find nearby snapshots
            for j, other in enumerate(snapshots):
                if j in used_indices:
                    continue
                
                distance = math.sqrt(
                    (snapshot.center[0] - other.center[0])**2 +
                    (snapshot.center[1] - other.center[1])**2
                )
                
                if distance < self.CLUSTER_DISTANCE:
                    cluster_members.append(other)
                    used_indices.add(j)
            
            # Create cluster
            cluster_id = len(clusters)
            clusters[cluster_id] = {
                'members': cluster_members,
                'center': snapshot.center,
                'timestamp': snapshot.timestamp,
                'avg_area': np.mean([m.area for m in cluster_members])
            }
        
        return clusters
    
    def analyze_frame_status(self, clusters: Dict) -> Dict:
        """Quick status analysis for current frame"""
        active = 0
        sick = 0
        dead = 0
        
        current_time = time.time()
        analysis_window = 5.0  # Short window for frame analysis
        
        for cluster_id, cluster_data in clusters.items():
            members = cluster_data['members']
            if len(members) < 2:
                active += 1
                continue
            
            # Get recent members
            recent_members = [m for m in members if m.timestamp >= current_time - analysis_window]
            if len(recent_members) < 2:
                active += 1
                continue
            
            # Simple movement analysis
            speeds = []
            vertical_count = 0
            
            for i in range(1, len(recent_members)):
                prev = recent_members[i-1]
                curr = recent_members[i]
                
                # Movement
                dx = curr.center[0] - prev.center[0]
                dy = curr.center[1] - prev.center[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                time_diff = curr.timestamp - prev.timestamp
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
                
                # Orientation
                if curr.is_vertical:
                    vertical_count += 1
            
            if not speeds:
                avg_speed = 0
            else:
                avg_speed = np.mean(speeds)
            
            vertical_ratio = vertical_count / len(recent_members)
            
            # Simple classification
            if avg_speed < 0.1 and vertical_ratio > 0.8:
                dead += 1
            elif avg_speed < 0.5 or vertical_ratio > 0.6:
                sick += 1
            else:
                active += 1
        
        return {
            'active': active,
            'sick': sick,
            'dead': dead,
            'total': active + sick + dead
        }
    
    def process_frame(self, current_detections: List[Dict], camera_type: str = 'top') -> Dict:
        """
        Process a frame - called EVERY frame
        Only collects data, doesn't update stable values
        
        Args:
            current_detections: List of detection dictionaries
            camera_type: 'top' or 'side' to track data separately
        """
        current_time = time.time()
        self.current_camera_type = camera_type
        
        # Get camera-specific data
        cam_data = self.camera_data[camera_type]
        cam_data['frame_count'] += 1
        
        # Create snapshots
        snapshots = []
        for det in current_detections:
            if self.validate_detection(det):
                snapshot = self.create_snapshot(det)
                snapshots.append(snapshot)
        
        # Cluster detections
        clusters = self.cluster_detections(snapshots)
        cluster_count = len(clusters)
        
        # Analyze status for this frame
        status = self.analyze_frame_status(clusters)
        
        # Store frame data in buffer
        frame_data = {
            'frame': cam_data['frame_count'],
            'timestamp': current_time,
            'detections': len(snapshots),
            'clusters': cluster_count,
            'snapshots': snapshots,
            'status': status,
            'cluster_data': clusters
        }
        
        cam_data['frame_buffer'].append(frame_data)
        cam_data['all_detections_history'].extend(snapshots)
        cam_data['cluster_counts_history'].append(cluster_count)
        
        # Add to current 15-second window
        cam_data['current_window_data']['frame_count'] += 1
        cam_data['current_window_data']['detection_counts'].append(len(snapshots))
        cam_data['current_window_data']['cluster_counts'].append(cluster_count)
        cam_data['current_window_data']['snapshots'].extend(snapshots)
        cam_data['current_window_data']['status_data'].append(status)
        
        # Check if 15 seconds have passed
        time_in_window = current_time - cam_data['current_window_data']['start_time']
        
        if time_in_window >= self.UPDATE_INTERVAL:
            # Time to update stable values
            self.update_stable_values(camera_type)
            
            # Reset window
            cam_data['current_window_data'] = {
                'start_time': current_time,
                'frame_count': 0,
                'detection_counts': [],
                'cluster_counts': [],
                'snapshots': [],
                'status_data': []
            }
            cam_data['last_15s_window_end'] = current_time
            cam_data['windows_processed'] += 1
        
        # Store current frame data
        cam_data['current_frame_data'] = {
            'frame': cam_data['frame_count'],
            'timestamp': current_time,
            'detections': len(snapshots),
            'clusters': cluster_count,
            'status': status,
            'next_update_in': max(0, cam_data['stable_values']['next_update'] - current_time)
        }
        
        return cam_data['current_frame_data']
    
    def update_stable_values(self, camera_type: str = 'top'):
        """
        Update stable values using last 15 seconds of data for specified camera
        Called automatically every 15 seconds
        """
        current_time = time.time()
        cam_data = self.camera_data[camera_type]
        
        # Get data from last 15 seconds
        window_start = current_time - self.ANALYSIS_WINDOW
        recent_frames = [f for f in cam_data['frame_buffer'] if f['timestamp'] >= window_start]
        
        if not recent_frames:
            return  # No data to analyze
        
        # Extract cluster counts from recent frames
        recent_cluster_counts = [f['clusters'] for f in recent_frames]
        recent_status_data = [f['status'] for f in recent_frames]
        
        # Calculate stable total count
        stable_total = self.calculate_stable_total(recent_cluster_counts, cam_data['stable_values'])
        
        # Calculate stable status counts
        stable_status = self.calculate_stable_status(recent_status_data, stable_total, cam_data['stable_values'])
        
        # Update statistics
        self.update_window_statistics(recent_cluster_counts, cam_data['window_statistics'])
        
        # Update stable values
        cam_data['stable_values']['total'] = stable_total
        cam_data['stable_values']['active'] = stable_status['active']
        cam_data['stable_values']['sick'] = stable_status['sick']
        cam_data['stable_values']['dead'] = stable_status['dead']
        cam_data['stable_values']['last_update'] = current_time
        cam_data['stable_values']['next_update'] = current_time + self.UPDATE_INTERVAL
        
        # Log update
        print(f"[{camera_type}][{time.strftime('%H:%M:%S')}] Stable values updated:")
        print(f"  Total: {stable_total}, Active: {stable_status['active']}, "
              f"Sick: {stable_status['sick']}, Dead: {stable_status['dead']}")
    
    def calculate_stable_total(self, recent_counts: List[int], stable_values: Dict) -> int:
        """Calculate stable total from recent counts"""
        if not recent_counts:
            return stable_values['total']  # Keep previous value
        
        # Strategy 1: Mode (most frequent)
        count_freq = {}
        for count in recent_counts:
            count_freq[count] = count_freq.get(count, 0) + 1
        
        if count_freq:
            mode_count = max(count_freq.items(), key=lambda x: x[1])[0]
        else:
            mode_count = 0
        
        # Strategy 2: Median (robust)
        median_count = int(np.median(recent_counts)) if recent_counts else 0
        
        # Strategy 3: Weighted towards higher counts (account for occlusion)
        weights = np.linspace(0.1, 1.0, len(recent_counts))
        weights = weights / weights.sum()
        weighted_avg = np.average(recent_counts, weights=weights)
        
        # Combine strategies
        if mode_count == median_count:
            stable_total = mode_count
        elif abs(mode_count - weighted_avg) <= 2:
            stable_total = int((mode_count * 0.6 + weighted_avg * 0.4) + 0.5)
        else:
            stable_total = median_count
        
        # Smooth with previous value
        if stable_values['total'] > 0:
            change = abs(stable_total - stable_values['total'])
            if change <= 3:
                # Small change, smooth it
                smoothing = 0.7
                stable_total = int(stable_values['total'] * smoothing + stable_total * (1 - smoothing))
        
        return max(0, stable_total)
    
    def calculate_stable_status(self, recent_status_data: List[Dict], total_count: int, stable_values: Dict) -> Dict:
        """Calculate stable status counts from recent data"""
        if not recent_status_data or total_count == 0:
            return {
                'active': stable_values['active'],
                'sick': stable_values['sick'],
                'dead': stable_values['dead']
            }
        
        # Calculate average proportions from recent frames
        avg_active = np.mean([s['active'] for s in recent_status_data])
        avg_sick = np.mean([s['sick'] for s in recent_status_data])
        avg_dead = np.mean([s['dead'] for s in recent_status_data])
        avg_total = avg_active + avg_sick + avg_dead
        
        if avg_total > 0:
            # Calculate proportions
            active_prop = avg_active / avg_total
            sick_prop = avg_sick / avg_total
            dead_prop = avg_dead / avg_total
            
            # Scale to total count
            active_count = int(total_count * active_prop + 0.5)
            sick_count = int(total_count * sick_prop + 0.5)
            dead_count = int(total_count * dead_prop + 0.5)
            
            # Ensure they sum to total
            total_from_status = active_count + sick_count + dead_count
            if total_from_status != total_count:
                # Adjust active count
                active_count = total_count - sick_count - dead_count
        else:
            # Default proportions
            active_count = total_count
            sick_count = 0
            dead_count = 0
        
        # Smooth with previous values
        smoothing = 0.6
        active_count = int(stable_values['active'] * smoothing + active_count * (1 - smoothing))
        sick_count = int(stable_values['sick'] * smoothing + sick_count * (1 - smoothing))
        dead_count = int(stable_values['dead'] * smoothing + dead_count * (1 - smoothing))
        
        # Final consistency check
        total_check = active_count + sick_count + dead_count
        if total_check != total_count:
            active_count = total_count - sick_count - dead_count
        
        return {
            'active': max(0, active_count),
            'sick': max(0, sick_count),
            'dead': max(0, dead_count)
        }
    
    def update_window_statistics(self, recent_counts: List[int], window_statistics: Dict):
        """Update window statistics"""
        if not recent_counts:
            return
        
        window_statistics['min_count'] = min(recent_counts)
        window_statistics['max_count'] = max(recent_counts)
        window_statistics['mean_count'] = np.mean(recent_counts)
        window_statistics['median_count'] = np.median(recent_counts)
        window_statistics['count_std'] = np.std(recent_counts) if len(recent_counts) > 1 else 0
        
        # Calculate mode
        count_freq = {}
        for count in recent_counts:
            count_freq[count] = count_freq.get(count, 0) + 1
        
        if count_freq:
            window_statistics['mode_count'] = max(count_freq.items(), key=lambda x: x[1])[0]
    
    def get_display_data(self) -> Dict:
        """
        Get display data for the current camera
        Returns counts including current frame detection count
        """
        cam_data = self.camera_data[self.current_camera_type]
        
        # Get stable counts
        stable_values = cam_data['stable_values']
        
        # Get current frame data if available
        current_detections = 0
        if cam_data['current_frame_data']:
            current_detections = cam_data['current_frame_data'].get('detections', 0)
        
        return {
            'total': stable_values['total'],
            'active': stable_values['active'],
            'sick': stable_values['sick'],
            'dead': stable_values['dead'],
            'current': current_detections  # Current frame count
        }
    
    def get_stable_counts(self, camera_type: str = None) -> Dict:
        """
        Get current stable counts for specified camera
        Can be called continuously, returns last 15-second update
        """
        if camera_type is None:
            camera_type = self.current_camera_type
        
        cam_data = self.camera_data[camera_type]
        stable_values = cam_data['stable_values']
        current_time = time.time()
        time_since_update = current_time - stable_values['last_update']
        time_to_next_update = max(0, stable_values['next_update'] - current_time)
        
        return {
            'total': stable_values['total'],
            'active': stable_values['active'],
            'sick': stable_values['sick'],
            'dead': stable_values['dead'],
            'last_update_seconds_ago': time_since_update,
            'next_update_in_seconds': time_to_next_update,
            'is_fresh': time_since_update < self.UPDATE_INTERVAL
        }
    
    def get_current_frame_data(self, camera_type: str = None) -> Optional[Dict]:
        """Get most recent frame data for specified camera"""
        if camera_type is None:
            camera_type = self.current_camera_type
        return self.camera_data[camera_type]['current_frame_data']
    
    def get_statistics(self, camera_type: str = None) -> Dict:
        """Get comprehensive statistics for specified camera"""
        if camera_type is None:
            camera_type = self.current_camera_type
        
        cam_data = self.camera_data[camera_type]
        current_time = time.time()
        
        return {
            'camera_type': camera_type,
            'stable_counts': self.get_stable_counts(camera_type),
            'window_statistics': cam_data['window_statistics'].copy(),
            'performance': {
                'total_frames': cam_data['frame_count'],
                'windows_processed': cam_data['windows_processed'],
                'frames_in_current_window': cam_data['current_window_data']['frame_count'],
                'time_in_current_window': current_time - cam_data['current_window_data']['start_time'],
                'uptime_minutes': (current_time - self.start_time) / 60.0
            },
            'data_volume': {
                'frame_buffer_size': len(cam_data['frame_buffer']),
                'total_detections': len(cam_data['all_detections_history']),
                'total_cluster_counts': len(cam_data['cluster_counts_history'])
            }
        }
    
    def reset(self, camera_type: str = None):
        """Reset analyzer for specified camera (or all cameras if None)"""
        if camera_type is None:
            # Reset all cameras
            for cam_type in ['top', 'side']:
                self.camera_data[cam_type] = self._init_camera_data()
        else:
            # Reset specific camera
            self.camera_data[camera_type] = self._init_camera_data()
    
    def force_update(self, camera_type: str = None):
        """Force immediate update of stable values for specified camera"""
        if camera_type is None:
            camera_type = self.current_camera_type
        print(f"Forcing update of stable values for {camera_type}...")
        self.update_stable_values(camera_type)