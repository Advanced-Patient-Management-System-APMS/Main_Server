import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from ultralytics import YOLO
from pytubefix import YouTube, Search
import time
from collections import deque



# Initialize YOLOv8-Pose model
model = YOLO('//best.onnx')
# Minimum confidence threshold for keypoint detection
"""TODO: 조정값 실험 및 조정값 분리"""
MIN_CONFIDENCE = 0.2  # keypoint confidence threshold
VELOCITY_THRESHOLD = 5.0  # fall detection rapid descent velocity threshold (pixels/frame)
ANGLE_THRESHOLD = 30.0  # this value or lower is considered horizontal (lying down) (degrees)
ANGLE_CHANGE_THRESHOLD = 45.0  # rapid angle change to detect falls (degrees)
SPREAD_INCREASE_THRESHOLD = 20.0  # limb spread increase during falls (pixels)
# Frame rate assumption for velocity calculation (adjust based on your video)
ASSUMED_FPS = 30.0


# ZONE DEFINITION
# BED_ZONE: Area of Interest - covers the mattress area
BED_ZONE = np.array([
    [200, 100],   # Top-left corner of bed
    [1080, 100],  # Top-right corner of bed
    [1080, 500],  # Bottom-right corner of bed
    [200, 500]    # Bottom-left corner of bed
], dtype=np.int32)
# FLOOR_ZONE: Danger Zone - covers floor area visible around the bed
FLOOR_ZONE = np.array([
    [0, 500],      # Top-left of floor zone
    [1280, 500],   # Top-right of floor zone
    [1280, 720],   # Bottom-right (frame bottom)
    [0, 720]       # Bottom-left (frame bottom)
], dtype=np.int32)

# State definitions
STATE_ON_BED = "ON_BED"
STATE_ON_FLOOR = "ON_FLOOR"
STATE_UNDEFINED = "UNDEFINED"


def get_yolo_pose_keypoints(frame: np.ndarray) -> Tuple[List[Dict[str, Tuple[float, float]]], List[np.ndarray]]:
    """NOTE: YOLO 모델을 사용하여 프레임에서 사람의 키포인트를 감지."""
    results = model(frame, verbose=False)
    KEYPOINT_MAP = {
        0: 'nose',
        5: 'left_shoulder',
        6: 'right_shoulder',
        11: 'left_hip',
        12: 'right_hip'
    }
    
    detected_persons = []
    detected_boxes = []
    
    # Check if any detections were made
    if results and len(results) > 0 and results[0].keypoints is not None:
        keypoints_data = results[0].keypoints
        
        # Loop through each detected person
        num_persons = len(keypoints_data.xy)
        
        for person_idx in range(num_persons):
            kpts_xy = keypoints_data.xy[person_idx].cpu().numpy()  # Shape: (17, 2)
            kpts_conf = keypoints_data.conf[person_idx].cpu().numpy()  # Shape: (17,)
            bbox_xyxy = keypoints_data.xyxy[person_idx].cpu().numpy()
            
            # Build keypoint dictionary for this person
            person_keypoints = {}
            
            for coco_idx, keypoint_name in KEYPOINT_MAP.items():
                # Check if confidence meets threshold
                if kpts_conf[coco_idx] >= MIN_CONFIDENCE:
                    x, y = kpts_xy[coco_idx]
                    person_keypoints[keypoint_name] = (float(x), float(y))
            
            # Only add person if they have at least some valid keypoints
            if len(person_keypoints) > 0:
                detected_persons.append(person_keypoints)
                detected_boxes.append(bbox_xyxy)
    
    return detected_persons, detected_boxes


def calculate_body_center(keypoints: Dict[str, Tuple[float, float]], 
                          bbox: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
    """NOTE: 키포인트와 BBox를 사용하여 신체 중심 좌표 계산."""
    
    if 'nose' in keypoints:
        return keypoints['nose']
    
    required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    available_coords = []
    for point_name in required_points:
        if point_name in keypoints:
            available_coords.append(keypoints[point_name])
    
    if len(available_coords) >= 2:
        avg_x = sum(coord[0] for coord in available_coords) / len(available_coords)
        avg_y = sum(coord[1] for coord in available_coords) / len(available_coords)
        return (avg_x, avg_y)

    if bbox is not None:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return (center_x, center_y)
    return None


def calculate_body_angle(keypoints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """NOTE: 키포인트를 사용하여 신체 각도 계산."""
    if 'left_shoulder' in keypoints and 'left_hip' in keypoints:
        shoulder = keypoints['left_shoulder']
        hip = keypoints['left_hip']
    elif 'right_shoulder' in keypoints and 'right_hip' in keypoints:
        shoulder = keypoints['right_shoulder']
        hip = keypoints['right_hip']
    else:
        return None
    
    dy = hip[1] - shoulder[1]
    dx = hip[0] - shoulder[0]
    
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    
    if angle > 90:
        angle = 180 - angle # Normalize to [0, 90]
    
    return angle


def calculate_limb_spread(keypoints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """NOTE: 키포인트를 사용하여 팔다리 벌어짐 계산."""
    required = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    
    if not all(k in keypoints for k in required):
        return None
    
    shoulder_width = np.linalg.norm(
        np.array(keypoints['left_shoulder']) - 
        np.array(keypoints['right_shoulder'])
    )
    
    hip_width = np.linalg.norm(
        np.array(keypoints['left_hip']) - 
        np.array(keypoints['right_hip'])
    )
    
    return shoulder_width + hip_width


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """NOTE: 점이 다각형 내부에 있는지 확인."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def determine_state(body_center: Tuple[float, float]) -> str:
    """NOTE: 신체 중심 위치에 따라 상태 결정."""
    if point_in_polygon(body_center, BED_ZONE):
        return STATE_ON_BED
    elif point_in_polygon(body_center, FLOOR_ZONE):
        return STATE_ON_FLOOR
    else:
        return STATE_UNDEFINED


# DETECTION LOGIC

class FallDetector:
    """NOTE: 낙상 감지 클래스."""
    
    def __init__(self, fps: float = ASSUMED_FPS):
        self.fps = fps
        self.pose_history = [] 
        self.history_size = 30  
        
        self.angle_threshold = ANGLE_THRESHOLD
        self.angle_change_threshold = ANGLE_CHANGE_THRESHOLD
        self.velocity_threshold = VELOCITY_THRESHOLD
        self.spread_increase_threshold = SPREAD_INCREASE_THRESHOLD
        
        self.fall_detected = False
        self.fall_frame = None
        
        # --- Occupancy Detection (재실/부재 감지) ---
        self.last_seen_frame = 0
        self.occupancy_state = "UNKNOWN" # "PRESENT", "ABSENT"
        self.occupancy_timeout_frames = int(self.fps * 10) 

        # --- Agitation Detection (위험행동/뒤척임 감지) ---
        self.agitation_window_size = int(self.fps * 5) 
        self.agitation_buffer = deque(maxlen=self.agitation_window_size)
        self.agitation_threshold = 10.0 
        self.is_agitated = False
        
    def reset(self):
        """NOTE: 감지기 상태 리셋."""
        self.pose_history = []
        self.fall_detected = False
        self.fall_frame = None
        
        self.last_seen_frame = 0
        self.occupancy_state = "UNKNOWN"
        self.agitation_buffer.clear()
        self.is_agitated = False
    
    def _update_occupancy(self, body_center_present: bool, frame_num: int):
        """NOTE: 재실/부재 상태 업데이트."""
        if body_center_present:
            self.last_seen_frame = frame_num
            self.occupancy_state = "PRESENT"
        elif (frame_num - self.last_seen_frame) > self.occupancy_timeout_frames:
            self.occupancy_state = "ABSENT"
            self.agitation_buffer.clear() 
            self.is_agitated = False

    def _update_agitation(self, body_center: Optional[Tuple[float, float]]):
        """NOTE: 위험행동 상태 업데이트."""
        if body_center is None or self.occupancy_state == "ABSENT":
            self.agitation_buffer.append(0.0)
            self.is_agitated = False
            return

        movement_velocity = 0.0
        if self.pose_history:
             last_pose = self.pose_history[-1]
             if last_pose['center'] is not None:
                 movement_velocity = np.linalg.norm(
                     np.array(body_center) - np.array(last_pose['center'])
                 )
        
        self.agitation_buffer.append(movement_velocity)

        if len(self.agitation_buffer) == self.agitation_window_size:
            valid_movements = [m for m in self.agitation_buffer if m > 0.1]
            if valid_movements:
                avg_velocity = np.mean(valid_movements)
                self.is_agitated = avg_velocity > self.agitation_threshold
            else:
                self.is_agitated = False
    
    def update(self, keypoints: Dict[str, Tuple[float, float]], 
               body_center: Optional[Tuple[float, float]], 
               bbox: Optional[np.ndarray], 
               frame_num: int) -> bool:
        """NOTE: 프레임별 업데이트 및 낙상 감지."""
        
        # --- 1. 재실/부재 및 위험행동 상태 업데이트 ---
        body_center_present = body_center is not None
        self._update_occupancy(body_center_present, frame_num)
        
        # --- 2. 낙상 감지를 위한 포즈 피처 추출 ---
        if not body_center_present:
            self.pose_history.append({
                'frame': frame_num, 'angle': None, 'spread': None, 'center': None
            })
            if len(self.pose_history) > self.history_size:
                self.pose_history.pop(0)
            
            self._update_agitation(None)
            return False
            
        has_keypoints = bool(keypoints)
        if has_keypoints:
            angle = calculate_body_angle(keypoints)
            spread = calculate_limb_spread(keypoints)
            angle = angle if angle is not None else 30.0 
            spread = spread if spread is not None else 0.0
        else:
            angle = 0.0 
            spread = 0.0
        
        # --- 3. 포즈 저장 ---
        current_pose = {
            'frame': frame_num,
            'angle': angle,
            'spread': spread,
            'center': body_center
        }
        self.pose_history.append(current_pose)
        
        self._update_agitation(body_center)
        
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
        
        if len(self.pose_history) < 10:
            return False
        
        
        # --- 4. 낙상 패턴 분석 ---
        if self.fall_detected:
            return False
            
        fall_detected_in_pattern = self._analyze_fall_pattern(has_keypoints)
        
        if fall_detected_in_pattern:
            self.fall_detected = True
            self.fall_frame = frame_num
            return True
        
        return False
    
    def _analyze_fall_pattern(self, has_keypoints: bool) -> bool:
        """NOTE: 낙상 패턴 분석."""
        if len(self.pose_history) < 10:
            return False
        
        recent_poses = self.pose_history[-10:] 
        
        angle_changes = []
        for i in range(1, len(recent_poses)):
            angle_diff = abs(recent_poses[i]['angle'] - recent_poses[i-1]['angle'])
            angle_changes.append(angle_diff)
        
        max_angle_change = max(angle_changes) if angle_changes else 0
        
        current_angle = recent_poses[-1]['angle']
        is_horizontal = current_angle < self.angle_threshold
        
        y_velocities = []
        for i in range(1, len(recent_poses)):
            if recent_poses[i]['center'] is None or recent_poses[i-1]['center'] is None:
                continue
            
            dy = recent_poses[i]['center'][1] - recent_poses[i-1]['center'][1]
            dt = recent_poses[i]['frame'] - recent_poses[i-1]['frame']
            
            if dt > 0:
                y_velocities.append(dy / dt) 
            
        max_velocity = max(y_velocities) if y_velocities else 0
        
        spread_increase = (recent_poses[-1]['spread'] - 
                          recent_poses[0]['spread'])
        
        if has_keypoints:
            fall_detected = (
                (max_angle_change > self.angle_change_threshold and is_horizontal) or
                
                (max_velocity > self.velocity_threshold and is_horizontal) or
                
                (max_angle_change > 20 and 
                 max_velocity > 3.0 and 
                 spread_increase > self.spread_increase_threshold)
            )
        else:
            fall_detected = (
                max_velocity > (self.velocity_threshold * 0.8) and
                is_horizontal 
            )
        
        return fall_detected


# VISUALIZATION FUNCTIONS
def draw_zones(frame: np.ndarray) -> np.ndarray:
    """NOTE: BED_ZONE 및 FLOOR_ZONE 시각화."""
    overlay = frame.copy()
    
    cv2.fillPoly(overlay, [BED_ZONE], color=(0, 255, 0))
    cv2.polylines(frame, [BED_ZONE], isClosed=True, color=(0, 255, 0), thickness=2)
    
    cv2.fillPoly(overlay, [FLOOR_ZONE], color=(0, 0, 255))
    cv2.polylines(frame, [FLOOR_ZONE], isClosed=True, color=(0, 0, 255), thickness=2)
    
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    return frame


def draw_body_center(frame: np.ndarray, body_center: Tuple[float, float]) -> np.ndarray:
    """NOTE: 신체 중심점 시각화."""
    center_int = (int(body_center[0]), int(body_center[1]))
    cv2.circle(frame, center_int, radius=10, color=(255, 0, 0), thickness=-1)
    cv2.circle(frame, center_int, radius=12, color=(255, 255, 255), thickness=2)
    return frame


def draw_state_text(frame: np.ndarray, state: str, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """NOTE: 상태 텍스트 시각화."""
    color_map = {
        STATE_ON_BED: (0, 255, 0),      # Green
        STATE_ON_FLOOR: (0, 165, 255),  # Orange
        STATE_UNDEFINED: (128, 128, 128) # Gray
    }
    
    color = color_map.get(state, (255, 255, 255))
    text = f"State: {state}"
    
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (position[0] - 5, position[1] - text_height - 5),
                  (position[0] + text_width + 5, position[1] + 5),
                  (0, 0, 0), -1)
    
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, color, 2, cv2.LINE_AA)
    
    return frame


def draw_fall_alert(frame: np.ndarray, frame_num: int) -> np.ndarray:
    """NOTE: 낙상 경고 시각화."""
    if (frame_num % 20) < 10:
        text = "FALL DETECTED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 4
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        frame_height, frame_width = frame.shape[:2]
        x = (frame_width - text_width) // 2
        y = (frame_height + text_height) // 2
        
        padding = 20
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     (0, 0, 0), -1)
        
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     (0, 0, 255), 4)
        
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return frame

def auto_calibrate_zones(video_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """NOTE: 비디오 내용을 기반으로 BED_ZONE 및 FLOOR_ZONE 자동 보정."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return BED_ZONE, FLOOR_ZONE
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    body_centers = []
    
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        
        persons, boxes = get_yolo_pose_keypoints(frame)
        if persons:
            center = calculate_body_center(persons[0], boxes[0] if boxes else None)
            if center:
                body_centers.append(center)
    
    cap.release()
    
    if not body_centers:
        return BED_ZONE, FLOOR_ZONE
    
    centers_array = np.array(body_centers)
    min_x, min_y = centers_array.min(axis=0) - 100 
    max_x, max_y = centers_array.max(axis=0) + 100
    
    min_x = max(0, int(min_x))
    min_y = max(0, int(min_y))
    max_x = min(width, int(max_x))
    max_y = min(height, int(max_y))
    
    bed_zone = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype=np.int32)
    floor_zone = np.array([
        [0, max_y],
        [width, max_y],
        [width, height],
        [0, height]
    ], dtype=np.int32)
    
    print(f"    BED_ZONE: [{min_x},{min_y}] to [{max_x},{max_y}]")
    print(f"    FLOOR_ZONE: below y={max_y}")
    
    return bed_zone, floor_zone


def process_video(video_path: Path, output_path: Path, use_auto_calibration: bool = True) -> Dict:
    """NOTE: 비디오 파일을 처리하고 낙상 감지 및 주석 비디오 생성."""
    print(f"Processing: {video_path}")
    
    if use_auto_calibration:
        bed_zone, floor_zone = auto_calibrate_zones(video_path)
    else:
        bed_zone, floor_zone = BED_ZONE, FLOOR_ZONE
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {'file': str(video_path), 'error': 'Could not open video', 'fall_detected': False}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = ASSUMED_FPS
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    detector = FallDetector(fps=fps)
    
    frame_num = 0
    fall_detected_overall = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        persons, boxes = get_yolo_pose_keypoints(frame)
        
        current_state = STATE_UNDEFINED
        body_center = None
        keypoints = {}
        bbox = None
        
        if len(persons) > 0:
            keypoints = persons[0]
            bbox = boxes[0]
            
            body_center = calculate_body_center(keypoints, bbox)
            
            if body_center:
                if point_in_polygon(body_center, bed_zone):
                    current_state = STATE_ON_BED
                elif point_in_polygon(body_center, floor_zone):
                    current_state = STATE_ON_FLOOR
                else:
                    current_state = STATE_UNDEFINED
        else:
            keypoints = {}
            bbox = None
            body_center = None
        
        fall_in_frame = detector.update(keypoints, body_center, bbox, frame_num)
        if fall_in_frame:
            fall_detected_overall = True
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [bed_zone], color=(0, 255, 0))
        cv2.polylines(frame, [bed_zone], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(overlay, [floor_zone], color=(0, 0, 255))
        cv2.polylines(frame, [floor_zone], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        if body_center:
            frame = draw_body_center(frame, body_center)
        
        frame = draw_state_text(frame, current_state)

        occ_text = f"Occupancy: {detector.occupancy_state}"
        occ_color = (0, 255, 255) if detector.occupancy_state == "PRESENT" else (128, 128, 128)
        cv2.putText(frame, occ_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, occ_color, 2, cv2.LINE_AA)

        if detector.is_agitated:
            agit_text = "AGITATION DETECTED"
            cv2.rectangle(frame, (8, 85), (450, 125), (0, 0, 0), -1) 
            cv2.putText(frame, agit_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 255), 2, cv2.LINE_AA) 
        
        if detector.fall_detected:
            frame = draw_fall_alert(frame, frame_num)
        
        out.write(frame)
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Progress: {progress:.1f}%", end='\r')
    
    cap.release()
    out.release()
    
    print(f"\n  Completed: Fall {'DETECTED' if fall_detected_overall else 'NOT DETECTED'}")
    
    return {
        'file': str(video_path),
        'output_file': str(output_path),
        'fall_detected': fall_detected_overall,
        'fall_frame': detector.fall_frame,
        'total_frames': frame_num,
        'fps': fps
    }


# DATA SOURCING (YouTube Download)
def download_test_videos(max_videos_per_category: int = 50):
    """NOTE: YouTube에서 테스트 비디오 다운로드."""
    print("=" * 80)
    print("DOWNLOADING TEST VIDEOS FROM YOUTUBE")
    
    fall_dir = Path("./test_videos/fall")
    non_fall_dir = Path("./test_videos/non_fall")
    fall_dir.mkdir(parents=True, exist_ok=True)
    non_fall_dir.mkdir(parents=True, exist_ok=True)
    
    fall_queries = [
        "cctv fall from bed",
        "hospital patient falls out of bed",
        "elderly fall caught on camera",
        "patient fall cctv",            
        "nursing home fall",             
        "patient falling from bed video", 
        "hospital fall accident",        
        "bed fall dataset",              
        "fall detection in hospital room"
    ]
    non_fall_queries = [
        "getting out of bed",           
        "rolling over in bed",            
        "morning stretching in bed",     
        "getting into bed",               
        "sitting on edge of bed",         
        "nurse assisting patient in bed", 
        "patient repositioning in bed",   
        "lying down quickly on bed",      
        "hospital patient ADL",           
        "picking up object from floor near bed" 
    ]
    
    def download_videos_for_queries(queries: List[str], output_dir: Path, 
                                     max_videos: int, category_name: str):
        """NOTE: 주어진 쿼리 목록에 대해 비디오 다운로드."""
        downloaded = 0
        
        for query in queries:
            if downloaded >= max_videos:
                break
            
            print(f"\n[{category_name}] Searching: '{query}'")
            
            try:
                # Search YouTube
                search = Search(query)
                results = search.videos[:10] 
                
                for video in results:
                    if downloaded >= max_videos:
                        break
                    
                    try:
                        print(f"  Downloading: {video.title[:50]}...")
                        
                        yt = YouTube(video.watch_url)
                        stream = yt.streams.filter(file_extension='mp4', 
                                                   progressive=True).order_by('resolution').desc().first()
                        
                        if stream:
                            safe_title = "".join(c for c in video.title if c.isalnum() or c in (' ', '-', '_'))[:50]
                            output_file = output_dir / f"{safe_title}_{video.video_id}.mp4"
                            
                            stream.download(output_path=str(output_dir), 
                                          filename=output_file.name)
                            
                            downloaded += 1
                            print(f"Downloaded ({downloaded}/{max_videos})")
                        else:
                            print(f"No suitable stream found")
                        
                        time.sleep(2)
                        
                    except Exception as e:
                        print(f"✗ Error downloading video: {e}")
                        continue
            
            except Exception as e:
                print(f"✗ Error searching for '{query}': {e}")
                continue
        
        print(f"\n[{category_name}] Downloaded {downloaded} videos")
        return downloaded
    
    fall_count = download_videos_for_queries(fall_queries, fall_dir, 
                                             max_videos_per_category, "FALL")
    
    non_fall_count = download_videos_for_queries(non_fall_queries, non_fall_dir, 
                                                  max_videos_per_category, "NON-FALL")
    
    print("\n" + "=" * 80)
    print(f"DOWNLOAD COMPLETE: {fall_count} fall videos, {non_fall_count} non-fall videos")
    print("=" * 80)
    
    return fall_count, non_fall_count


# EVALUATION & REPORTING
def run_evaluation():
    """NOTE: 전체 평가 파이프라인 실행"""    
    print("\nStep 1: Downloading test videos...")
    try:
        download_test_videos(max_videos_per_category=25)
    except Exception as e:
        print(f"Warning: Error during download: {e}")


    print("\nStep 2: Processing videos and generating annotations...")

    test_dir = Path("./test_videos")
    annotated_dir = Path("./test_videos/annotated")
    
    results = []
    
    fall_dir = test_dir / "fall"
    if fall_dir.exists():
        fall_videos = list(fall_dir.glob("*.mp4"))
        print(f"\nProcessing {len(fall_videos)} FALL videos...")
        
        for video_path in fall_videos:
            output_path = annotated_dir / "fall" / f"{video_path.stem}_annotated.mp4"
            
            result = process_video(video_path, output_path)
            result['ground_truth'] = 'FALL'
            result['relative_path'] = f"fall/{video_path.name}"
            
            if result.get('fall_detected', False):
                result['classification'] = 'TP'  # True Positive
            else:
                result['classification'] = 'FN'  # False Negative
            
            results.append(result)
    
    non_fall_dir = test_dir / "non_fall"
    if non_fall_dir.exists():
        non_fall_videos = list(non_fall_dir.glob("*.mp4"))
        print(f"\nProcessing {len(non_fall_videos)} NON-FALL videos...")
        
        for video_path in non_fall_videos:
            output_path = annotated_dir / "non_fall" / f"{video_path.stem}_annotated.mp4"
            
            result = process_video(video_path, output_path)
            result['ground_truth'] = 'NON_FALL'
            result['relative_path'] = f"non_fall/{video_path.name}"
            
            if result.get('fall_detected', False):
                result['classification'] = 'FP'  # False Positive
            else:
                result['classification'] = 'TN'  # True Negative
            
            results.append(result)
    
    # Step 3: Generate report
    print("\n" + "=" * 80)
    print("Step 3: Generating evaluation report...")
    
    generate_report(results)
    print(f"Annotated videos saved to: {annotated_dir}")


def generate_report(results: List[Dict]):
    """NOTE: 평가 결과를 기반으로 보고서 생성."""
    # Calculate metrics
    tp = sum(1 for r in results if r.get('classification') == 'TP')
    fn = sum(1 for r in results if r.get('classification') == 'FN')
    fp = sum(1 for r in results if r.get('classification') == 'FP')
    tn = sum(1 for r in results if r.get('classification') == 'TN')
    
    total = tp + fn + fp + tn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Bed Fall Detection System - Evaluation Report

**Generated:** {timestamp}

---


| Metric | Value |
|--------|-------|
| **Total Videos Processed** | {total} |
| **Accuracy** | {accuracy:.2%} |
| **Precision** | {precision:.2%} |
| **Recall** | {recall:.2%} |
| **F1-Score** | {f1_score:.2%} |

### Confusion Matrix

|               | Predicted: FALL | Predicted: NON-FALL |
|---------------|-----------------|---------------------|
| **Actual: FALL** | {tp} (TP) | {fn} (FN) |
| **Actual: NON-FALL** | {fp} (FP) | {tn} (TN) |

---

## Detailed Results

### Ground Truth: FALL Videos ({tp + fn} total)

#### ✓ Correctly Detected Falls (True Positives: {tp})

"""
    
    fall_tp = [r for r in results if r.get('ground_truth') == 'FALL' and r.get('classification') == 'TP']
    if fall_tp:
        for i, result in enumerate(fall_tp, 1):
            report += f"{i}. {result.get('relative_path', 'Unknown')}\n"
            report += f"   - Detection Frame: {result.get('fall_frame', 'N/A')}\n"
            report += f"   - Total Frames: {result.get('total_frames', 'N/A')}\n"
            report += f"   - Status: DETECTED\n\n"
    else:
        report += "No true positives\n\n"
    
    report += f"Missed Falls (False Negatives: {fn})\n\n"
    
    # Add FN details
    fall_fn = [r for r in results if r.get('ground_truth') == 'FALL' and r.get('classification') == 'FN']
    if fall_fn:
        for i, result in enumerate(fall_fn, 1):
            report += f"{i}. {result.get('relative_path', 'Unknown')}\n"
            report += f"   - Total Frames: {result.get('total_frames', 'N/A')}\n"
            report += f"   - Status: MISSED\n\n"
    else:
        report += "No false negatives\n\n"
    
    report += f"""---

### Ground Truth: NON-FALL Videos ({tn + fp} total)

#### True Negatives: {tn})

"""
    
    non_fall_tn = [r for r in results if r.get('ground_truth') == 'NON_FALL' and r.get('classification') == 'TN']
    if non_fall_tn:
        for i, result in enumerate(non_fall_tn, 1):
            report += f"{i}. {result.get('relative_path', 'Unknown')}\n"
            report += f"   - Total Frames: {result.get('total_frames', 'N/A')}\n"
            report += f"   - Status: CORRECTLY IGNORED\n\n"
    else:
        report += "No true negatives\n\n"
    
    report += f"False Alarms (False Positives: {fp})\n\n"
    
    non_fall_fp = [r for r in results if r.get('ground_truth') == 'NON_FALL' and r.get('classification') == 'FP']
    if non_fall_fp:
        for i, result in enumerate(non_fall_fp, 1):
            report += f"{i}. {result.get('relative_path', 'Unknown')}\n"
            report += f"   - Detection Frame: {result.get('fall_frame', 'N/A')}\n"
            report += f"   - Total Frames: {result.get('total_frames', 'N/A')}\n"
            report += f"   - Status: ❌ FALSE POSITIVE\n\n"
    else:
        report += "No false positives\n\n"
    
    report += f"""---

## Algorithm Configuration

### Zone Definitions

- BED_ZONE: {BED_ZONE.tolist()}
- FLOOR_ZONE: {FLOOR_ZONE.tolist()}

### Detection Parameters

- Model: YOLOv8n-Pose
- Minimum Keypoint Confidence: {MIN_CONFIDENCE}
- Velocity Threshold: {VELOCITY_THRESHOLD} pixels/frame
- Assumed FPS: {ASSUMED_FPS}

### Keypoints Used

- Left Shoulder
- Right Shoulder
- Left Hip
- Right Hip

---

## Interpretation

### Accuracy ({accuracy:.2%})
The percentage of correct predictions (both falls and non-falls) out of all videos tested.

### Precision ({precision:.2%})
Of all videos where the system predicted a fall, {precision:.2%} were actual falls. Higher precision means fewer false alarms.

### Recall ({recall:.2%})
Of all actual fall videos, the system detected {recall:.2%} of them. Higher recall means fewer missed falls.

### Recommendations

""" 
    report += """
---
"""
    
    report_path = Path("evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n Report saved to: {report_path}")
    json_path = Path("evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'tn': tn
            },
            'results': results
        }, f, indent=2)




def main():
    """NOTE: 메인 실행 함수."""    
    try:
        run_evaluation()
        print("\n\n Evaluation completed successfully.")
    except KeyboardInterrupt:
        print("\n\n Process interrupted by user")
    except Exception as e:
        print(f"\n\n Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
