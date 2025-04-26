#this
from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from typing import Tuple, List, Dict, Set
import math


# YOLO-format bounding box (normalized)
x_center = 0.469531
y_center = 0.497685
bbox_width = 0.422396
bbox_height = 0.993519
model_class = YOLO("brand_best.pt")

class PalletTracker:
    def __init__(self, 
                 model_path: str,
                 max_age: int = 120,
                 n_init: int = 1,
                 conf_threshold: float = 0.3,
                 max_inactive_frames: int = 60):
        self.model = YOLO(model_path)
        self.deep_sort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.5,
            nn_budget=100,
            nms_max_overlap=0.5,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        self.conf_threshold = conf_threshold
        self.max_inactive_frames = max_inactive_frames
        self.unique_object_ids: Set[int] = set()
        self.last_seen_frame: Dict[int, int] = {}
        self.recent_tracks: Dict[int, List[int]] = {}

    @staticmethod
    def iou(box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = ((box1[2] - box1[0]) * (box1[3] - box1[1]) + 
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - 
                intersection)
        return intersection / union if union > 0 else 0

    @staticmethod
    def is_red(image: np.ndarray) -> str:
        """Determine if an image region contains red color."""
        RED_RANGES = [
            ((0, 120, 70), (10, 255, 255)),
            ((170, 120, 70), (180, 255, 255))
        ]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)
        
        for lower, upper in RED_RANGES:
            red_mask += cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        red_ratio = np.count_nonzero(red_mask) / (image.shape[0] * image.shape[1])
        return "red" if red_ratio > 0.1 else "not red"

    
    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Enhance image quality for better detection."""
        # Apply contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced


    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, int]:
        """Process a single frame and return the annotated frame and object count."""
        #frame = self.enhance_image(frame)
        h, w, _ = frame.shape
        # Convert normalized to pixel coordinates
        cx = int(x_center * w)
        cy = int(y_center * h)
        bw = int(bbox_width * w)
        bh = int(bbox_height * h)

        x1 = max(cx - bw // 2, 0)
        y1 = max(cy - bh // 2, 0)
        x2 = min(cx + bw // 2, w)
        y2 = min(cy + bh // 2, h)

        cropped_frame = frame[y1:y2, x1:x2]
        cv2.imshow("Cropped", cropped_frame)
        cv2.waitKey(1)  # Wait 1 ms so the window can update

        # Run inference on the cropped region
        results = self.model.track(source=cropped_frame, imgsz=960, conf=self.conf_threshold,
                                iou=0.4, save=False, persist=True)
        detections = self._get_detections(cropped_frame, results)
        detections.sort(key=lambda x: x[1], reverse=True)

        results_brand = model_class.track(source=cropped_frame, imgsz=1280, conf=self.conf_threshold,
                                iou=0.4, save=False, persist=True)
        detections_brand = self._get_detections(cropped_frame, results_brand)
        detections_brand.sort(key=lambda x: x[1], reverse=True)

        filtered_detections = self._filter_detections(detections)
        filtered_detections_brand = self._filter_detections(detections_brand)
        class_id = 0
        # Convert detections to absolute coordinates in original frame
        absolute_detections = []
        for (box, conf, color_status) in filtered_detections:
            x, y, w_box, h_box = box
            abs_box = [x + x1, y + y1, w_box, h_box]
            absolute_detections.append((abs_box, conf, color_status))

        absolute_brand_detections = []
        for (box, conf, color_status) in filtered_detections_brand:
            x, y, w_box, h_box = box
            abs_box = [x + x1, y + y1, w_box, h_box]
            absolute_brand_detections.append((abs_box, conf,model_class.names[class_id]))

        # Update DeepSORT with full-frame-sized detections
        tracks = self.deep_sort.update_tracks(
            [(d[0], d[1]) for d in absolute_detections],
            frame=frame
        )

        # Process tracks and draw annotations
        frame = self._process_tracks(frame, tracks, absolute_detections, frame_number)

        return frame, len(self.unique_object_ids)


    
    def _get_detections(self, frame: np.ndarray, results) -> List:
        """Extract detections from YOLO results."""
        detections = []
        for frame_result in results:
            for obj in frame_result.boxes:
                obj_conf = obj.conf.tolist()
                xyxy = obj.xyxy.tolist()
                
                for i in range(len(xyxy)):
                    if obj_conf[i] < self.conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    region = frame[y1:y2, x1:x2]
                    color_status = self.is_red(region)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], 
                                    obj_conf[i], color_status))
        return detections

    def _filter_detections(self, detections: List) -> List:
        """Filter detections based on IOU threshold."""
        filtered_detections = []
        drawn_boxes = []
        
        for detection in detections:
            x1, y1, w, h = detection[0]
            current_box = [x1, y1, x1 + w, y1 + h]
            
            if not any(self.iou(current_box, drawn_box) > 0.4
                      for drawn_box in drawn_boxes):
                drawn_boxes.append(current_box)
                filtered_detections.append(detection)
                
        return filtered_detections

    def _process_tracks(self, frame: np.ndarray, tracks, 
                       filtered_detections: List, frame_number: int) -> np.ndarray:
        """Process tracks and draw annotations on frame."""
        for track, (_, conf_value, color_status) in zip(tracks, filtered_detections):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Handle track persistence
            track_id = self._handle_track_persistence(track_id, [x1, y1, x2, y2])
            
            # Draw annotations
            frame = self._draw_annotations(frame, x1, y1, x2, y2, 
                                        track_id, (color_status if color_status is not None else "") , conf_value)
            
            # Update tracking info
            self.last_seen_frame[track_id] = frame_number
            self.unique_object_ids.add(track_id)
        
        return frame

    
    def _handle_track_persistence(self, track_id: int, bbox: List[int]) -> int:
        """Track ID persistence using Euclidean distance tracking."""
        # Calculate current object's center coordinates
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        DISTANCE_THRESHOLD = 70
        NEW_PLT_Y_DIFF = 10  # Check y-axis to avoid same ID for pallets behind others
        
        min_distance = float('inf')
        matched_id = track_id  # Default to same track_id
        
        # Compare with previously detected objects
        for prev_id, (px, py) in self.recent_tracks.items():
            # Calculate Manhattan distance
            distance = abs(cx - px) + abs(cy - py)
            
            # Skip if y-diff is too large (likely different pallets)
            if abs(cy - py) > NEW_PLT_Y_DIFF:
                continue
                
            # If object is within threshold distance, consider it the same object
            if distance < DISTANCE_THRESHOLD and distance < min_distance:
                min_distance = distance
                matched_id = prev_id  # Match to closest object
        
        # Update tracking information
        self.recent_tracks[matched_id] = (cx, cy)
        return matched_id


    def _draw_annotations(self, frame: np.ndarray, x1: int, y1: int, 
                         x2: int, y2: int, track_id: int, 
                         color_status: str, conf_value: float) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (f"{color_status} | Conf: {conf_value:.2f}")
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

def main():
    # Initialize video capture and writer
    video_path = "/Users/egeardaozturk/savola_label/Loading_Area/Doc 2/test_white-green-1.mp4"
    model_path = "/Users/egeardaozturk/savola_label/plt_count_clr/best_model.pt"
    output_path = "output.mp4"
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (frame_width, frame_height))
    
    # Initialize tracker
    tracker = PalletTracker(model_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        processed_frame, total_unique_objects = tracker.process_frame(
            frame, frame_number
        )
        
        # Add count to frame
        cv2.putText(processed_frame, 
                   f'Total Pallet Count: {total_unique_objects}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        out.write(processed_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()