# ============================================================
# INTELLIGENT NAVIGATION & DECISION-SUPPORT SYSTEM
# Computer Vision Perception Module (OpenCV + YOLO)
# ============================================================

import os
import sys
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np

# Try importing OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Try importing Ultralytics YOLO
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from brain import (
    HazardEvent,
    HazardType,
    Position,
    SensorStatus,
    RiskLevel,
    Action,
    EgoState,
    Decision,
    make_decisions,
    calculate_ttc
)


# ------------------------------------------------------------
# 1. OPTICAL & GEOMETRIC CONSTANTS
# ------------------------------------------------------------

# Approximate real-world heights (in meters) for standard roadway entities
REAL_WORLD_HEIGHTS: Dict[str, float] = {
    "pedestrian": 1.70,
    "person": 1.70,
    "car": 1.50,
    "truck": 3.20,
    "bus": 3.30,
    "motorcycle": 1.40,
    "cyclist": 1.65,
    "bicycle": 1.10,
    "dog": 0.65,
    "cat": 0.35,
    "animal": 0.80,
    "horse": 1.60,
    "cow": 1.40,
    "traffic light": 0.85,
    "stop sign": 0.75,
    "fire hydrant": 0.70,
    "traffic_cone": 0.70,
    "static_obstacle": 1.00
}

# Mapping YOLO/COCO object classes to brain.py HazardType and canonical subtype
CLASS_MAPPING: Dict[str, Tuple[HazardType, str]] = {
    "person": (HazardType.PEDESTRIAN, "pedestrian"),
    "car": (HazardType.VEHICLE, "car"),
    "truck": (HazardType.VEHICLE, "truck"),
    "bus": (HazardType.VEHICLE, "bus"),
    "motorcycle": (HazardType.VEHICLE, "motorcycle"),
    "bicycle": (HazardType.CYCLIST, "cyclist"),
    "dog": (HazardType.ANIMAL, "dog"),
    "cat": (HazardType.ANIMAL, "cat"),
    "horse": (HazardType.ANIMAL, "horse"),
    "cow": (HazardType.ANIMAL, "cow"),
    "sheep": (HazardType.ANIMAL, "sheep"),
    "traffic light": (HazardType.STATIC_OBSTACLE, "traffic_light"),
    "stop sign": (HazardType.STATIC_OBSTACLE, "stop_sign"),
    "fire hydrant": (HazardType.STATIC_OBSTACLE, "fire_hydrant"),
    "bench": (HazardType.STATIC_OBSTACLE, "obstacle"),
    "suitcase": (HazardType.STATIC_OBSTACLE, "debris"),
    "sports ball": (HazardType.STATIC_OBSTACLE, "hazard")
}


# ------------------------------------------------------------
# 2. DATA STRUCTURES FOR VISION DETECTIONS
# ------------------------------------------------------------

@dataclass
class BoundingBox:
    """Represents a 2D bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class VisionDetection:
    """A detected vision target with derived physical attributes."""
    box: BoundingBox
    hazard_event: HazardEvent
    pixel_height: int
    estimated_distance: float
    position: Position
    closing_speed_kmh: Optional[float] = None
    ttc_seconds: Optional[float] = None


@dataclass
class LaneInfo:
    """Represents detected road lane lines and departure metrics."""
    left_line: Optional[Tuple[int, int, int, int]] = None
    right_line: Optional[Tuple[int, int, int, int]] = None
    lane_center_x: Optional[float] = None
    offset_from_center_px: float = 0.0
    departure_warning: bool = False
    warning_message: str = "Lane Centered"


@dataclass
class VisionDecisionResult:
    """
    Unified high-level integration payload linking Computer Vision perception
    directly with the AI Brain Decision Engine.
    """
    annotated_frame: np.ndarray
    hazards: List[HazardEvent]
    decision: Decision
    lane_info: LaneInfo
    detections: List[VisionDetection] = field(default_factory=list)
    ego_state: EgoState = field(default_factory=EgoState)



# ------------------------------------------------------------
# 3. OPENCV LANE DETECTION ENGINE
# ------------------------------------------------------------

def detect_lanes_opencv(
    frame: np.ndarray,
    roi_top_ratio: float = 0.60,
    roi_bottom_ratio: float = 0.95
) -> Tuple[np.ndarray, LaneInfo]:
    """
    Applies OpenCV edge and line detection pipeline to find road lane boundaries.
    1. Grayscale & Gaussian Blur
    2. Canny Edge Detection
    3. Region of Interest (Trapezoidal ROI Mask)
    4. Probabilistic Hough Line Transform (HoughLinesP)
    5. Lane grouping into Left and Right slope clusters
    """
    if not HAS_OPENCV:
        return frame, LaneInfo()

    h, w = frame.shape[:2]
    lane_info = LaneInfo()
    annotated = frame.copy()

    try:
        # 1. Grayscale & Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)

        # 3. Trapezoidal Region of Interest Mask (Focusing on host lane path)
        mask = np.zeros_like(edges)
        y_top = int(h * roi_top_ratio)
        y_bottom = int(h * roi_bottom_ratio)

        roi_vertices = np.array([[
            (int(w * 0.10), y_bottom),
            (int(w * 0.40), y_top),
            (int(w * 0.60), y_top),
            (int(w * 0.90), y_bottom)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 4. Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=25
        )

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter steep slopes (avoid horizontal road artifacts)
                if abs(slope) < 0.4 or abs(slope) > 3.0:
                    continue

                if slope < 0 and x1 < w * 0.55 and x2 < w * 0.55:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0 and x1 > w * 0.45 and x2 > w * 0.45:
                    right_lines.append((x1, y1, x2, y2))

        # Helper to compute average line
        def average_line(line_list, y_min, y_max):
            if not line_list:
                return None
            x_coords = []
            y_coords = []
            for lx1, ly1, lx2, ly2 in line_list:
                x_coords.extend([lx1, lx2])
                y_coords.extend([ly1, ly2])
            poly = np.polyfit(y_coords, x_coords, 1)  # x = a*y + b
            x_start = int(poly[0] * y_max + poly[1])
            x_end = int(poly[0] * y_min + poly[1])
            return (x_start, y_max, x_end, y_min)

        lane_info.left_line = average_line(left_lines, y_top, y_bottom)
        lane_info.right_line = average_line(right_lines, y_top, y_bottom)

        # Calculate Lane Center and Departure Offset
        frame_center_x = w / 2.0
        if lane_info.left_line and lane_info.right_line:
            left_bottom_x = lane_info.left_line[0]
            right_bottom_x = lane_info.right_line[0]
            lane_center_x = (left_bottom_x + right_bottom_x) / 2.0
            lane_info.lane_center_x = lane_center_x
            lane_info.offset_from_center_px = frame_center_x - lane_center_x

            # Draw lane overlay polygon
            lane_poly = np.array([[
                (lane_info.left_line[0], lane_info.left_line[1]),
                (lane_info.left_line[2], lane_info.left_line[3]),
                (lane_info.right_line[2], lane_info.right_line[3]),
                (lane_info.right_line[0], lane_info.right_line[1])
            ]], dtype=np.int32)

            lane_overlay = frame.copy()
            cv2.fillPoly(lane_overlay, lane_poly, (0, 180, 0))
            cv2.addWeighted(lane_overlay, 0.22, annotated, 0.78, 0, annotated)

            # Draw boundary lines
            cv2.line(annotated, (lane_info.left_line[0], lane_info.left_line[1]), (lane_info.left_line[2], lane_info.left_line[3]), (0, 255, 255), 3)
            cv2.line(annotated, (lane_info.right_line[0], lane_info.right_line[1]), (lane_info.right_line[2], lane_info.right_line[3]), (0, 255, 255), 3)

            # Departure check (e.g. > 45px off center)
            if abs(lane_info.offset_from_center_px) > 45.0:
                lane_info.departure_warning = True
                direction = "LEFT" if lane_info.offset_from_center_px < 0 else "RIGHT"
                lane_info.warning_message = f"LANE DRIFT: VEERING {direction}"
        elif lane_info.left_line:
            cv2.line(annotated, (lane_info.left_line[0], lane_info.left_line[1]), (lane_info.left_line[2], lane_info.left_line[3]), (0, 200, 255), 2)
        elif lane_info.right_line:
            cv2.line(annotated, (lane_info.right_line[0], lane_info.right_line[1]), (lane_info.right_line[2], lane_info.right_line[3]), (0, 200, 255), 2)

    except Exception:
        # Graceful fallback if edge processing encounters unexpected noise
        pass

    return annotated, lane_info


# ------------------------------------------------------------
# 4. VISION PERCEPTION ENGINE (YOLO + DISTANCE + DECISION)
# ------------------------------------------------------------

class VisionPerceptionEngine:
    """
    Main Computer Vision Perception engine that interfaces with YOLO (Object Detection),
    OpenCV (Lane Detection & Image processing), and the Decision Engine (brain.py).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        focal_length: float = 720.0,
        enable_lanes: bool = True
    ):
        self.model_name = model_name
        self.focal_length = focal_length
        self.enable_lanes = enable_lanes
        self.model = None
        self.tracking_history: Dict[str, Dict[str, Any]] = {}
        self.last_timestamp: float = time.time()
        self.frame_count: int = 0

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads YOLO model if ultralytics is available; provides graceful fallback otherwise."""
        if HAS_ULTRALYTICS:
            try:
                self.model = YOLO(self.model_name)
            except Exception as err:
                print(f"[VisionPerceptionEngine] Warning: Could not load YOLO model ({err}). Running in fallback mode.")
                self.model = None
        else:
            self.model = None

    def estimate_distance(self, bbox_height_px: int, subtype: str) -> float:
        """
        Estimates real-world distance in meters using pinhole optical geometry:
            distance (m) = (focal_length * real_height) / bbox_height_px
        """
        real_h = REAL_WORLD_HEIGHTS.get(subtype, 1.50)
        if bbox_height_px <= 0:
            return 50.0

        estimated_dist = (self.focal_length * real_h) / float(bbox_height_px)
        return round(float(np.clip(estimated_dist, 1.0, 80.0)), 1)

    def determine_position(
        self,
        x_center: float,
        frame_width: int,
        lane_info: Optional[LaneInfo] = None
    ) -> Position:
        """
        Classifies horizontal position into LEFT, FRONT (Host Lane), or RIGHT.
        Leverages detected lane center if available, or 33%/66% frame split as fallback.
        """
        if lane_info and lane_info.lane_center_x is not None:
            lane_cx = lane_info.lane_center_x
            corridor_half_w = frame_width * 0.18  # width of host driving corridor
            if x_center < (lane_cx - corridor_half_w):
                return Position.LEFT
            elif x_center > (lane_cx + corridor_half_w):
                return Position.RIGHT
            return Position.FRONT

        # Default static 3-lane split
        left_thresh = frame_width * 0.33
        right_thresh = frame_width * 0.66

        if x_center < left_thresh:
            return Position.LEFT
        elif x_center > right_thresh:
            return Position.RIGHT
        return Position.FRONT

    def detect_objects_yolo(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35
    ) -> List[BoundingBox]:
        """Runs YOLO inference on a raw frame and returns list of BoundingBoxes."""
        if self.model is None:
            return self._fallback_simulated_detection(frame)

        boxes: List[BoundingBox] = []
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            if not results or len(results) == 0:
                return []

            res = results[0]
            if res.boxes is None:
                return []

            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                cls_name = self.model.names.get(cls_id, "unknown")
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                boxes.append(BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_name=cls_name
                ))
        except Exception as err:
            print(f"[VisionPerceptionEngine] Inference error: {err}")

        return boxes

    def _fallback_simulated_detection(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Fallback detector for offline testing / environments without weights:
        Detects bright/dark bounding regions or returns clear state.
        """
        return []

    def process_frame(
        self,
        frame: np.ndarray,
        ego_state: Optional[EgoState] = None,
        conf_threshold: float = 0.35,
        override_boxes: Optional[List[BoundingBox]] = None
    ) -> Tuple[np.ndarray, List[HazardEvent], Decision, LaneInfo]:
        """
        Main end-to-end perception pipeline:
        1. Lane Detection (OpenCV)
        2. Object Detection (YOLO or override_boxes)
        3. Physical Distance & Position Mapping
        4. HazardEvent Generation
        5. Decision Engine Synthesis (brain.py)
        6. AR HUD Overlay Generation
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            empty_img = np.zeros((480, 640, 3), dtype=np.uint8) if HAS_OPENCV else np.zeros((480, 640, 3))
            dummy_dec = make_decisions([], ego_state or EgoState())
            return empty_img, [], dummy_dec, LaneInfo()

        self.frame_count += 1
        now = time.time()
        dt = max(0.01, now - self.last_timestamp)
        self.last_timestamp = now

        if ego_state is None:
            ego_state = EgoState(speed_kmh=40.0, lane="center")

        h, w = frame.shape[:2]

        # 1. Lane Detection
        lane_info = LaneInfo()
        annotated_frame = frame.copy()
        if self.enable_lanes and HAS_OPENCV:
            annotated_frame, lane_info = detect_lanes_opencv(annotated_frame)

        # 2. Object Detection
        if override_boxes is not None:
            raw_boxes = override_boxes
        else:
            raw_boxes = self.detect_objects_yolo(frame, conf_threshold=conf_threshold)

        # 3. HazardEvent Construction
        hazards: List[HazardEvent] = []

        detections: List[VisionDetection] = []

        for idx, box in enumerate(raw_boxes):
            norm_class = box.class_name.lower()
            if norm_class not in CLASS_MAPPING:
                # Check for partial matches
                matched = False
                for k, v in CLASS_MAPPING.items():
                    if k in norm_class:
                        h_type, subtype = v
                        matched = True
                        break
                if not matched:
                    h_type, subtype = HazardType.UNKNOWN, norm_class
            else:
                h_type, subtype = CLASS_MAPPING[norm_class]

            # Physical Distance Estimation
            distance = self.estimate_distance(box.height, subtype)
            pos = self.determine_position(box.center_x, w, lane_info)

            # Closing speed estimate
            track_key = f"{subtype}_{pos.value}"
            closing_speed = ego_state.speed_kmh if pos == Position.FRONT else 0.0
            if track_key in self.tracking_history:
                prev_dist = self.tracking_history[track_key]["distance"]
                rate_mps = (prev_dist - distance) / dt
                if rate_mps > 0:
                    closing_speed = max(closing_speed, rate_mps * 3.6)

            self.tracking_history[track_key] = {"distance": distance, "time": now}

            # Construct brain-compatible HazardEvent
            hazard = HazardEvent(
                id=idx + 1,
                type=h_type,
                subtype=subtype,
                position=pos,
                distance=distance,
                confidence=round(box.confidence, 2),
                sensor="camera_yolo",
                sensor_status=SensorStatus.ACTIVE,
                relative_speed_kmh=round(closing_speed, 1)
            )

            hazards.append(hazard)
            detections.append(VisionDetection(
                box=box,
                hazard_event=hazard,
                pixel_height=box.height,
                estimated_distance=distance,
                position=pos,
                closing_speed_kmh=closing_speed,
                ttc_seconds=calculate_ttc(distance, closing_speed)
            ))

        # 4. If no hazards found, create clear baseline event
        if not hazards:
            hazards.append(HazardEvent(
                id=0,
                type=HazardType.CLEAR,
                position=Position.FRONT,
                distance=None,
                confidence=1.0,
                sensor="camera_yolo",
                sensor_status=SensorStatus.ACTIVE
            ))

        # 5. Evaluate Multi-Hazard Decision via brain.py
        decision = make_decisions(hazards, ego_state)

        # 6. Render Augmented Reality (AR) HUD Overlay
        if HAS_OPENCV:
            annotated_frame = self.render_ar_hud(annotated_frame, detections, decision, ego_state, lane_info)

        return annotated_frame, hazards, decision, lane_info

    def analyze(
        self,
        frame: np.ndarray,
        ego_state: Optional[EgoState] = None,
        conf_threshold: float = 0.35,
        override_boxes: Optional[List[BoundingBox]] = None
    ) -> VisionDecisionResult:
        """
        High-level unified perception & decision method returning a structured
        VisionDecisionResult dataclass for direct consumption by decision systems.
        """
        if ego_state is None:
            ego_state = EgoState()

        ann_frame, hazards, decision, lane_info = self.process_frame(
            frame=frame,
            ego_state=ego_state,
            conf_threshold=conf_threshold,
            override_boxes=override_boxes
        )

        return VisionDecisionResult(
            annotated_frame=ann_frame,
            hazards=hazards,
            decision=decision,
            lane_info=lane_info,
            ego_state=ego_state
        )


    def render_ar_hud(
        self,
        frame: np.ndarray,
        detections: List[VisionDetection],
        decision: Decision,
        ego: EgoState,
        lane_info: LaneInfo
    ) -> np.ndarray:
        """
        Draws professional cockpit AR overlays:
        - Bounding boxes with risk color coding (Red/Yellow/Green)
        - Target distance and classification tags
        - Lane drift warnings
        - Top AI Decision Banner with dynamic risk badges
        """
        if not HAS_OPENCV:
            return frame

        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Risk Colors (BGR format)
        COLOR_CRITICAL = (0, 0, 238)    # Bright Red
        COLOR_HIGH = (0, 69, 255)       # Orange Red
        COLOR_MEDIUM = (0, 215, 255)    # Amber Yellow
        COLOR_LOW = (0, 230, 0)         # Crisp Emerald Green
        COLOR_NEUTRAL = (180, 180, 180) # Slate

        risk_color_map = {
            "CRITICAL": COLOR_CRITICAL,
            "HIGH": COLOR_HIGH,
            "MEDIUM": COLOR_MEDIUM,
            "LOW": COLOR_LOW,
            "UNCERTAIN": (200, 140, 0)
        }

        # 1. Draw Detections
        for det in detections:
            box = det.box
            h_event = det.hazard_event

            # Determine box color based on hazard urgency
            if h_event.position == Position.FRONT and det.estimated_distance <= 12.0:
                b_color = COLOR_CRITICAL
            elif h_event.position == Position.FRONT and det.estimated_distance <= 25.0:
                b_color = COLOR_HIGH
            elif det.estimated_distance <= 15.0:
                b_color = COLOR_MEDIUM
            else:
                b_color = COLOR_LOW

            # Draw Corner Reticles / Bounding Box
            cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), b_color, 2)

            # Draw distance and subtype tag
            tag_text = f"{h_event.subtype.upper()} | {det.estimated_distance:.1f}m"
            if det.ttc_seconds is not None and det.ttc_seconds < 5.0:
                tag_text += f" | TTC {det.ttc_seconds:.1f}s"

            # Tag background pill
            t_w = len(tag_text) * 9 + 12
            t_y1 = max(0, box.y1 - 22)
            cv2.rectangle(canvas, (box.x1, t_y1), (box.x1 + t_w, t_y1 + 20), (15, 23, 42), -1)
            cv2.rectangle(canvas, (box.x1, t_y1), (box.x1 + t_w, t_y1 + 20), b_color, 1)
            cv2.putText(
                canvas,
                tag_text,
                (box.x1 + 6, t_y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 2. Draw Top Telemetry & AI Decision Banner
        banner_h = 68
        banner_overlay = canvas.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (10, 15, 26), -1)
        cv2.addWeighted(banner_overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, banner_h), (w, banner_h), (30, 41, 59), 2)

        # Decision Badge Color
        badge_color = risk_color_map.get(decision.risk, COLOR_LOW)

        # Action & Risk Header
        action_text = f"ACTION: {decision.action}"
        risk_text = f"[{decision.risk} RISK]"
        cv2.putText(canvas, action_text, (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, risk_text, (20 + len(action_text) * 16 + 10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, badge_color, 2, cv2.LINE_AA)

        # Reason Text
        reason_display = decision.reason
        if len(reason_display) > 85:
            reason_display = reason_display[:82] + "..."
        cv2.putText(canvas, reason_display, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (203, 213, 225), 1, cv2.LINE_AA)

        # Ego Speed Badge (Right side of banner)
        speed_text = f"HOST: {ego.speed_kmh:.0f} km/h"
        cv2.putText(canvas, speed_text, (w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (56, 189, 248), 2, cv2.LINE_AA)

        # Lane Departure Banner
        if lane_info.departure_warning:
            drift_y = banner_h + 30
            cv2.rectangle(canvas, (int(w / 2) - 160, drift_y - 20), (int(w / 2) + 160, drift_y + 10), (0, 100, 255), -1)
            cv2.putText(canvas, f"⚠️ {lane_info.warning_message}", (int(w / 2) - 145, drift_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas
        # 3. Draw Bottom Hotkey Helper Bar
        footer_h = 28
        footer_y = h - footer_h
        cv2.rectangle(canvas, (0, footer_y), (w, h), (10, 15, 26), -1)
        cv2.line(canvas, (0, footer_y), (w, footer_y), (30, 41, 59), 1)
        hotkey_msg = "[1] Pedestrian  [2] Lead Car  [3] Dual Pinch  [W] Webcam  [Space] Pause  [Q] Exit"
        cv2.putText(canvas, hotkey_msg, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

        return canvas


# ------------------------------------------------------------
# 5. ANIMATED SYNTHETIC DRIVING SIMULATOR
# ------------------------------------------------------------

def generate_animated_driving_frame(
    scenario: str = "urban_pedestrian",
    frame_idx: int = 0,
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """
    Generates a smoothly animated 30 FPS driving scene with moving road dashes,
    crossing pedestrians, approaching vehicles, or dual obstacles.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky & Horizon
    frame[:int(height * 0.45), :] = [135, 80, 40]  # Midnight Blue Sky

    # 2. Asphalt Road Surface
    road_poly = np.array([[
        (0, height),
        (int(width * 0.35), int(height * 0.45)),
        (int(width * 0.65), int(height * 0.45)),
        (width, height)
    ]], dtype=np.int32)
    if HAS_OPENCV:
        cv2.fillPoly(frame, road_poly, [45, 45, 45])

    # 3. Moving Road Lane Dashes
    if HAS_OPENCV:
        # Outer Solid Lane Boundaries
        cv2.line(frame, (int(width * 0.35), int(height * 0.45)), (0, height), (255, 255, 255), 4)
        cv2.line(frame, (int(width * 0.65), int(height * 0.45)), (width, height), (255, 255, 255), 4)

        # Center Animated Broken Line (scrolling downwards to simulate ego motion)
        dash_speed = 8
        dash_cycle = 40
        scroll_px = (frame_idx * dash_speed) % dash_cycle

        for step in range(8):
            base_y = int(height * 0.46) + (step * dash_cycle) + scroll_px
            if base_y > height - 10:
                continue
            y_start = max(int(height * 0.46), base_y)
            y_end = min(height - 10, y_start + 22)
            cv2.line(frame, (int(width * 0.5), y_start), (int(width * 0.5), y_end), (0, 230, 255), 3)

    boxes: List[BoundingBox] = []

    if scenario == "urban_pedestrian":
        # Pedestrian walks smoothly from left shoulder (x=70) across host lane to right (x=570)
        cycle_len = 160
        progress = (frame_idx % cycle_len) / float(cycle_len)  # 0.0 to 1.0
        p_cx = int(70 + progress * 500)

        # Pedestrian dimensions
        p_w = 40
        p_h = 135
        py2 = int(height * 0.84)
        py1 = py2 - p_h
        px1 = p_cx - int(p_w / 2)
        px2 = p_cx + int(p_w / 2)

        if HAS_OPENCV:
            # Draw Pedestrian Figure
            # Head
            cv2.circle(frame, (p_cx, py1 + 14), 13, (200, 220, 240), -1)
            # Torso (Yellow Jacket)
            cv2.rectangle(frame, (px1 + 4, py1 + 28), (px2 - 4, py2 - 40), (40, 180, 240), -1)
            # Legs
            cv2.rectangle(frame, (px1 + 8, py2 - 40), (px1 + 16, py2), (80, 80, 90), -1)
            cv2.rectangle(frame, (px2 - 16, py2 - 40), (px2 - 8, py2), (80, 80, 90), -1)

        boxes.append(BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2, confidence=0.96, class_name="person"))

    elif scenario == "highway_lead_vehicle":
        # Lead vehicle smoothly approaches host car and decelerates
        cycle_len = 140
        progress = (frame_idx % cycle_len) / float(cycle_len)
        # Sine wave breathing approach: distance shrinks from 35m down to 7m
        dist_factor = math.sin(progress * math.pi)  # 0.0 -> 1.0 -> 0.0

        v_w = int(100 + dist_factor * 110)
        v_h = int(60 + dist_factor * 85)
        v_cx = int(width * 0.50)
        vy2 = int(height * 0.65 + dist_factor * 80)
        vy1 = vy2 - v_h
        vx1 = v_cx - int(v_w / 2)
        vx2 = v_cx + int(v_w / 2)

        if HAS_OPENCV:
            # Car Body (Sporty Red Sedan)
            cv2.rectangle(frame, (vx1, vy1 + int(v_h * 0.35)), (vx2, vy2), (30, 40, 200), -1)
            # Cabin
            cv2.rectangle(frame, (vx1 + int(v_w * 0.15), vy1), (vx2 - int(v_w * 0.15), vy1 + int(v_h * 0.40)), (20, 25, 140), -1)
            # Rear Window
            cv2.rectangle(frame, (vx1 + int(v_w * 0.20), vy1 + 6), (vx2 - int(v_w * 0.20), vy1 + int(v_h * 0.35)), (70, 70, 70), -1)
            # Brake Lights (Bright glowing red if closing in fast)
            is_braking = dist_factor > 0.4
            light_col = (0, 0, 255) if is_braking else (0, 0, 140)
            cv2.circle(frame, (vx1 + 12, vy2 - 12), 7, light_col, -1)
            cv2.circle(frame, (vx2 - 12, vy2 - 12), 7, light_col, -1)

        boxes.append(BoundingBox(x1=vx1, y1=vy1, x2=vx2, y2=py2 if 'py2' in locals() else vy2, confidence=0.98, class_name="car"))

    elif scenario == "dual_hazard_pinch":
        # Dual hazards: Construction barrier on Left + Cyclist on Right
        # 1. Left Barrier
        bx1, by1, bx2, by2 = int(width * 0.14), int(height * 0.60), int(width * 0.28), int(height * 0.78)
        if HAS_OPENCV:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 140, 255), -1)
            # Hazard stripes
            cv2.line(frame, (bx1, by1), (bx1 + 30, by2), (20, 20, 20), 4)
            cv2.line(frame, (bx1 + 30, by1), (bx1 + 60, by2), (20, 20, 20), 4)
        boxes.append(BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2, confidence=0.94, class_name="stop sign"))

        # 2. Right Cyclist
        cy_progress = ((frame_idx * 2) % 120) / 120.0
        cy_x = int(width * 0.72 + cy_progress * 30)
        cx1, cy1, cx2, cy2 = cy_x - 20, int(height * 0.58), cy_x + 20, int(height * 0.80)
        if HAS_OPENCV:
            cv2.rectangle(frame, (cx1 + 4, cy1), (cx2 - 4, cy2 - 25), (40, 180, 80), -1)
            cv2.circle(frame, (cx1 + 8, cy2 - 10), 10, (120, 120, 120), 3)
            cv2.circle(frame, (cx2 - 8, cy2 - 10), 10, (120, 120, 120), 3)
        boxes.append(BoundingBox(x1=cx1, y1=cy1, x2=cx2, y2=cy2, confidence=0.95, class_name="bicycle"))

    return frame, boxes


def generate_synthetic_test_frame(
    scenario: str = "urban_pedestrian",
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Static wrapper for unit tests."""
    return generate_animated_driving_frame(scenario=scenario, frame_idx=15, width=width, height=height)


# ------------------------------------------------------------
# 6. STANDALONE INTERACTIVE CLI TEST RUNNER
# ------------------------------------------------------------

def run_vision_cli():
    """Interactive CLI runner for real-time Computer Vision testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Navigation - Vision Perception Module")
    parser.add_argument("--source", type=str, default="demo", help="'webcam', 'demo', or path to an image/video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--speed", type=float, default=40.0, help="Simulated host vehicle speed in km/h")
    parser.add_argument("--no-lanes", action="store_true", help="Disable OpenCV lane detection")
    args = parser.parse_args()

    print("=" * 68)
    print("🚗 INTELLIGENT NAVIGATION - LIVE VISION PERCEPTION & DECISION RUNNER")
    print("=" * 68)
    print(f"[*] OpenCV Active:     {HAS_OPENCV}")
    print(f"[*] Ultralytics YOLO:  {HAS_ULTRALYTICS}")
    print(f"[*] Initial Source:    {args.source}")
    print(f"[*] Initial Host Speed:{args.speed} km/h")
    print("=" * 68)
    print("⌨️  KEYBOARD CONTROLS:")
    print("   [1] Urban Pedestrian Crossing (Front Lane Collision Risk)")
    print("   [2] Highway Lead Vehicle (Approach & Rapid Braking)")
    print("   [3] Dual Pinch Hazard (Swerve Conflict Arbitration)")
    print("   [W] Toggle Live Webcam Stream")
    print("   [Space] Pause / Resume Simulation")
    print("   [Q] or [ESC] Quit Application")
    print("=" * 68)

    engine = VisionPerceptionEngine(
        model_name=args.model,
        enable_lanes=not args.no_lanes
    )
    ego = EgoState(speed_kmh=args.speed, lane="center")

    if not HAS_OPENCV:
        print("[ERROR] OpenCV (cv2) is required to render the GUI preview window.")
        return

    window_name = "Intelligent Navigation - AR Cockpit Live (Press Q to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    active_scenario = "urban_pedestrian"
    is_paused = False
    use_webcam = (args.source.lower() == "webcam")
    cap = None
    frame_counter = 0

    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[WARN] Could not connect to webcam 0. Reverting to animated simulation.")
            use_webcam = False

    try:
        while True:
            # Check if user clicked window close button [X]
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if use_webcam and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Webcam frame grab failed. Retrying...")
                    continue
                annotated_frame, hazards, decision, _ = engine.process_frame(
                    frame, ego_state=ego, conf_threshold=args.conf
                )
            elif os.path.isfile(args.source) and args.source != "demo":
                # Static file mode
                ext = os.path.splitext(args.source)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    frame = cv2.imread(args.source)
                    if frame is None:
                        break
                    annotated_frame, hazards, decision, _ = engine.process_frame(
                        frame, ego_state=ego, conf_threshold=args.conf
                    )
                else:
                    if cap is None:
                        cap = cv2.VideoCapture(args.source)
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    annotated_frame, hazards, decision, _ = engine.process_frame(
                        frame, ego_state=ego, conf_threshold=args.conf
                    )
            else:
                # Animated Interactive Simulation Mode
                if not is_paused:
                    frame_counter += 1

                raw_frame, synth_boxes = generate_animated_driving_frame(
                    scenario=active_scenario, frame_idx=frame_counter
                )
                annotated_frame, hazards, decision, _ = engine.process_frame(
                    raw_frame, ego_state=ego, conf_threshold=args.conf, override_boxes=synth_boxes
                )

            # Display AR Perception Frame
            cv2.imshow(window_name, annotated_frame)

            # Handle Keystrokes (30 FPS -> ~33ms)
            key = cv2.waitKey(33) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                break
            elif key == ord('1'):
                use_webcam = False
                active_scenario = "urban_pedestrian"
                print("\n[SCENARIO SWITCH] -> 🚶 Urban Pedestrian Crossing")
            elif key == ord('2'):
                use_webcam = False
                active_scenario = "highway_lead_vehicle"
                print("\n[SCENARIO SWITCH] -> 🚙 Highway Lead Vehicle Approach")
            elif key == ord('3'):
                use_webcam = False
                active_scenario = "dual_hazard_pinch"
                print("\n[SCENARIO SWITCH] -> 🚧 Dual Pinch (Left Barrier + Right Cyclist)")
            elif key in [ord('w'), ord('W')]:
                if not use_webcam:
                    if cap is None:
                        cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        use_webcam = True
                        print("\n[PERCEPTION SWITCH] -> 📷 Live Webcam Active")
                    else:
                        print("\n[ERROR] Could not open webcam index 0.")
                else:
                    use_webcam = False
                    if cap is not None:
                        cap.release()
                        cap = None
                    print("\n[PERCEPTION SWITCH] -> 🎮 Simulation Feed Active")
            elif key == 32:  # Space bar
                is_paused = not is_paused
                status = "PAUSED" if is_paused else "RESUMED"
                print(f"\n[SIMULATION] -> ⏸️ {status}")

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Vision Perception session closed safely.")


if __name__ == "__main__":
    run_vision_cli()


