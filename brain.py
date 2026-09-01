# ==========================================
# INTELLIGENT NAVIGATION SYSTEM
# Decision Engine (Upgraded)
# ==========================================

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Union


# ------------------------------------------
# ENUMS & CONSTANTS
# ------------------------------------------

class HazardType(str, Enum):
    CLEAR = "clear"
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"
    STATIC_OBSTACLE = "static_obstacle"
    CYCLIST = "cyclist"
    ANIMAL = "animal"
    SENSOR_FAILURE = "sensor_failure"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: Any) -> "HazardType":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.UNKNOWN
        raw = value.value if hasattr(value, "value") else str(value)
        val = str(raw).lower().strip()
        for member in cls:
            if member.value == val or member.name.lower() == val:
                return member
        if "obstacle" in val:
            return cls.STATIC_OBSTACLE
        return cls.UNKNOWN

    # For backward-compatibility
    from_string = from_value


class Position(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    REAR = "rear"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: Any) -> "Position":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.UNKNOWN
        raw = value.value if hasattr(value, "value") else str(value)
        val = str(raw).lower().strip()
        for member in cls:
            if member.value == val or member.name.lower() == val:
                return member
        return cls.UNKNOWN

    from_string = from_value


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNCERTAIN = "UNCERTAIN"


class Action(str, Enum):
    CONTINUE = "CONTINUE"
    MAINTAIN_SPEED = "MAINTAIN_SPEED"
    SLOW_DOWN = "SLOW_DOWN"
    BRAKE = "BRAKE"
    STOP = "STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"


class SensorStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"

    @classmethod
    def from_value(cls, value: Any) -> "SensorStatus":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.ACTIVE
        raw = value.value if hasattr(value, "value") else str(value)
        val = str(raw).lower().strip()
        for member in cls:
            if member.value == val or member.name.lower() == val:
                return member
        return cls.ACTIVE

    from_string = from_value


# ------------------------------------------
# DATA MODELS
# ------------------------------------------

@dataclass
class EgoState:
    """Represents the host vehicle's current state."""
    speed_kmh: float = 40.0
    lane: str = "center"
    sensor_status: SensorStatus = SensorStatus.ACTIVE

    def __post_init__(self):
        if not isinstance(self.sensor_status, SensorStatus):
            self.sensor_status = SensorStatus.from_value(self.sensor_status)

    @property
    def speed_ms(self) -> float:
        return self.speed_kmh / 3.6


@dataclass
class HazardEvent:
    """Represents a detected environmental hazard or status event."""
    id: Optional[int] = None
    type: Union[HazardType, str] = HazardType.UNKNOWN
    subtype: Optional[str] = None
    position: Union[Position, str] = Position.FRONT
    distance: Optional[float] = None  # in meters
    confidence: float = 1.0  # 0.0 to 1.0
    sensor_status: Union[SensorStatus, str] = SensorStatus.ACTIVE
    sensor: str = "camera"
    relative_speed_kmh: Optional[float] = None  # positive if closing in

    def __post_init__(self):
        if not isinstance(self.type, HazardType):
            self.type = HazardType.from_value(self.type)
        if not isinstance(self.position, Position):
            self.position = Position.from_value(self.position)
        if not isinstance(self.sensor_status, SensorStatus):
            self.sensor_status = SensorStatus.from_value(self.sensor_status)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HazardEvent":
        """Factory to safely convert a dictionary into a HazardEvent."""
        return cls(
            id=data.get("id"),
            type=data.get("type", HazardType.UNKNOWN),
            subtype=data.get("subtype"),
            position=data.get("position", Position.FRONT),
            distance=float(data["distance"]) if data.get("distance") is not None else None,
            confidence=float(data.get("confidence", 1.0)),
            sensor_status=data.get("sensor_status", SensorStatus.ACTIVE),
            sensor=data.get("sensor", "camera"),
            relative_speed_kmh=float(data["relative_speed_kmh"]) if data.get("relative_speed_kmh") is not None else None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, HazardType) else str(self.type),
            "subtype": self.subtype,
            "position": self.position.value if isinstance(self.position, Position) else str(self.position),
            "distance": self.distance,
            "confidence": self.confidence,
            "sensor_status": self.sensor_status.value if isinstance(self.sensor_status, SensorStatus) else str(self.sensor_status),
            "sensor": self.sensor,
            "relative_speed_kmh": self.relative_speed_kmh
        }


@dataclass
class Decision:
    """Represents the output decision, risk assessment, and explanation."""
    risk: str
    action: str
    reason: str
    ttc_seconds: Optional[float] = None
    target_speed_kmh: Optional[float] = None
    priority_level: int = 1  # 1 (lowest) to 5 (highest/critical)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Provides backward-compatible dictionary output."""
        return {
            "risk": self.risk,
            "action": self.action,
            "reason": self.reason,
            "ttc_seconds": self.ttc_seconds,
            "target_speed_kmh": self.target_speed_kmh,
            "priority_level": self.priority_level,
            "metadata": self.metadata
        }

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]


# ------------------------------------------
# HELPER KINEMATIC & TTC FUNCTIONS
# ------------------------------------------

def calculate_ttc(distance: Optional[float], closing_speed_kmh: Optional[float]) -> Optional[float]:
    """
    Calculates Time-To-Collision (TTC) in seconds.
    TTC = distance (m) / closing_speed (m/s)
    """
    if distance is None or closing_speed_kmh is None or closing_speed_kmh <= 0:
        return None
    closing_speed_ms = closing_speed_kmh / 3.6
    if closing_speed_ms <= 0.1:
        return None
    return round(distance / closing_speed_ms, 2)


# ------------------------------------------
# SINGLE EVENT DECISION ENGINE
# ------------------------------------------

def make_decision(
    event: Union[Dict[str, Any], HazardEvent],
    ego_state: Optional[Union[Dict[str, Any], EgoState]] = None
) -> Decision:
    """
    Receives a hazard event (dict or HazardEvent) and evaluates:
    - risk: LOW | MEDIUM | HIGH | CRITICAL | UNCERTAIN
    - action: CONTINUE | SLOW_DOWN | BRAKE | STOP | EMERGENCY_STOP | MOVE_LEFT | MOVE_RIGHT
    - reason: Human-readable explainable rationale
    """
    # Normalize inputs
    hazard = HazardEvent.from_dict(event) if isinstance(event, dict) else event
    has_explicit_ego = ego_state is not None
    if ego_state is None:
        ego = EgoState()
    elif isinstance(ego_state, dict):
        ego = EgoState(
            speed_kmh=ego_state.get("speed_kmh", 40.0),
            lane=ego_state.get("lane", "center"),
            sensor_status=SensorStatus.from_value(ego_state.get("sensor_status", "active"))
        )
    else:
        ego = ego_state

    # --------------------------------------
    # 1. SENSOR FAILURE CHECK
    # --------------------------------------
    if hazard.sensor_status == SensorStatus.FAILED or hazard.type == HazardType.SENSOR_FAILURE:
        return Decision(
            risk=RiskLevel.UNCERTAIN.value,
            action=Action.SLOW_DOWN.value,
            reason=(
                "Sensor failure detected. Environmental awareness is compromised, "
                "so the vehicle must slow down and prepare for safe pull-over."
            ),
            priority_level=4,
            target_speed_kmh=max(10.0, ego.speed_kmh * 0.5)
        )

    # --------------------------------------
    # 2. CLEAR ROAD CHECK
    # --------------------------------------
    if hazard.type == HazardType.CLEAR:
        return Decision(
            risk=RiskLevel.LOW.value,
            action=Action.CONTINUE.value,
            reason="No hazards are currently detected. The vehicle can continue safely.",
            priority_level=1,
            target_speed_kmh=ego.speed_kmh
        )

    # --------------------------------------
    # 3. SENSOR DEGRADED / LOW CONFIDENCE
    # --------------------------------------
    if hazard.sensor_status == SensorStatus.DEGRADED and hazard.confidence < 0.4:
        return Decision(
            risk=RiskLevel.UNCERTAIN.value,
            action=Action.SLOW_DOWN.value,
            reason=(
                f"Sensors are degraded with very low confidence ({hazard.confidence * 100:.0f}%). "
                "Reducing speed to maintain defensive safety buffer."
            ),
            priority_level=3,
            target_speed_kmh=max(15.0, ego.speed_kmh * 0.6)
        )

    if hazard.confidence < 0.5:
        hazard_name = hazard.subtype if hazard.subtype else (hazard.type.value if isinstance(hazard.type, HazardType) else str(hazard.type))
        return Decision(
            risk=RiskLevel.UNCERTAIN.value,
            action=Action.SLOW_DOWN.value,
            reason=(
                f"The sensor detected a possible {hazard_name}, "
                f"but confidence is low ({hazard.confidence * 100:.0f}%). "
                "The vehicle should slow down until the situation is verified."
            ),
            priority_level=2,
            target_speed_kmh=max(20.0, ego.speed_kmh * 0.7)
        )

    # --------------------------------------
    # 4. KINEMATICS / TIME-TO-COLLISION (TTC)
    # --------------------------------------
    has_velocity_info = (hazard.relative_speed_kmh is not None)
    closing_speed = hazard.relative_speed_kmh if hazard.relative_speed_kmh is not None else None
    ttc = calculate_ttc(hazard.distance, closing_speed) if (hazard.position == Position.FRONT and has_velocity_info) else None

    distance = hazard.distance if hazard.distance is not None else 999.0
    hazard_type = hazard.type
    position = hazard.position

    # --------------------------------------
    # 5. FRONT HAZARD EVALUATION
    # --------------------------------------
    if position == Position.FRONT:
        # Dynamic TTC criteria (applicable when velocity is explicitly provided)
        is_critical_ttc = ttc is not None and ttc < 1.5
        is_high_risk_ttc = ttc is not None and ttc < 2.5

        if distance < 8.0 or is_critical_ttc:
            risk = RiskLevel.HIGH.value
            if hazard_type in [HazardType.PEDESTRIAN, HazardType.CYCLIST, HazardType.ANIMAL]:
                action = Action.STOP.value
                priority = 5
                target_speed = 0.0
            else:
                action = Action.BRAKE.value
                priority = 4
                target_speed = max(0.0, ego.speed_kmh * 0.3)

        elif distance <= 20.0 or is_high_risk_ttc:
            risk = RiskLevel.MEDIUM.value
            action = Action.SLOW_DOWN.value
            priority = 3
            target_speed = max(15.0, ego.speed_kmh * 0.6)

        else:
            risk = RiskLevel.LOW.value
            action = Action.CONTINUE.value
            priority = 1
            target_speed = ego.speed_kmh

    # --------------------------------------
    # 6. LATERAL HAZARDS (LEFT / RIGHT)
    # --------------------------------------
    elif position == Position.LEFT:
        if distance < 8.0:
            risk = RiskLevel.HIGH.value
            action = Action.SLOW_DOWN.value
            priority = 4
            target_speed = max(15.0, ego.speed_kmh * 0.5)
        else:
            risk = RiskLevel.MEDIUM.value
            action = Action.MOVE_RIGHT.value
            priority = 2
            target_speed = ego.speed_kmh

    elif position == Position.RIGHT:
        if distance < 8.0:
            risk = RiskLevel.HIGH.value
            action = Action.SLOW_DOWN.value
            priority = 4
            target_speed = max(15.0, ego.speed_kmh * 0.5)
        else:
            risk = RiskLevel.MEDIUM.value
            action = Action.MOVE_LEFT.value
            priority = 2
            target_speed = ego.speed_kmh

    # --------------------------------------
    # 7. REAR / UNKNOWN POSITION
    # --------------------------------------
    elif position == Position.REAR:
        if distance < 6.0:
            risk = RiskLevel.MEDIUM.value
            action = Action.MAINTAIN_SPEED.value
            priority = 2
            target_speed = ego.speed_kmh
        else:
            risk = RiskLevel.LOW.value
            action = Action.CONTINUE.value
            priority = 1
            target_speed = ego.speed_kmh
    else:
        risk = RiskLevel.UNCERTAIN.value
        action = Action.SLOW_DOWN.value
        priority = 2
        target_speed = max(20.0, ego.speed_kmh * 0.8)

    # --------------------------------------
    # 8. HUMAN-READABLE EXPLANATION
    # --------------------------------------
    hazard_label = hazard.subtype if hazard.subtype else (hazard_type.value if isinstance(hazard_type, HazardType) else str(hazard_type))
    ttc_phrase = f" (TTC: {ttc}s)" if ttc is not None else ""
    dist_phrase = f"{distance:.0f}m away" if distance < 900 and distance == int(distance) else (f"{distance:.1f}m away" if distance < 900 else "unknown distance")

    pos_str = position.value if isinstance(position, Position) else str(position)
    reason = (
        f"{hazard_label.replace('_', ' ').title()} detected {dist_phrase} "
        f"on the {pos_str}{ttc_phrase}. "
        f"Risk level is {risk}. "
        f"The recommended action is {action}."
    )

    return Decision(
        risk=risk,
        action=action,
        reason=reason,
        ttc_seconds=ttc,
        target_speed_kmh=target_speed,
        priority_level=priority,
        metadata={"hazard_id": hazard.id, "confidence": hazard.confidence, "sensor": hazard.sensor}
    )


# ------------------------------------------
# MULTI-HAZARD ARBITRATION ENGINE
# ------------------------------------------

def make_decisions(
    events: List[Union[Dict[str, Any], HazardEvent]],
    ego_state: Optional[Union[Dict[str, Any], EgoState]] = None
) -> Decision:
    """
    Arbitrates across multiple simultaneous hazards and vehicle surroundings:
    - Analyzes front, left, and right channels simultaneously.
    - Prevents unsafe lane swerves (e.g. avoids MOVE_RIGHT if right lane is blocked).
    - Selects the most critical safety-prioritized action.
    """
    if not events:
        return make_decision({"type": "clear", "sensor_status": "active"}, ego_state)

    # Normalize hazards
    hazards = [
        HazardEvent.from_dict(e) if isinstance(e, dict) else e
        for e in events
    ]

    # Evaluate individual decisions
    individual_decisions = [make_decision(h, ego_state) for h in hazards]

    # Map occupancy by position
    left_hazards = [h for h in hazards if h.position == Position.LEFT and (h.distance or 99) < 20]
    right_hazards = [h for h in hazards if h.position == Position.RIGHT and (h.distance or 99) < 20]

    # Sort decisions by priority (highest priority first)
    sorted_decisions = sorted(individual_decisions, key=lambda d: d.priority_level, reverse=True)
    primary_decision = sorted_decisions[0]

    # Check for Evasive Maneuver Conflicts (Swerve Clearance Matrix)
    if primary_decision.action == Action.MOVE_RIGHT.value and right_hazards:
        obstacle_right = right_hazards[0].subtype or (right_hazards[0].type.value if isinstance(right_hazards[0].type, HazardType) else str(right_hazards[0].type))
        return Decision(
            risk=RiskLevel.HIGH.value,
            action=Action.SLOW_DOWN.value,
            reason=(
                f"Obstacle on left detected, but right lane is also occupied by {obstacle_right}. "
                "Evasive lane change is unsafe; slowing down in-lane instead."
            ),
            priority_level=4,
            target_speed_kmh=20.0,
            metadata={"arbitration": "blocked_swerve_right", "hazards_count": len(hazards)}
        )

    if primary_decision.action == Action.MOVE_LEFT.value and left_hazards:
        obstacle_left = left_hazards[0].subtype or (left_hazards[0].type.value if isinstance(left_hazards[0].type, HazardType) else str(left_hazards[0].type))
        return Decision(
            risk=RiskLevel.HIGH.value,
            action=Action.SLOW_DOWN.value,
            reason=(
                f"Obstacle on right detected, but left lane is also occupied by {obstacle_left}. "
                "Evasive lane change is unsafe; slowing down in-lane instead."
            ),
            priority_level=4,
            target_speed_kmh=20.0,
            metadata={"arbitration": "blocked_swerve_left", "hazards_count": len(hazards)}
        )

    # Multi-hazard composite explanation if multiple hazards exist
    if len(hazards) > 1 and primary_decision.risk in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]:
        primary_decision.reason += f" (Arbitrated among {len(hazards)} active environmental hazards)."

    return primary_decision


# ==========================================
# TESTING ENTRY POINT
# ==========================================

if __name__ == "__main__":
    test_event = {
        "id": 1,
        "type": "pedestrian",
        "subtype": None,
        "position": "front",
        "distance": 6,
        "confidence": 0.98,
        "sensor_status": "active"
    }

    decision = make_decision(test_event)

    print("\n--- DECISION ---")
    print("Risk:", decision["risk"])
    print("Action:", decision["action"])
    print("Reason:", decision["reason"])
    if decision.ttc_seconds:
        print("TTC:", decision.ttc_seconds)