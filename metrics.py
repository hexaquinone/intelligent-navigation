from datetime import datetime
from typing import Dict, Any, Optional, List, Union

metrics: Dict[str, Any] = {
    "total_events": 0,
    "hazards_detected": 0,
    "sensor_failures": 0,
    "high_risk_events": 0,
    "warnings_count": 0,
    "brake_events": 0,
    "trip_distance_km": 0.0,
    "confidence_samples": [],
    "event_history": []
}


def reset_metrics():
    """Resets all trip performance metrics and history to initial state."""
    global metrics
    metrics["total_events"] = 0
    metrics["hazards_detected"] = 0
    metrics["sensor_failures"] = 0
    metrics["high_risk_events"] = 0
    metrics["warnings_count"] = 0
    metrics["brake_events"] = 0
    metrics["trip_distance_km"] = 0.0
    metrics["confidence_samples"] = []
    metrics["event_history"] = []


def record_event(
    event: Union[Dict[str, Any], Any],
    risk: Optional[str] = None,
    action: Optional[str] = None,
    speed_kmh: float = 40.0,
    dt_seconds: float = 3.0,
    reason: Optional[str] = None,
    distance_m: Optional[float] = None,
    position: Optional[str] = None
):
    """
    Records a telemetry event and updates cumulative trip metrics:
    - Increments event counters
    - Tracks hazards vs clear events
    - Categorizes warning vs braking interventions
    - Computes accumulated trip distance (km)
    - Logs confidence samples and rolling audit history
    """
    metrics["total_events"] += 1

    # Extract event attributes whether event is dict or object
    event_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else "unknown")
    if hasattr(event_type, "value"):
        event_type = event_type.value
    event_type = str(event_type).lower()

    sensor_status = getattr(event, "sensor_status", None) or (event.get("sensor_status") if isinstance(event, dict) else "active")
    if hasattr(sensor_status, "value"):
        sensor_status = sensor_status.value
    sensor_status = str(sensor_status).lower()

    confidence = getattr(event, "confidence", None)
    if confidence is None and isinstance(event, dict):
        confidence = event.get("confidence", 1.0)
    if confidence is not None:
        try:
            metrics["confidence_samples"].append(float(confidence))
        except (ValueError, TypeError):
            pass

    # Extract spatial attributes if not provided
    if distance_m is None:
        distance_m = getattr(event, "distance", None) or (event.get("distance") if isinstance(event, dict) else None)
    if position is None:
        pos_val = getattr(event, "position", None) or (event.get("position") if isinstance(event, dict) else "center")
        position = getattr(pos_val, "value", str(pos_val)).lower()

    # Hazard classification
    if event_type not in ["clear", "sensor_gap", "unknown"] and "clear" not in event_type:
        metrics["hazards_detected"] += 1

    # Sensor failures
    if sensor_status == "failed" or event_type in ["sensor_failure", "sensor_gap"]:
        metrics["sensor_failures"] += 1

    # Risk level tracking
    if risk in ["HIGH", "CRITICAL"]:
        metrics["high_risk_events"] += 1

    # Action intervention tracking
    if action:
        act_upper = action.upper()
        if act_upper in ["SLOW_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "MAINTAIN_SPEED"]:
            metrics["warnings_count"] += 1
        elif act_upper in ["BRAKE", "STOP", "EMERGENCY_STOP"]:
            metrics["brake_events"] += 1

    # Incremental distance calculation: d = speed (km/h) * (dt / 3600 h)
    dist_increment = (speed_kmh * (dt_seconds / 3600.0))
    metrics["trip_distance_km"] = round(metrics["trip_distance_km"] + dist_increment, 2)

    # Append to rolling audit history (max 100 entries)
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "event_type": event_type.replace("_", " ").title(),
        "position": position.upper() if position else "FRONT",
        "distance_m": f"{distance_m:.1f} m" if distance_m is not None else "--",
        "speed_kmh": f"{speed_kmh:.0f} km/h",
        "risk": risk or "LOW",
        "action": action.replace("_", " ").upper() if action else "CONTINUE",
        "reason": reason or "Nominal road conditions."
    }
    metrics["event_history"].append(entry)
    if len(metrics["event_history"]) > 100:
        metrics["event_history"].pop(0)


def calculate_safety_score() -> int:
    """Computes a dynamic 0-100% Safety Score based on trip safety interventions."""
    total = metrics["total_events"]
    if total == 0:
        return 100

    score = 100.0
    # Penalty for critical/high risk events that required hard emergency braking
    score -= (metrics["high_risk_events"] * 4.0)
    score -= (metrics["brake_events"] * 2.5)
    score -= (metrics["sensor_failures"] * 5.0)

    # Reward for safe distance accumulated
    score += min(15.0, metrics["trip_distance_km"] * 2.0)

    return max(10, min(100, int(round(score))))


def get_metrics() -> Dict[str, Any]:
    """Returns current metrics snapshot with computed averages and safety score."""
    conf_samples = metrics["confidence_samples"]
    avg_confidence = round(sum(conf_samples) / len(conf_samples), 2) if conf_samples else 1.0

    return {
        "total_events": metrics["total_events"],
        "hazards_detected": metrics["hazards_detected"],
        "sensor_failures": metrics["sensor_failures"],
        "high_risk_events": metrics["high_risk_events"],
        "warnings_count": metrics["warnings_count"],
        "brake_events": metrics["brake_events"],
        "trip_distance_km": round(metrics["trip_distance_km"], 2),
        "average_confidence": avg_confidence,
        "safety_score": calculate_safety_score()
    }


def get_event_history() -> List[Dict[str, Any]]:
    """Returns the chronological audit log of recorded events."""
    return list(metrics.get("event_history", []))



if __name__ == "__main__":
    from simulation import get_event

    reset_metrics()
    for i in range(6):
        ev = get_event(i)
        record_event(ev, risk="MEDIUM", action="SLOW_DOWN", speed_kmh=40.0, dt_seconds=3.0)

    print("Sample Metrics Snapshot:")
    print(get_metrics())

