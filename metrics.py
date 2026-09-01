# ==========================================
# INTELLIGENT NAVIGATION SYSTEM
# Trip Metrics & Performance Tracker
# ==========================================

from typing import Dict, Any, Optional, List, Union

metrics: Dict[str, Any] = {
    "total_events": 0,
    "hazards_detected": 0,
    "sensor_failures": 0,
    "high_risk_events": 0,
    "warnings_count": 0,
    "brake_events": 0,
    "trip_distance_km": 0.0,
    "confidence_samples": []
}


def reset_metrics():
    """Resets all trip performance metrics to initial state."""
    global metrics
    metrics["total_events"] = 0
    metrics["hazards_detected"] = 0
    metrics["sensor_failures"] = 0
    metrics["high_risk_events"] = 0
    metrics["warnings_count"] = 0
    metrics["brake_events"] = 0
    metrics["trip_distance_km"] = 0.0
    metrics["confidence_samples"] = []


def record_event(
    event: Union[Dict[str, Any], Any],
    risk: Optional[str] = None,
    action: Optional[str] = None,
    speed_kmh: float = 40.0,
    dt_seconds: float = 3.0
):
    """
    Records a telemetry event and updates cumulative trip metrics:
    - Increments event counters
    - Tracks hazards vs clear events
    - Categorizes warning vs braking interventions
    - Computes accumulated trip distance (km)
    - Logs confidence samples
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


def get_metrics() -> Dict[str, Any]:
    """Returns current metrics snapshot with computed averages."""
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
        "average_confidence": avg_confidence
    }


if __name__ == "__main__":
    from simulation import get_event

    reset_metrics()
    for i in range(6):
        ev = get_event(i)
        record_event(ev, risk="MEDIUM", action="SLOW_DOWN", speed_kmh=40.0, dt_seconds=3.0)

    print("Sample Metrics Snapshot:")
    print(get_metrics())


# ==========================================
 # INTELLIGENT NAVIGATION SYSTEM 
 # Trip Metrics 
 # ========================================== 
metrics = { 
    "total_events": 0, 
    "hazards_detected": 0, 
    "sensor_failures": 0, 
    "high_risk_events": 0, 
    "warnings_count": 0, 
    "brake_events": 0, 
    } 
def record_event(event, risk=None, action=None):
    metrics["total_events"] += 1
    # Count hazards 
    if event["type"] not in ["clear", "sensor_failure", "sensor_gap"]: 
        metrics["hazards_detected"] += 1
    # Count sensor failures 
    if event.get("sensor_status") == "failed": 
        metrics["sensor_failures"] += 1  
    # Count high-risk decisions 
    if risk in ["HIGH", "CRITICAL"]: 
        metrics["high_risk_events"] += 1 
    # Count warnings / slow-down decisions 
    if action in ["SLOW_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]: 
         metrics["warnings_count"] += 1 
    # Count braking / stopping decisions 
    if action in ["BRAKE", "STOP", "EMERGENCY_STOP"]: 
        metrics["brake_events"] += 1 
    def get_metrics(): 
        return metrics 
    def reset_metrics(): 
        """Reset metrics before starting a new trip.""" 
        for key in metrics: 
            metrics[key] = 0 
    if __name__ == "__main__": 
        print(get_metrics())
