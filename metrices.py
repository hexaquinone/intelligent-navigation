metrics = {
    "total_events": 0,
    "hazards_detected": 0,
    "sensor_failures": 0,
    "high_risk_events": 0
}


def record_event(event, risk=None):
    metrics["total_events"] += 1

    if event["type"] not in ["clear", "sensor_gap"]:
        metrics["hazards_detected"] += 1

    if event["sensor_status"] == "failed":
        metrics["sensor_failures"] += 1

    if risk == "HIGH":
        metrics["high_risk_events"] += 1


def get_metrics():
    return metrics

if __name__ == "__main__":
    from simulation import get_event

    for i in range(6):
        event = get_event(i)
        record_event(event)

    print(get_metrics())
    