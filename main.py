from simulation import SimulationEngine
from metrics import record_event, get_metrics

def run_trip():

    print("=" * 60)
    print("INTELLIGENT NAVIGATION SYSTEM")
    print("=" * 60)

    results = SimulationEngine.run_trip_simulation()

    for step in results:

        print("\n" + "-" * 60)

        print(f"Timestep: {step['timestep']}")
        print(f"Description: {step['description']}")
        print(f"Ego Speed: {step['ego_speed_kmh']} km/h")

        # Show hazards
        for hazard in step["hazards"]:

            print(f"\nHazard: {hazard['type']}")
            print(f"Position: {hazard['position']}")
            print(f"Distance: {hazard['distance']}")
            print(f"Confidence: {hazard['confidence']}")
            print(f"Sensor: {hazard['sensor_status']}")

            # Send event to metrics
            record_event(
                hazard,
                risk=step["decision"]["risk"],
                action=step["decision"]["action"]
                )

        # Show AI decision
        decision = step["decision"]

        print("\nAI DECISION")
        print(f"Risk: {decision['risk']}")
        print(f"Action: {decision['action']}")
        print(f"Reason: {decision['reason']}")

    # Final metrics
    print("\n" + "=" * 60)
    print("TRIP METRICS")
    print("=" * 60)

    metrics = get_metrics()

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_trip()