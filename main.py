import sys
import argparse
from simulation import SimulationEngine
from metrics import record_event, get_metrics
from brain import process_vision_frame, EgoState

# Ensure UTF-8 output on Windows terminals
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass



def run_trip():
    print("=" * 60)
    print("🚗 INTELLIGENT NAVIGATION SYSTEM - TRIP SIMULATION")
    print("=" * 60)

    results = SimulationEngine.run_trip_simulation()

    for step in results:
        print("\n" + "-" * 60)
        print(f"Timestep:    {step['timestep']}")
        print(f"Description: {step['description']}")
        print(f"Ego Speed:   {step['ego_speed_kmh']} km/h")

        # Show hazards
        for hazard in step["hazards"]:
            print(f"\nHazard:      {hazard['type']} ({hazard.get('subtype', '--')})")
            print(f"Position:    {hazard['position']}")
            print(f"Distance:    {hazard['distance']}m")
            print(f"Confidence:  {hazard['confidence'] * 100:.0f}%")
            print(f"Sensor:      {hazard['sensor_status']}")

            # Send event to metrics
            record_event(
                hazard,
                risk=step["decision"]["risk"],
                action=step["decision"]["action"]
            )

        # Show AI decision
        decision = step["decision"]
        print("\n🧠 AI BRAIN DECISION:")
        print(f"Risk:   {decision['risk']}")
        print(f"Action: {decision['action']}")
        print(f"Reason: {decision['reason']}")

    # Final metrics
    print("\n" + "=" * 60)
    print("📊 TRIP SUMMARY & PERFORMANCE METRICS")
    print("=" * 60)

    metrics = get_metrics()
    for key, value in metrics.items():
        print(f"• {key.replace('_', ' ').title()}: {value}")


def run_vision(source: str = "demo"):
    """Launches the integrated Computer Vision Perception & Decision pipeline."""
    from vision import run_vision_cli
    sys.argv = ["vision.py", "--source", source]
    run_vision_cli()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Navigation & Decision-Support System")
    parser.add_argument("--mode", choices=["trip", "vision", "webcam"], default="trip", help="Run mode: 'trip' (sensor simulation) or 'vision' (YOLO + OpenCV perception)")
    args = parser.parse_args()

    if args.mode == "trip":
        run_trip()
    elif args.mode == "webcam":
        run_vision("webcam")
    else:
        run_vision("demo")