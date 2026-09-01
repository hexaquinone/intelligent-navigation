# ==========================================
# INTELLIGENT NAVIGATION SYSTEM
# Sensor & Environment Simulator
# ==========================================

import time
from typing import List, Dict, Any, Optional, Generator
from brain import (
    HazardEvent,
    HazardType,
    Position,
    SensorStatus,
    EgoState,
    make_decision,
    make_decisions,
    Decision
)

# ------------------------------------------
# SENSOR PROFILES & CONFIG
# ------------------------------------------

SENSOR_PROFILES = {
    "camera": {
        "modalities": ["object_detection", "lane_detection", "sign_recognition"],
        "base_confidence": 0.96,
        "range_m": 80.0,
        "weather_sensitivity": "high"
    },
    "lidar": {
        "modalities": ["3d_pointcloud", "distance_ranging", "obstacle_mapping"],
        "base_confidence": 0.98,
        "range_m": 120.0,
        "weather_sensitivity": "low"
    },
    "radar": {
        "modalities": ["doppler_velocity", "distance_ranging"],
        "base_confidence": 0.92,
        "range_m": 150.0,
        "weather_sensitivity": "none"
    }
}


# ------------------------------------------
# PRESET SCENARIO CATALOG
# ------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "urban_pedestrian": {
        "title": "Urban Pedestrian Crossing",
        "description": "Pedestrian suddenly steps onto the roadway 6 meters ahead of host vehicle.",
        "ego_state": EgoState(speed_kmh=35.0, lane="center"),
        "events": [
            HazardEvent(
                id=101,
                type=HazardType.PEDESTRIAN,
                position=Position.FRONT,
                distance=6.0,
                confidence=0.98,
                sensor="camera",
                sensor_status=SensorStatus.ACTIVE,
                relative_speed_kmh=35.0
            )
        ]
    },
    "highway_lead_vehicle": {
        "title": "Highway Lead Vehicle Rapid Deceleration",
        "description": "Lead vehicle 22m ahead braking rapidly on high-speed highway (closing speed 45 km/h).",
        "ego_state": EgoState(speed_kmh=90.0, lane="center"),
        "events": [
            HazardEvent(
                id=102,
                type=HazardType.VEHICLE,
                position=Position.FRONT,
                distance=22.0,
                confidence=0.97,
                sensor="radar",
                sensor_status=SensorStatus.ACTIVE,
                relative_speed_kmh=45.0
            )
        ]
    },
    "dual_hazard_pinch": {
        "title": "Tight Pinch (Dual Surrounding Hazards)",
        "description": "Construction barrier on left (12m) AND cyclist on right (10m) — tests swerve safety matrix.",
        "ego_state": EgoState(speed_kmh=40.0, lane="center"),
        "events": [
            HazardEvent(
                id=103,
                type=HazardType.STATIC_OBSTACLE,
                subtype="construction_barrier",
                position=Position.LEFT,
                distance=12.0,
                confidence=0.95,
                sensor="lidar",
                sensor_status=SensorStatus.ACTIVE
            ),
            HazardEvent(
                id=104,
                type=HazardType.CYCLIST,
                subtype="cyclist",
                position=Position.RIGHT,
                distance=10.0,
                confidence=0.94,
                sensor="camera",
                sensor_status=SensorStatus.ACTIVE
            )
        ]
    },
    "adverse_weather": {
        "title": "Heavy Fog & Adverse Weather (Degraded Sensors)",
        "description": "Camera vision degraded due to severe fog with low detection confidence (35%).",
        "ego_state": EgoState(speed_kmh=50.0, lane="center"),
        "events": [
            HazardEvent(
                id=105,
                type=HazardType.STATIC_OBSTACLE,
                subtype="debris",
                position=Position.FRONT,
                distance=14.0,
                confidence=0.35,
                sensor="camera",
                sensor_status=SensorStatus.DEGRADED
            )
        ]
    },
    "sensor_hardware_failure": {
        "title": "Sudden Sensor Hardware Disconnect",
        "description": "Primary perception pipeline drops offline with hardware error.",
        "ego_state": EgoState(speed_kmh=60.0, lane="center"),
        "events": [
            HazardEvent(
                id=106,
                type=HazardType.SENSOR_FAILURE,
                position=Position.UNKNOWN,
                distance=None,
                confidence=0.0,
                sensor="camera",
                sensor_status=SensorStatus.FAILED
            )
        ]
    },
    "clear_road": {
        "title": "Open Highway (Clear Road)",
        "description": "No obstacles detected; nominal cruising conditions.",
        "ego_state": EgoState(speed_kmh=70.0, lane="center"),
        "events": [
            HazardEvent(
                id=107,
                type=HazardType.CLEAR,
                position=Position.FRONT,
                distance=None,
                confidence=1.0,
                sensor="camera",
                sensor_status=SensorStatus.ACTIVE
            )
        ]
    }
}


# ------------------------------------------
# SEQUENTIAL TRIP TIMELINE SIMULATION
# ------------------------------------------

TRIP_TIMELINE = [
    {
        "timestep": 1,
        "description": "Cruising down residential boulevard.",
        "ego_speed": 40.0,
        "events": [
            {"id": 1, "type": "clear", "position": "front", "distance": None, "confidence": 1.0, "sensor_status": "active"}
        ]
    },
    {
        "timestep": 2,
        "description": "Parked car door opened on left lane margin.",
        "ego_speed": 40.0,
        "events": [
            {"id": 2, "type": "static_obstacle", "subtype": "open car door", "position": "left", "distance": 14.0, "confidence": 0.92, "sensor_status": "active"}
        ]
    },
    {
        "timestep": 3,
        "description": "Child chasing ball enters front lane 7m away!",
        "ego_speed": 35.0,
        "events": [
            {"id": 3, "type": "pedestrian", "subtype": "child", "position": "front", "distance": 7.0, "confidence": 0.99, "sensor_status": "active", "relative_speed_kmh": 35.0}
        ]
    },
    {
        "timestep": 4,
        "description": "Child clears road; vehicle resumes travel under foggy drizzle.",
        "ego_speed": 20.0,
        "events": [
            {"id": 4, "type": "vehicle", "position": "front", "distance": 18.0, "confidence": 0.40, "sensor_status": "degraded", "relative_speed_kmh": 10.0}
        ]
    },
    {
        "timestep": 5,
        "description": "Highway entrance - clear high-speed merge.",
        "ego_speed": 65.0,
        "events": [
            {"id": 5, "type": "clear", "position": "front", "distance": None, "confidence": 1.0, "sensor_status": "active"}
        ]
    }
]


# ------------------------------------------
# SIMULATION ENGINE & RUNNER
# ------------------------------------------

class SimulationEngine:
    """Manages scenario execution and step-by-step telemetry generation."""

    @staticmethod
    def list_scenarios() -> List[str]:
        return list(SCENARIOS.keys())

    @staticmethod
    def get_scenario(key: str) -> Optional[Dict[str, Any]]:
        return SCENARIOS.get(key)

    @staticmethod
    def run_scenario(key: str) -> Dict[str, Any]:
        scenario = SCENARIOS.get(key)
        if not scenario:
            raise ValueError(f"Scenario '{key}' not found. Available: {list(SCENARIOS.keys())}")

        ego_state = scenario["ego_state"]
        events = scenario["events"]
        decision = make_decisions(events, ego_state)

        return {
            "key": key,
            "title": scenario["title"],
            "description": scenario["description"],
            "ego_speed_kmh": ego_state.speed_kmh,
            "events_count": len(events),
            "events": [e.to_dict() for e in events],
            "decision": decision.to_dict()
        }

    @staticmethod
    def run_trip_simulation(delay_sec: float = 0.0) -> List[Dict[str, Any]]:
        """Executes a chronological multi-step drive simulation."""
        results = []
        for step in TRIP_TIMELINE:
            ego = EgoState(speed_kmh=step["ego_speed"])
            raw_events = step["events"]
            events = [HazardEvent.from_dict(e) for e in raw_events]
            decision = make_decisions(events, ego)

            step_record = {
                "timestep": step["timestep"],
                "description": step["description"],
                "ego_speed_kmh": step["ego_speed"],
                "hazards": [e.to_dict() for e in events],
                "decision": decision.to_dict()
            }
            results.append(step_record)
            if delay_sec > 0:
                time.sleep(delay_sec)
        return results


# ------------------------------------------
# INTERACTIVE CLI MODE
# ------------------------------------------

def interactive_mode():
    """Interactive command-line hazard injector."""
    print("\n" + "=" * 55)
    print("🚗 INTELLIGENT NAVIGATION - SENSOR HAZARD INJECTOR")
    print("=" * 55)

    try:
        htype = input("Hazard Type (pedestrian/vehicle/obstacle/cyclist/clear/sensor_failure) [clear]: ").strip() or "clear"
        if htype == "clear":
            event = HazardEvent(type=HazardType.CLEAR, position=Position.FRONT)
        elif htype == "sensor_failure":
            event = HazardEvent(type=HazardType.SENSOR_FAILURE, position=Position.UNKNOWN, sensor_status=SensorStatus.FAILED)
        else:
            pos = input("Position (front/left/right/rear) [front]: ").strip() or "front"
            dist_str = input("Distance in meters (e.g. 7.5) [15.0]: ").strip() or "15.0"
            conf_str = input("Sensor Confidence 0.0-1.0 [0.95]: ").strip() or "0.95"
            speed_str = input("Host Vehicle Speed km/h [40.0]: ").strip() or "40.0"
            status_str = input("Sensor Status (active/degraded/failed) [active]: ").strip() or "active"

            event = HazardEvent(
                type=HazardType.from_string(htype),
                position=Position.from_string(pos),
                distance=float(dist_str),
                confidence=float(conf_str),
                sensor_status=SensorStatus.from_string(status_str)
            )
            ego = EgoState(speed_kmh=float(speed_str))
            decision = make_decision(event, ego)

            print("\n[RESULT]")
            print(f"Risk:   {decision.risk}")
            print(f"Action: {decision.action}")
            print(f"Reason: {decision.reason}")
            if decision.ttc_seconds:
                print(f"TTC:    {decision.ttc_seconds}s")
            return

        decision = make_decision(event)
        print("\n[RESULT]")
        print(f"Risk:   {decision.risk}")
        print(f"Action: {decision.action}")
        print(f"Reason: {decision.reason}")

    except Exception as e:
        print(f"Error processing input: {e}")


# ------------------------------------------
# CLI DEMONSTRATION
# ------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ALL PRESET SCENARIOS")
    print("=" * 60)

    for sc_key in SimulationEngine.list_scenarios():
        res = SimulationEngine.run_scenario(sc_key)
        print(f"\n[SCENARIO] {res['title']}")
        print(f"Description: {res['description']}")
        print(f"Ego Speed:   {res['ego_speed_kmh']} km/h")
        print(f"Decision:    ACTION={res['decision']['action']} | RISK={res['decision']['risk']}")
        print(f"Reason:      {res['decision']['reason']}")
        if res['decision']['ttc_seconds']:
            print(f"TTC:         {res['decision']['ttc_seconds']}s")
        print("-" * 60)
