# ==========================================
# INTELLIGENT NAVIGATION SYSTEM
# Sensor & Environment Simulator
# ==========================================

import time
from typing import List, Dict, Any, Optional
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

SENSOR_PROFILES: Dict[str, Dict[str, Any]] = {
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
                subtype="pedestrian",
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
                subtype="lead_vehicle",
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
                subtype="camera_offline",
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
                subtype=None,
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

TRIP_TIMELINE: List[Dict[str, Any]] = [
    {
        "timestep": 1,
        "time_offset_sec": 0,
        "description": "Nominal start: cruising down clear boulevard at 40 km/h.",
        "ego_speed": 40.0,
        "events": [
            {
                "id": 1,
                "type": "clear",
                "subtype": None,
                "position": "front",
                "distance": None,
                "confidence": 1.0,
                "sensor": "camera",
                "sensor_status": "active",
                "relative_speed_kmh": 0.0
            }
        ]
    },
    {
        "timestep": 2,
        "time_offset_sec": 4,
        "description": "Approaching slower vehicle ahead (18m, moderate closing speed).",
        "ego_speed": 40.0,
        "events": [
            {
                "id": 2,
                "type": "vehicle",
                "subtype": "sedan",
                "position": "front",
                "distance": 18.0,
                "confidence": 0.95,
                "sensor": "camera",
                "sensor_status": "active",
                "relative_speed_kmh": 15.0
            }
        ]
    },
    {
        "timestep": 3,
        "time_offset_sec": 8,
        "description": "Pedestrian suddenly steps onto front lane 6m away!",
        "ego_speed": 35.0,
        "events": [
            {
                "id": 3,
                "type": "pedestrian",
                "subtype": "pedestrian",
                "position": "front",
                "distance": 6.0,
                "confidence": 0.98,
                "sensor": "camera",
                "sensor_status": "active",
                "relative_speed_kmh": 35.0
            }
        ]
    },
    {
        "timestep": 4,
        "time_offset_sec": 12,
        "description": "Construction barrier on left lane margin (12m away).",
        "ego_speed": 30.0,
        "events": [
            {
                "id": 4,
                "type": "static_obstacle",
                "subtype": "construction_barrier",
                "position": "left",
                "distance": 12.0,
                "confidence": 0.92,
                "sensor": "lidar",
                "sensor_status": "active",
                "relative_speed_kmh": 0.0
            }
        ]
    },
    {
        "timestep": 5,
        "time_offset_sec": 16,
        "description": "Dual tight situation: debris on left (10m) and cyclist on right (8m).",
        "ego_speed": 35.0,
        "events": [
            {
                "id": 5,
                "type": "static_obstacle",
                "subtype": "debris",
                "position": "left",
                "distance": 10.0,
                "confidence": 0.90,
                "sensor": "lidar",
                "sensor_status": "active",
                "relative_speed_kmh": 0.0
            },
            {
                "id": 6,
                "type": "cyclist",
                "subtype": "cyclist",
                "position": "right",
                "distance": 8.0,
                "confidence": 0.93,
                "sensor": "camera",
                "sensor_status": "active",
                "relative_speed_kmh": 10.0
            }
        ]
    },
    {
        "timestep": 6,
        "time_offset_sec": 20,
        "description": "Entering dense fog tunnel; sensor input degraded with low confidence.",
        "ego_speed": 40.0,
        "events": [
            {
                "id": 7,
                "type": "vehicle",
                "subtype": "truck",
                "position": "front",
                "distance": 16.0,
                "confidence": 0.38,
                "sensor": "camera",
                "sensor_status": "degraded",
                "relative_speed_kmh": 10.0
            }
        ]
    },
    {
        "timestep": 7,
        "time_offset_sec": 24,
        "description": "Temporary perception disconnect (hardware glitch).",
        "ego_speed": 30.0,
        "events": [
            {
                "id": 8,
                "type": "sensor_failure",
                "subtype": "sensor_gap",
                "position": "unknown",
                "distance": None,
                "confidence": 0.0,
                "sensor": "camera",
                "sensor_status": "failed",
                "relative_speed_kmh": None
            }
        ]
    },
    {
        "timestep": 8,
        "time_offset_sec": 28,
        "description": "Sensors restored; highway merge ahead is clear.",
        "ego_speed": 45.0,
        "events": [
            {
                "id": 9,
                "type": "clear",
                "subtype": None,
                "position": "front",
                "distance": None,
                "confidence": 1.0,
                "sensor": "camera",
                "sensor_status": "active",
                "relative_speed_kmh": 0.0
            }
        ]
    }
]


# Backward-compatible flat scenario list
scenario: List[Dict[str, Any]] = [
    {
        "time": step.get("time_offset_sec", idx * 5),
        "type": step["events"][0]["type"],
        "distance": step["events"][0]["distance"],
        "position": step["events"][0]["position"],
        "confidence": step["events"][0]["confidence"],
        "sensor_status": step["events"][0]["sensor_status"]
    }
    for idx, step in enumerate(TRIP_TIMELINE)
]


def get_event(index: int) -> Dict[str, Any]:
    """Retrieves an event from the flat scenario list by index (backward-compatible)."""
    return scenario[index % len(scenario)]


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
    def get_trip_timeline() -> List[Dict[str, Any]]:
        return TRIP_TIMELINE

    @staticmethod
    def run_scenario(key: str) -> Dict[str, Any]:
        scenario_data = SCENARIOS.get(key)
        if not scenario_data:
            raise ValueError(f"Scenario '{key}' not found. Available: {list(SCENARIOS.keys())}")

        ego_state = scenario_data["ego_state"]
        events = scenario_data["events"]
        decision = make_decisions(events, ego_state)

        return {
            "key": key,
            "title": scenario_data["title"],
            "description": scenario_data["description"],
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
# CLI TEST ENTRY
# ------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ALL PRESET SIMULATION SCENARIOS")
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
