# ==========================================
# INTELLIGENT NAVIGATION SYSTEM
# Test Suite for Decision Engine & Simulator
# ==========================================

import sys
from brain import (
    make_decision,
    make_decisions,
    HazardEvent,
    HazardType,
    Position,
    SensorStatus,
    EgoState,
    RiskLevel,
    Action
)
from simulation import SimulationEngine

# Ensure UTF-8 output if terminal supports it, or use safe ASCII representations
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run_legacy_tests():
    """Validates full backward compatibility with legacy dictionary-based events."""
    print("\n" + "=" * 60)
    print("TEST SUITE 1: LEGACY DICTIONARY INPUTS")
    print("=" * 60)

    legacy_events = [
        {
            "id": 1,
            "type": "clear",
            "subtype": None,
            "position": "front",
            "distance": None,
            "confidence": 1.0,
            "sensor_status": "active",
            "expected_action": "CONTINUE",
            "expected_risk": "LOW"
        },
        {
            "id": 2,
            "type": "vehicle",
            "subtype": None,
            "position": "front",
            "distance": 25,
            "confidence": 0.95,
            "sensor_status": "active",
            "expected_action": "CONTINUE",
            "expected_risk": "LOW"
        },
        {
            "id": 3,
            "type": "vehicle",
            "subtype": None,
            "position": "front",
            "distance": 15,
            "confidence": 0.95,
            "sensor_status": "active",
            "expected_action": "SLOW_DOWN",
            "expected_risk": "MEDIUM"
        },
        {
            "id": 4,
            "type": "pedestrian",
            "subtype": None,
            "position": "front",
            "distance": 6,
            "confidence": 0.98,
            "sensor_status": "active",
            "expected_action": "STOP",
            "expected_risk": "HIGH"
        },
        {
            "id": 5,
            "type": "static_obstacle",
            "subtype": "tree",
            "position": "left",
            "distance": 12,
            "confidence": 0.90,
            "sensor_status": "active",
            "expected_action": "MOVE_RIGHT",
            "expected_risk": "MEDIUM"
        },
        {
            "id": 6,
            "type": "static_obstacle",
            "subtype": "building",
            "position": "right",
            "distance": 12,
            "confidence": 0.92,
            "sensor_status": "active",
            "expected_action": "MOVE_LEFT",
            "expected_risk": "MEDIUM"
        },
        {
            "id": 7,
            "type": "sensor_failure",
            "subtype": None,
            "position": None,
            "distance": None,
            "confidence": 0.0,
            "sensor_status": "failed",
            "expected_action": "SLOW_DOWN",
            "expected_risk": "UNCERTAIN"
        },
        {
            "id": 8,
            "type": "pedestrian",
            "subtype": None,
            "position": "front",
            "distance": 10,
            "confidence": 0.30,
            "sensor_status": "active",
            "expected_action": "SLOW_DOWN",
            "expected_risk": "UNCERTAIN"
        }
    ]

    passed = 0
    for event in legacy_events:
        decision = make_decision(event)
        act_ok = decision["action"] == event["expected_action"]
        risk_ok = decision["risk"] == event["expected_risk"]
        
        status_label = "[PASS]" if (act_ok and risk_ok) else "[FAIL]"
        if act_ok and risk_ok:
            passed += 1

        print(f"{status_label} Event ID {event['id']} ({event['type']}):")
        print(f"       Action: {decision['action']} (Expected: {event['expected_action']})")
        print(f"       Risk:   {decision['risk']} (Expected: {event['expected_risk']})")
        print(f"       Reason: {decision['reason']}\n")

    print(f"Summary: {passed}/{len(legacy_events)} legacy tests passed.")
    assert passed == len(legacy_events), "Not all legacy tests passed!"


def run_advanced_tests():
    """Validates TTC kinematics, multi-hazard arbitration, and sensor degradation."""
    print("\n" + "=" * 60)
    print("TEST SUITE 2: ADVANCED KINEMATICS & ARBITRATION")
    print("=" * 60)

    # 1. Dynamic TTC Highway Braking
    highway_event = HazardEvent(
        id=201,
        type=HazardType.VEHICLE,
        position=Position.FRONT,
        distance=20.0,
        confidence=0.98,
        relative_speed_kmh=80.0  # Closing at ~22.2 m/s -> TTC < 1.0s (Critical)
    )
    d1 = make_decision(highway_event, EgoState(speed_kmh=100.0))
    print("[TEST] Dynamic TTC Highway Braking:")
    print(f"       TTC: {d1.ttc_seconds}s | Action: {d1.action} | Risk: {d1.risk}")
    assert d1.action == Action.BRAKE.value, f"Expected BRAKE, got {d1.action}"
    assert d1.risk in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]

    # 2. Multi-Hazard Swerve Conflict Matrix (Tight Pinch)
    left_hazard = HazardEvent(type=HazardType.STATIC_OBSTACLE, subtype="guardrail", position=Position.LEFT, distance=10.0)
    right_hazard = HazardEvent(type=HazardType.CYCLIST, subtype="bicyclist", position=Position.RIGHT, distance=8.0)
    
    d2 = make_decisions([left_hazard, right_hazard], EgoState(speed_kmh=45.0))
    print("\n[TEST] Multi-Hazard Swerve Safety Check:")
    print(f"       Action: {d2.action} | Risk: {d2.risk}")
    print(f"       Reason: {d2.reason}")
    assert d2.action == Action.SLOW_DOWN.value, "Must not swerve into occupied right lane!"
    assert "occupied" in d2.reason.lower() or "unsafe" in d2.reason.lower()

    # 3. Degraded Sensor Mode
    degraded_event = HazardEvent(
        id=203,
        type=HazardType.STATIC_OBSTACLE,
        position=Position.FRONT,
        distance=15.0,
        confidence=0.35,
        sensor_status=SensorStatus.DEGRADED
    )
    d3 = make_decision(degraded_event)
    print("\n[TEST] Degraded Sensor Handling:")
    print(f"       Action: {d3.action} | Risk: {d3.risk}")
    print(f"       Reason: {d3.reason}")
    assert d3.risk == RiskLevel.UNCERTAIN.value
    assert d3.action == Action.SLOW_DOWN.value

    print("\nAll advanced test cases passed successfully!")


def run_scenario_suite():
    """Executes preset scenarios from the simulation catalog."""
    print("\n" + "=" * 60)
    print("TEST SUITE 3: SIMULATION SCENARIOS CATALOG")
    print("=" * 60)

    for sc_name in SimulationEngine.list_scenarios():
        res = SimulationEngine.run_scenario(sc_name)
        print(f"[SCENARIO] {res['title']}: Action={res['decision']['action']} | Risk={res['decision']['risk']}")

    print("\nAll simulation scenarios executed successfully.")


if __name__ == "__main__":
    run_legacy_tests()
    run_advanced_tests()
    run_scenario_suite()
    print("\n" + "=" * 60)
    print("ALL TEST SUITES COMPLETED WITH 100% SUCCESS")
    print("=" * 60)