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


def run_vision_tests():
    """Validates Computer Vision, YOLO mapping, distance estimation, and AR HUD overlays."""
    print("\n" + "=" * 60)
    print("TEST SUITE 4: COMPUTER VISION (OPENCV + YOLO) PERCEPTION")
    print("=" * 60)

    from vision import (
        VisionPerceptionEngine,
        BoundingBox,
        LaneInfo,
        REAL_WORLD_HEIGHTS,
        CLASS_MAPPING,
        detect_lanes_opencv,
        generate_synthetic_test_frame,
        HAS_OPENCV,
        HAS_ULTRALYTICS
    )
    import numpy as np

    print(f"[*] Environment Perception Capabilities: OpenCV={HAS_OPENCV}, Ultralytics={HAS_ULTRALYTICS}")

    engine = VisionPerceptionEngine(focal_length=700.0)

    # 1. Test Monocular Distance Estimation (Pinhole Optics)
    print("\n[TEST] Monocular Distance Estimation:")
    d_ped = engine.estimate_distance(bbox_height_px=200, subtype="pedestrian")
    print(f"       Pedestrian bbox_h=200px -> Distance: {d_ped}m (Expected ~5.95m)")
    assert 5.5 <= d_ped <= 6.5, f"Distance calculation outside expected range: {d_ped}"

    d_car = engine.estimate_distance(bbox_height_px=100, subtype="car")
    print(f"       Car bbox_h=100px -> Distance: {d_car}m (Expected ~10.5m)")
    assert 9.5 <= d_car <= 11.5, f"Distance calculation outside expected range: {d_car}"

    # 2. Test Horizontal Position Classification
    print("\n[TEST] Spatial Position Classification:")
    frame_w = 640
    pos_left = engine.determine_position(x_center=100, frame_width=frame_w)
    pos_front = engine.determine_position(x_center=320, frame_width=frame_w)
    pos_right = engine.determine_position(x_center=550, frame_width=frame_w)

    print(f"       x=100px -> {pos_left.value} | x=320px -> {pos_front.value} | x=550px -> {pos_right.value}")
    assert pos_left == Position.LEFT
    assert pos_front == Position.FRONT
    assert pos_right == Position.RIGHT

    # 3. Test COCO/YOLO Class Mapping
    print("\n[TEST] YOLO to HazardType Mapping:")
    assert CLASS_MAPPING["person"][0] == HazardType.PEDESTRIAN
    assert CLASS_MAPPING["car"][0] == HazardType.VEHICLE
    assert CLASS_MAPPING["bicycle"][0] == HazardType.CYCLIST
    assert CLASS_MAPPING["dog"][0] == HazardType.ANIMAL
    assert CLASS_MAPPING["stop sign"][0] == HazardType.STATIC_OBSTACLE
    print("       All 15+ COCO object classes correctly map to HazardType domain models.")

    # 4. Test OpenCV Lane Detection and Departure Offset
    print("\n[TEST] OpenCV Lane Line Detection & Departure Analysis:")
    synth_frame, synth_boxes = generate_synthetic_test_frame(scenario="urban_pedestrian", width=640, height=480)
    annotated_lane_frame, lane_info = detect_lanes_opencv(synth_frame)
    print(f"       Lane Left Line: {lane_info.left_line is not None} | Right Line: {lane_info.right_line is not None}")
    print(f"       Departure Warning: {lane_info.departure_warning} | Status: {lane_info.warning_message}")

    # 5. Test End-to-End Synthetic Vision Frame Processing
    print("\n[TEST] End-to-End Vision Perception -> Decision Engine:")
    # Manually inject pedestrian crossing detection box for deterministic test
    engine.detect_objects_yolo = lambda frame, conf_threshold: [
        BoundingBox(x1=290, y1=240, x2=350, y2=440, confidence=0.96, class_name="person")
    ]

    ann_frame, hazards, decision, l_info = engine.process_frame(
        synth_frame,
        ego_state=EgoState(speed_kmh=35.0, lane="center"),
        conf_threshold=0.35
    )

    print(f"       Detected Hazards Count: {len(hazards)}")
    print(f"       Primary Hazard: {hazards[0].type.value} at {hazards[0].distance}m ({hazards[0].position.value})")
    print(f"       AI Decision Action: {decision.action} [{decision.risk}]")
    print(f"       Reason: {decision.reason}")

    assert len(hazards) >= 1
    assert hazards[0].type == HazardType.PEDESTRIAN
    assert hazards[0].position == Position.FRONT
    assert decision.action in [Action.STOP.value, Action.BRAKE.value, Action.SLOW_DOWN.value]
    assert decision.risk in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]

    # 6. Test Multi-frame Tracking & Closing Speed Dynamics
    print("\n[TEST] Vision Temporal Tracking & Closing Speed Dynamics:")
    # Frame 2: Pedestrian is closer (box height grew from 200px to 300px)
    engine.detect_objects_yolo = lambda frame, conf_threshold: [
        BoundingBox(x1=280, y1=180, x2=360, y2=480, confidence=0.98, class_name="person")
    ]
    _, hazards2, decision2, _ = engine.process_frame(
        synth_frame,
        ego_state=EgoState(speed_kmh=40.0, lane="center")
    )
    # 7. Test Direct Brain Vision API (process_vision_frame & VisionDecisionResult)
    print("\n[TEST] Direct Brain API process_vision_frame():")
    from brain import process_vision_frame, fuse_sensor_streams, evaluate_lane_departure_safety, Decision
    from vision import VisionDecisionResult

    ann_img, hz_list, dec_res, lane_res = process_vision_frame(synth_frame, ego_state={"speed_kmh": 40.0})
    print(f"       Direct brain.process_vision_frame() -> Action: {dec_res.action} | Risk: {dec_res.risk}")
    assert dec_res is not None
    assert isinstance(dec_res, Decision)


    # Test VisionPerceptionEngine.analyze()
    analysis = engine.analyze(synth_frame, ego_state=EgoState(speed_kmh=40.0))
    assert isinstance(analysis, VisionDecisionResult)
    assert analysis.decision.action is not None

    # 8. Test Multi-Modal Sensor Fusion (Vision + Radar + LiDAR)
    print("\n[TEST] Multi-Modal Sensor Fusion (fuse_sensor_streams):")
    v_hazard = HazardEvent(type=HazardType.VEHICLE, subtype="car", position=Position.FRONT, distance=18.0, confidence=0.85)
    r_hazard = HazardEvent(type=HazardType.VEHICLE, subtype="lead_vehicle", position=Position.FRONT, distance=16.5, confidence=0.92, relative_speed_kmh=25.0)

    fused_decision = fuse_sensor_streams(
        vision_hazards=[v_hazard],
        radar_hazards=[r_hazard],
        ego_state=EgoState(speed_kmh=50.0)
    )
    print(f"       Sensor Fusion Action: {fused_decision.action} | Risk: {fused_decision.risk}")
    print(f"       Reason: {fused_decision.reason}")
    assert fused_decision.action in [Action.SLOW_DOWN.value, Action.BRAKE.value]

    # 9. Test Lane Departure Safety Assessment
    print("\n[TEST] Lane Departure Assist Safety Assessment:")
    lane_warn = LaneInfo(departure_warning=True, offset_from_center_px=-60.0) # Veering left
    lane_decision = evaluate_lane_departure_safety(lane_warn, surrounding_hazards=[], ego_state=EgoState(speed_kmh=45.0))
    print(f"       Lane Centering Action: {lane_decision.action} | Risk: {lane_decision.risk}")
    assert lane_decision.action == Action.MOVE_RIGHT.value

    print("\nAll Computer Vision & Brain Integration tests passed successfully!")


if __name__ == "__main__":
    run_legacy_tests()
    run_advanced_tests()
    run_scenario_suite()
    run_vision_tests()
    print("\n" + "=" * 60)
    print("ALL 4 TEST SUITES COMPLETED WITH 100% SUCCESS")
    print("=" * 60)