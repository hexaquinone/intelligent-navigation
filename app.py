# ============================================================
# INTELLIGENT NAVIGATION & DECISION-SUPPORT SYSTEM
# Autonomous Vehicle Cockpit & Multi-Sensor Perception Suite
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from brain import (
    make_decision,
    make_decisions,
    fuse_sensor_streams,
    HazardEvent,
    HazardType,
    Position,
    SensorStatus,
    RiskLevel,
    Action,
    EgoState,
    Decision,
    calculate_ttc
)
from simulation import (
    SimulationEngine,
    SCENARIOS,
    TRIP_TIMELINE,
    SENSOR_PROFILES
)
from metrics import (
    record_event,
    get_metrics,
    reset_metrics
)
from road import render_road_simulation_component
from vision import (
    VisionPerceptionEngine,
    VisionDecisionResult,
    generate_animated_driving_frame,
    generate_synthetic_test_frame,
    HAS_OPENCV,
    HAS_ULTRALYTICS
)
import numpy as np
import cv2
import tempfile
from PIL import Image


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Navigation & Cockpit HUD",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CYBERNETIC COCKPIT CSS THEME
# ============================================================

st.markdown("""
<style>
/* Base Dark Theme */
.stApp {
    background-color: #06090e;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}

/* Header */
.hud-title {
    font-size: 2.3rem;
    font-weight: 850;
    letter-spacing: -0.5px;
    color: #f8fafc;
    margin-bottom: 0px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hud-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 2px;
    margin-bottom: 1.0rem;
}

/* Panels */
.hud-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}

.section-label {
    font-size: 1.05rem;
    font-weight: 750;
    color: #f1f5f9;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Action Boxes */
.act-box {
    text-align: center;
    padding: 18px;
    border-radius: 12px;
    font-size: 2.0rem;
    font-weight: 900;
    letter-spacing: 1px;
    margin: 8px 0 14px 0;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
}

.act-continue { background: linear-gradient(135deg, #064e3b, #047857); color: #ecfdf5; border: 1px solid #10b981; }
.act-slow { background: linear-gradient(135deg, #78350f, #b45309); color: #fffbeb; border: 1px solid #f59e0b; }
.act-brake { background: linear-gradient(135deg, #881337, #be123c); color: #fff1f2; border: 1px solid #f43f5e; }
.act-stop { background: linear-gradient(135deg, #7f1d1d, #b91c1c); color: #fef2f2; border: 1px solid #ef4444; }
.act-swerve { background: linear-gradient(135deg, #1e1b4b, #4338ca); color: #eef2ff; border: 1px solid #6366f1; }

/* Metrics */
div[data-testid="stMetric"] {
    background: #0d1522;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 800;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #04070b;
    border-right: 1px solid #1e293b;
}

/* Badges */
.badge-active { background: #064e3b; color: #34d399; padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; }
.badge-degraded { background: #78350f; color: #fbbf24; padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; }
.badge-failed { background: #7f1d1d; color: #f87171; padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. KINEMATIC COMPUTATION & BEV ROAD VISUALIZER
# ============================================================

def compute_kinematics(ego_speed_kmh: float, distance_m: Optional[float], closing_speed_kmh: Optional[float] = None) -> Dict[str, Any]:
    """Computes physics-based stopping distance, required deceleration, and safety margin."""
    v_ms = ego_speed_kmh / 3.6
    t_reaction = 1.2  # driver & ADAS reaction time in seconds
    mu = 0.75  # tire-road friction coefficient
    g = 9.81

    d_reaction = v_ms * t_reaction
    d_braking = (v_ms ** 2) / (2 * mu * g)
    total_stopping_dist = d_reaction + d_braking

    req_decel = None
    safety_margin = None

    if distance_m is not None and distance_m > 0:
        closing_v_ms = (closing_speed_kmh / 3.6) if closing_speed_kmh is not None else v_ms
        if closing_v_ms > 0:
            req_decel = round((closing_v_ms ** 2) / (2 * distance_m), 2)
        safety_margin = round(distance_m - total_stopping_dist, 1)

    return {
        "speed_ms": round(v_ms, 1),
        "reaction_dist_m": round(d_reaction, 1),
        "braking_dist_m": round(d_braking, 1),
        "total_stopping_dist_m": round(total_stopping_dist, 1),
        "required_decel_ms2": req_decel,
        "safety_margin_m": safety_margin
    }


def render_bev_road_component(hazards: List[HazardEvent], decision: Decision, ego_speed_kmh: float):
    """Renders the 2D Bird's-Eye View (BEV) road canvas inside an embedded HTML iframe."""
    svg_w = 460
    svg_h = 420

    lane_w = 110
    road_left = 65
    road_right = road_left + (3 * lane_w)

    left_lane_x = road_left + (lane_w / 2)
    center_lane_x = left_lane_x + lane_w
    right_lane_x = center_lane_x + lane_w

    ego_x = center_lane_x
    ego_y = svg_h - 50

    # Maneuver vector
    arrow_markup = ""
    act_str = str(decision.action).upper()
    if "MOVE_RIGHT" in act_str:
        arrow_markup = f'<path d="M {ego_x} {ego_y - 20} Q {ego_x + 30} {ego_y - 70} {right_lane_x} {ego_y - 120}" fill="none" stroke="#60a5fa" stroke-width="4" stroke-dasharray="6,4" marker-end="url(#arrow-blue)"/>'
    elif "MOVE_LEFT" in act_str:
        arrow_markup = f'<path d="M {ego_x} {ego_y - 20} Q {ego_x - 30} {ego_y - 70} {left_lane_x} {ego_y - 120}" fill="none" stroke="#60a5fa" stroke-width="4" stroke-dasharray="6,4" marker-end="url(#arrow-blue)"/>'
    elif "STOP" in act_str or "BRAKE" in act_str:
        arrow_markup = f'<line x1="{road_left + 15}" y1="{ego_y - 60}" x2="{road_right - 15}" y2="{ego_y - 60}" stroke="#ef4444" stroke-width="4" stroke-dasharray="8,4"/>'

    # Hazard markers
    hazard_markers = ""
    for h in hazards:
        h_type_str = h.type.value if hasattr(h.type, "value") else str(h.type).lower()
        if "clear" in h_type_str or "sensor_failure" in h_type_str or "sensor_gap" in h_type_str:
            continue

        h_dist = h.distance if h.distance is not None else 25.0
        norm_dist = min(max(h_dist, 3.0), 50.0) / 50.0
        h_y = (ego_y - 30) - (norm_dist * (svg_h - 110))

        pos_str = h.position.value if hasattr(h.position, "value") else str(h.position).lower()
        if pos_str == "left":
            h_x = left_lane_x
        elif pos_str == "right":
            h_x = right_lane_x
        else:
            h_x = center_lane_x

        color = "#ef4444" if decision.risk in ["HIGH", "CRITICAL"] else ("#f59e0b" if decision.risk == "MEDIUM" else "#3b82f6")
        icon = "🚶" if "pedestrian" in h_type_str else ("🚙" if "vehicle" in h_type_str else ("🚴" if "cyclist" in h_type_str else ("🚧" if "obstacle" in h_type_str else "⚠️")))

        hazard_markers += f'''
        <g transform="translate({h_x}, {h_y})">
            <circle cx="0" cy="0" r="22" fill="{color}" opacity="0.25"/>
            <circle cx="0" cy="0" r="16" fill="#1e293b" stroke="{color}" stroke-width="2.5"/>
            <text x="0" y="5" font-size="14" text-anchor="middle">{icon}</text>
            <rect x="-24" y="-30" width="48" height="15" rx="4" fill="#0f172a" stroke="{color}" stroke-width="1"/>
            <text x="0" y="-19" font-size="9" fill="#f8fafc" font-weight="bold" text-anchor="middle">{h_dist:.1f}m</text>
        </g>
        '''

    svg_content = f'''
    <svg width="100%" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background:#090d14; border-radius:14px; border:1px solid #1e293b;">
        <defs>
            <marker id="arrow-blue" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#60a5fa"/>
            </marker>
        </defs>

        <!-- Asphalt Road Surface -->
        <rect x="{road_left}" y="10" width="{road_right - road_left}" height="{svg_h - 20}" rx="6" fill="#151d28" stroke="#334155" stroke-width="2"/>

        <!-- Distance Rings -->
        <line x1="{road_left}" y1="80" x2="{road_right}" y2="80" stroke="#334155" stroke-width="1" stroke-dasharray="3,3"/>
        <text x="{road_left - 8}" y="83" font-size="10" fill="#64748b" text-anchor="end">40m</text>
        <line x1="{road_left}" y1="180" x2="{road_right}" y2="180" stroke="#334155" stroke-width="1" stroke-dasharray="3,3"/>
        <text x="{road_left - 8}" y="183" font-size="10" fill="#64748b" text-anchor="end">20m</text>
        <line x1="{road_left}" y1="260" x2="{road_right}" y2="260" stroke="#f59e0b" stroke-width="1" stroke-dasharray="4,4" opacity="0.6"/>
        <text x="{road_left - 8}" y="263" font-size="10" fill="#f59e0b" text-anchor="end">10m</text>
        <line x1="{road_left}" y1="310" x2="{road_right}" y2="310" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,2"/>
        <text x="{road_left - 8}" y="313" font-size="10" fill="#ef4444" text-anchor="end">5m</text>

        <!-- Lane Lines -->
        <line x1="{road_left + lane_w}" y1="10" x2="{road_left + lane_w}" y2="{svg_h - 10}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="14,10"/>
        <line x1="{road_left + 2 * lane_w}" y1="10" x2="{road_left + 2 * lane_w}" y2="{svg_h - 10}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="14,10"/>

        <!-- Lane Headers -->
        <text x="{left_lane_x}" y="30" font-size="10" fill="#475569" font-weight="bold" text-anchor="middle">LEFT</text>
        <text x="{center_lane_x}" y="30" font-size="10" fill="#475569" font-weight="bold" text-anchor="middle">CENTER</text>
        <text x="{right_lane_x}" y="30" font-size="10" fill="#475569" font-weight="bold" text-anchor="middle">RIGHT</text>

        <!-- Headlight Beam Glow -->
        <polygon points="{ego_x - 14},{ego_y - 25} {ego_x - 85},{ego_y - 200} {ego_x + 85},{ego_y - 200} {ego_x + 14},{ego_y - 25}" fill="url(#beam-glow)" opacity="0.15"/>
        <linearGradient id="beam-glow" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stop-color="#38bdf8"/>
            <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>
        </linearGradient>

        <!-- Vectors & Markers -->
        {arrow_markup}
        {hazard_markers}

        <!-- Ego Host Car -->
        <g transform="translate({ego_x}, {ego_y})">
            <rect x="-18" y="-32" width="36" height="56" rx="8" fill="#0284c7" stroke="#38bdf8" stroke-width="2"/>
            <path d="M -12 -12 L 12 -12 L 9 0 L -9 0 Z" fill="#0369a1"/>
            <circle cx="-12" cy="-30" r="3" fill="#fef08a"/>
            <circle cx="12" cy="-30" r="3" fill="#fef08a"/>
            <rect x="-15" y="20" width="8" height="3" fill="#ef4444"/>
            <rect x="7" y="20" width="8" height="3" fill="#ef4444"/>
            <text x="0" y="38" font-size="9" fill="#94a3b8" font-weight="bold" text-anchor="middle">HOST ({ego_speed_kmh:.0f} km/h)</text>
        </g>
    </svg>
    '''

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
    </style>
    </head>
    <body>
        {svg_content}
    </body>
    </html>
    """
    components.html(full_html, height=430, scrolling=False)


# ============================================================
# 4. SESSION STATE MANAGEMENT
# ============================================================

def initialize_session_state() -> None:
    defaults = {
        "sim_mode": "🚗 Live Trip Timeline",
        "simulation_running": False,
        "simulation_index": 0,
        "event_history": [],
        "processed_steps": set(),
        "selected_scenario": list(SCENARIOS.keys())[0],
        "fault_fog": False,
        "fault_cam_blackout": False,
        "fault_lidar_noise": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# 5. SIDEBAR & OPERATIONAL MODE SELECTION
# ============================================================

def render_sidebar_controls():
    with st.sidebar:
        st.markdown("### ⚙️ Command Control Center")

        active_mode = st.radio(
            "Operational Mode",
            [
                "🚗 Live Trip Timeline",
                "👁️ Live Vision & YOLO Perception",
                "🔬 Preset Scenario Explorer",
                "🛠️ Interactive What-If Sandbox",
                "🎮 3D Road Simulator (road.py)"
            ],
            key="sim_mode"
        )

        st.divider()

        playback_speed = 3.0
        sandbox_values = {}

        if active_mode == "🚗 Live Trip Timeline":
            st.subheader("🎮 Drive Timeline Controls")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶️ Start Drive", use_container_width=True):
                    st.session_state.simulation_running = True
            with c2:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.simulation_running = False

            c3, c4 = st.columns(2)
            with c3:
                if st.button("⏭️ Next Step", use_container_width=True):
                    st.session_state.simulation_running = False
                    if st.session_state.simulation_index < len(TRIP_TIMELINE) - 1:
                        st.session_state.simulation_index += 1
                    else:
                        st.session_state.simulation_index = 0
                    st.rerun()
            with c4:
                if st.button("🔄 Reset Drive", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.session_state.simulation_index = 0
                    st.session_state.event_history = []
                    st.session_state.processed_steps = set()
                    reset_metrics()
                    st.rerun()

            current_step_idx = st.slider(
                "Timeline Position",
                min_value=0,
                max_value=len(TRIP_TIMELINE) - 1,
                value=st.session_state.simulation_index,
                format="Step %d"
            )
            if current_step_idx != st.session_state.simulation_index:
                st.session_state.simulation_index = current_step_idx
                st.session_state.simulation_running = False
                st.rerun()

            playback_speed = st.slider("Playback Speed (sec)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

        elif active_mode == "👁️ Live Vision & YOLO Perception":
            st.subheader("👁️ Perception Settings")
            sandbox_values["vision_source"] = st.radio(
                "Input Source",
                ["Preset Animated Driving Scenes", "Upload Image or Dashcam Video", "Live Camera Snapshot (Webcam)"]
            )
            sandbox_values["vision_conf"] = st.slider("YOLO Confidence", 0.10, 0.95, 0.35, 0.05)
            sandbox_values["vision_speed"] = st.slider("Host Ego Speed (km/h)", 0.0, 120.0, 40.0, 5.0)
            sandbox_values["enable_lanes"] = st.checkbox("Enable OpenCV Lane Tracking", value=True)
            sandbox_values["enable_fusion"] = st.checkbox("🔀 Multi-Sensor Fusion (Radar/LiDAR)", value=False)


        elif active_mode == "🔬 Preset Scenario Explorer":
            st.subheader("📚 Scenario Catalog")
            sc_keys = SimulationEngine.list_scenarios()
            selected_sc = st.selectbox(
                "Select Scenario",
                options=sc_keys,
                format_func=lambda k: SCENARIOS[k]["title"]
            )
            st.session_state.selected_scenario = selected_sc

        elif active_mode == "🛠️ Interactive What-If Sandbox":
            st.subheader("🛠️ Hazard Injector")
            sandbox_values = {
                "speed": st.slider("Host Speed (km/h)", 0.0, 130.0, 45.0, 5.0),
                "hazard_type": st.selectbox("Primary Hazard", ["pedestrian", "vehicle", "static_obstacle", "cyclist", "clear"]),
                "distance": st.slider("Distance (m)", 2.0, 60.0, 14.0, 1.0),
                "position": st.selectbox("Position", ["front", "left", "right"]),
                "closing_speed": st.slider("Closing Speed (km/h)", 0.0, 80.0, 20.0, 5.0),
            }

            enable_pinch = st.checkbox("Dual Hazard (Swerve Conflict Matrix)", value=False)
            if enable_pinch:
                sandbox_values["secondary_type"] = st.selectbox("Secondary Hazard", ["cyclist", "static_obstacle", "vehicle"])
                sandbox_values["secondary_position"] = "right" if sandbox_values["position"] == "left" else "left"
                sandbox_values["secondary_distance"] = st.slider("Secondary Distance (m)", 3.0, 30.0, 10.0, 1.0)
            sandbox_values["enable_secondary"] = enable_pinch

        else:
            st.subheader("🎮 3D Road Simulator Settings")
            st.caption("Powered by `road.py` & `simulation.html`")
            sim_h = st.slider("Canvas Viewport Height (px)", min_value=650, max_value=1200, value=920, step=50)
            sandbox_values["sim_height"] = sim_h

            st.info("💡 **Tip:** You can also run the simulator standalone with `streamlit run road.py`")

        st.divider()
        st.subheader("📡 Subsystem Telemetry")
        st.markdown("🟢 **Perception Fusion:** `ONLINE`")
        st.markdown("🧠 **Brain Engine:** `ACTIVE`")
        st.markdown("👁️ **Computer Vision (YOLO):** `READY`")
        st.markdown("📊 **Blackbox Audit:** `LOGGING`")
        st.markdown("🚗 **Road Simulator (`road.py`):** `LINKED`")

    return active_mode, playback_speed, sandbox_values




# ============================================================
# 6. DATA INGESTION & SENSOR OVERLAYS
# ============================================================

def apply_sensor_faults(current_hazards: List[HazardEvent]) -> None:
    if st.session_state.fault_cam_blackout:
        for h in current_hazards:
            if h.sensor == "camera":
                h.sensor_status = SensorStatus.FAILED
                h.confidence = 0.0
                h.type = HazardType.SENSOR_FAILURE

    if st.session_state.fault_fog:
        for h in current_hazards:
            h.sensor_status = SensorStatus.DEGRADED
            h.confidence = round(h.confidence * 0.4, 2)


def get_hazard_label(hazard: HazardEvent) -> str:
    if getattr(hazard, "subtype", None):
        return str(hazard.subtype).title()
    h_type = getattr(hazard.type, "value", str(hazard.type))
    return str(h_type).replace("_", " ").title()


def get_position_label(value: Any) -> str:
    return str(getattr(value, "value", value or "unknown")).title()


def build_history_entry(primary_hazard: HazardEvent, decision: Decision, step_id: Any) -> Dict[str, Any]:
    dist_str = f"{primary_hazard.distance:.1f} m" if primary_hazard.distance is not None else "--"
    h_type_str = getattr(primary_hazard.type, "value", str(primary_hazard.type))
    pos_str = getattr(primary_hazard.position, "value", str(primary_hazard.position))

    return {
        "Step": step_id,
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Hazard": h_type_str.replace("_", " ").title(),
        "Position": pos_str.title(),
        "Distance": dist_str,
        "Confidence": f"{primary_hazard.confidence * 100:.0f}%",
        "Risk": decision.risk,
        "Action": decision.action.replace("_", " "),
        "TTC": f"{decision.ttc_seconds}s" if decision.ttc_seconds else "--",
        "Target Speed": f"{decision.target_speed_kmh:.0f} km/h" if decision.target_speed_kmh is not None else "--",
        "Reason": decision.reason,
    }


def render_top_hud(ego_state: EgoState, decision: Decision, kinematics: Dict[str, Any]) -> None:
    st.markdown('<div class="hud-title">🚗 Intelligent Navigation & Decision-Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-subtitle">Explainable Autonomous Driving Assistance, 2D BEV Perception, & Real-Time Kinematics</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Host Speed", f"{ego_state.speed_kmh:.0f} km/h")
    with k2:
        tgt = f"{decision.target_speed_kmh:.0f} km/h" if decision.target_speed_kmh is not None else f"{ego_state.speed_kmh:.0f} km/h"
        st.metric("Target Speed", tgt)
    with k3:
        risk_color = "🔴" if decision.risk in ["HIGH", "CRITICAL"] else ("🟡" if decision.risk == "MEDIUM" else "🟢")
        st.metric("Risk Level", f"{risk_color} {decision.risk}")
    with k4:
        ttc_label = f"⚡ {decision.ttc_seconds}s" if decision.ttc_seconds else "🛡️ Safe"
        st.metric("Time-To-Collision", ttc_label)
    with k5:
        st.metric("Stopping Distance", f"{kinematics['total_stopping_dist_m']} m")

    st.divider()


def render_sector_cards(current_hazards: List[HazardEvent]) -> None:
    s_left, s_front, s_right = st.columns(3)

    left_hazards = [h for h in current_hazards if (h.position == Position.LEFT or str(h.position).lower() == "left")]
    front_hazards = [h for h in current_hazards if (h.position == Position.FRONT or str(h.position).lower() == "front")]
    right_hazards = [h for h in current_hazards if (h.position == Position.RIGHT or str(h.position).lower() == "right")]

    with s_left:
        st.markdown("##### ⬅️ Left Sector")
        if left_hazards:
            for lh in left_hazards:
                dist_txt = f"{lh.distance:.1f}m" if lh.distance is not None else "N/A"
                st.warning(f"⚠️ **{get_hazard_label(lh)}** ({dist_txt})")
        else:
            st.success("🟢 Clear")

    with s_front:
        st.markdown("##### ⬆️ Front Sector")
        if front_hazards:
            for fh in front_hazards:
                fh_type_str = getattr(fh.type, "value", str(fh.type)).lower()
                if "clear" in fh_type_str:
                    st.success("🟢 Clear")
                else:
                    dist_txt = f"{fh.distance:.1f}m" if fh.distance is not None else "N/A"
                    st.error(f"⚠️ **{get_hazard_label(fh)}** ({dist_txt})")
        elif any(getattr(h.type, "value", str(h.type)).lower() == "sensor_failure" for h in current_hazards):
            st.error("📡 Sensor Gap")
        else:
            st.success("🟢 Clear")

    with s_right:
        st.markdown("##### ➡️ Right Sector")
        if right_hazards:
            for rh in right_hazards:
                dist_txt = f"{rh.distance:.1f}m" if rh.distance is not None else "N/A"
                st.warning(f"⚠️ **{get_hazard_label(rh)}** ({dist_txt})")
        else:
            st.success("🟢 Clear")


def action_class_for(action: str) -> str:
    act_str = str(action).upper()
    if "BRAKE" in act_str:
        return "act-brake"
    if "STOP" in act_str:
        return "act-stop"
    if "SLOW" in act_str:
        return "act-slow"
    if "MOVE" in act_str:
        return "act-swerve"
    return "act-continue"


def render_main_cockpit(current_hazards: List[HazardEvent], decision: Decision, ego_state: EgoState, kinematics: Dict[str, Any], step_desc: str) -> None:
    if step_desc:
        st.info(f"📍 **Drive Context:** {step_desc}")

    col_left, col_right = st.columns([1.05, 1.0])

    with col_left:
        st.markdown('<div class="section-label">🗺️ Bird\'s-Eye View (BEV) Road Perception</div>', unsafe_allow_html=True)
        render_bev_road_component(current_hazards, decision, ego_state.speed_kmh)
        render_sector_cards(current_hazards)

    with col_right:
        st.markdown('<div class="section-label">🧠 Brain Decision & Explainability Console</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="act-box {action_class_for(decision.action)}">'
            f'🚦 {decision.action.replace("_", " ")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### 💡 Explainable AI Rationale (Why?):")
        st.info(decision.reason)

        st.markdown("#### 📐 Kinematic Safety Telemetry:")
        kn1, kn2, kn3 = st.columns(3)
        with kn1:
            st.metric("Reaction Distance", f"{kinematics['reaction_dist_m']} m")
        with kn2:
            st.metric("Braking Distance", f"{kinematics['braking_dist_m']} m")
        with kn3:
            decel_disp = f"{kinematics['required_decel_ms2']} m/s²" if kinematics['required_decel_ms2'] is not None else "0.0 m/s²"
            st.metric("Required Decel", decel_disp)

        st.divider()
        st.markdown("#### 🔬 Sensor Diagnostics & Fault Injection:")
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            st.session_state.fault_fog = st.checkbox("🌫️ Severe Fog (Degraded)", value=st.session_state.fault_fog)
        with sf2:
            st.session_state.fault_cam_blackout = st.checkbox("🔌 Camera Disconnect (Failed)", value=st.session_state.fault_cam_blackout)
        with sf3:
            st.session_state.fault_lidar_noise = st.checkbox("🌧️ LiDAR Glare", value=st.session_state.fault_lidar_noise)


def render_analytics(history: List[Dict[str, Any]]) -> None:
    st.divider()
    st.markdown('<div class="section-label">📊 Cumulative Performance & Blackbox Audit</div>', unsafe_allow_html=True)

    live_metrics = get_metrics()
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Total Distance", f"{live_metrics['trip_distance_km']:.2f} km")
    with p2:
        st.metric("Hazards Detected", live_metrics["hazards_detected"])
    with p3:
        st.metric("Warnings Issued", live_metrics["warnings_count"])
    with p4:
        st.metric("Brake Actions", live_metrics["brake_events"])
    with p5:
        st.metric("Avg Confidence", f"{live_metrics['average_confidence'] * 100:.0f}%")

    if len(history) > 0:
        graph_df = pd.DataFrame(history)

        risk_num_map = {"UNCERTAIN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        graph_df["Risk Numeric"] = graph_df["Risk"].map(risk_num_map).fillna(0)
        graph_df["Distance Numeric"] = pd.to_numeric(graph_df["Distance"].str.replace(" m", "").replace("--", None), errors="coerce")

        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("⚠️ Risk Timeline")
            st.line_chart(graph_df[["Step", "Risk Numeric"]].set_index("Step"), y="Risk Numeric", color="#ef4444")
            st.caption("0: Uncertain | 1: Low | 2: Medium | 3: High | 4: Critical")

        with ch2:
            st.subheader("📏 Hazard Distance Timeline (m)")
            dist_data = graph_df[["Step", "Distance Numeric"]].dropna().set_index("Step")
            if not dist_data.empty:
                st.line_chart(dist_data, y="Distance Numeric", color="#3b82f6")

        st.subheader("📋 Blackbox Decision Audit History")
        filter_col, dl_col = st.columns([0.7, 0.3])
        with filter_col:
            risk_filter = st.selectbox("Filter Risk", ["ALL", "HIGH & CRITICAL", "MEDIUM", "LOW", "UNCERTAIN"])

        display_table = graph_df.copy()
        if risk_filter == "HIGH & CRITICAL":
            display_table = display_table[display_table["Risk"].isin(["HIGH", "CRITICAL"])]
        elif risk_filter != "ALL":
            display_table = display_table[display_table["Risk"] == risk_filter]

        st.dataframe(display_table, use_container_width=True, hide_index=True)

        with dl_col:
            csv_data = display_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Telemetry (CSV)",
                data=csv_data,
                file_name=f"telemetry_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("💡 Start the trip timeline or inject hazards to populate telemetry analytics and blackbox logs.")


def run_timeline_loop(playback_speed: float) -> None:
    if st.session_state.simulation_running:
        time.sleep(playback_speed)
        if st.session_state.simulation_index < len(TRIP_TIMELINE) - 1:
            st.session_state.simulation_index += 1
        else:
            st.session_state.simulation_running = False
        st.rerun()


active_mode, playback_speed, sandbox_values = render_sidebar_controls()


# ============================================================
# 6. MODE DISPATCH & COCKPIT / SIMULATOR RENDERING
# ============================================================

if active_mode == "🎮 3D Road Simulator (road.py)":
    st.markdown('<div class="hud-title">🚗 AI Road Safety Simulation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-subtitle">Interactive Real-Time Driving Simulation, Multi-Hazard Road Physics, & Perception Overlay (via road.py & simulation.html)</div>', unsafe_allow_html=True)

    info_c1, info_c2, info_c3 = st.columns(3)
    with info_c1:
        st.info("🎮 **Simulation Controls:** Use interactive onscreen buttons or WASD / Arrow keys to maneuver.")
    with info_c2:
        st.success("🟢 **Perception Status:** Canvas Graphics & Physics Engine Active")
    with info_c3:
        st.warning("⚡ **Standalone Mode:** You can also launch directly via `streamlit run road.py`")

    canvas_h = sandbox_values.get("sim_height", 920)
    render_road_simulation_component(height=canvas_h)

elif active_mode == "👁️ Live Vision & YOLO Perception":
    st.markdown('<div class="hud-title">👁️ Computer Vision & YOLO Perception Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-subtitle">Real-Time Object Detection (YOLOv8), Monocular Distance Estimation, OpenCV Lane Tracking, and Explainable Driving Decisions</div>', unsafe_allow_html=True)

    source_choice = sandbox_values.get("vision_source", "Preset Animated Driving Scenes")
    conf_val = sandbox_values.get("vision_conf", 0.35)
    speed_val = sandbox_values.get("vision_speed", 40.0)
    enable_lanes = sandbox_values.get("enable_lanes", True)
    enable_fusion = sandbox_values.get("enable_fusion", False)

    ego_state = EgoState(speed_kmh=speed_val, lane="center")
    frame_to_process = None
    override_boxes = None

    if source_choice == "Preset Animated Driving Scenes":
        sc_col1, sc_col2 = st.columns([1.2, 1.0])
        with sc_col1:
            sc_choice = st.selectbox(
                "Select Driving Scenario",
                [
                    "🚶 Urban Pedestrian Crossing (Center Lane Risk)",
                    "🚙 Highway Lead Vehicle Rapid Deceleration",
                    "🚧 Dual Hazard Pinch (Left Barrier + Right Cyclist)"
                ]
            )
            sc_key = "urban_pedestrian" if "Pedestrian" in sc_choice else ("highway_lead_vehicle" if "Highway" in sc_choice else "dual_hazard_pinch")
        with sc_col2:
            frame_slider = st.slider("🎬 Animation Timeline Step (30 FPS)", 0, 160, 25, 1)

        frame_to_process, override_boxes = generate_animated_driving_frame(scenario=sc_key, frame_idx=frame_slider)

    elif source_choice == "Upload Image or Dashcam Video":
        uploaded_file = st.file_uploader("Upload Dashcam Image or Video (JPG, PNG, MP4, AVI, MOV)", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                image_pil = Image.open(uploaded_file).convert("RGB")
                frame_to_process = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            else:
                # Video file processing
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

                v_col1, v_col2 = st.columns([1.2, 1.0])
                with v_col1:
                    st.caption(f"🎥 Video Loaded: {total_frames} frames ({fps} FPS)")
                with v_col2:
                    frame_num = st.slider("Video Frame Scrubber", 0, max(0, total_frames - 1), 0, 1)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame_read = cap.read()
                cap.release()
                if ret:
                    frame_to_process = frame_read
                else:
                    st.error("Could not extract selected frame from video.")
        else:
            st.info("👆 Please upload a dashcam image (.jpg/.png) or video clip (.mp4) above to run Computer Vision perception.")

    elif source_choice == "Live Camera Snapshot (Webcam)":
        cam_snap = st.camera_input("Capture Dashcam Frame from Webcam")
        if cam_snap is not None:
            bytes_data = cam_snap.getvalue()
            frame_to_process = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if frame_to_process is not None:
        engine = VisionPerceptionEngine(model_name="yolov8n.pt", enable_lanes=enable_lanes)
        annotated_frame, current_hazards, decision, lane_info = engine.process_frame(
            frame_to_process, ego_state=ego_state, conf_threshold=conf_val, override_boxes=override_boxes
        )

        # Multi-modal Sensor Fusion if enabled
        if enable_fusion and current_hazards:
            radar_sim = [
                HazardEvent(
                    id=h.id,
                    type=h.type,
                    subtype=h.subtype,
                    position=h.position,
                    distance=h.distance * 0.98 if h.distance else None,
                    confidence=0.96,
                    sensor="radar",
                    relative_speed_kmh=h.relative_speed_kmh or 15.0
                )
                for h in current_hazards if h.type != HazardType.CLEAR
            ]
            decision = fuse_sensor_streams(current_hazards, radar_hazards=radar_sim, ego_state=ego_state)

        primary_hazard = current_hazards[0] if current_hazards else HazardEvent(type=HazardType.CLEAR)
        kinematics = compute_kinematics(ego_state.speed_kmh, primary_hazard.distance, primary_hazard.relative_speed_kmh)

        # Render Top HUD
        render_top_hud(ego_state, decision, kinematics)

        # Central Split
        c_vis, c_bev = st.columns([1.25, 1.0])
        with c_vis:
            st.markdown('<div class="section-label">📸 AR Cockpit Vision Perception (OpenCV + YOLO)</div>', unsafe_allow_html=True)
            rgb_disp = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_disp, use_container_width=True)

            if lane_info.departure_warning:
                st.error(f"⚠️ **{lane_info.warning_message}** (Offset: {lane_info.offset_from_center_px:.1f}px)")
            elif lane_info.left_line and lane_info.right_line:
                st.success("🟢 **Lane Keeping Assist:** Vehicle Centered in Host Lane")

        with c_bev:
            st.markdown('<div class="section-label">🗺️ Synchronized 2D Bird\'s-Eye View (BEV)</div>', unsafe_allow_html=True)
            render_bev_road_component(current_hazards, decision, ego_state.speed_kmh)

            st.markdown(
                f'<div class="act-box {action_class_for(decision.action)}">'
                f'🚦 {decision.action.replace("_", " ")}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.info(f"💡 **AI Rationale:** {decision.reason}")

        # Action Buttons for Blackbox Logging
        log_col1, log_col2 = st.columns([0.6, 0.4])
        with log_col1:
            if st.button("💾 Record Vision Detection to Blackbox Audit Log", use_container_width=True):
                for h in current_hazards:
                    record_event(
                        event=h,
                        risk=decision.risk,
                        action=decision.action,
                        speed_kmh=ego_state.speed_kmh,
                        dt_seconds=3.0
                    )
                st.session_state.event_history.append(build_history_entry(primary_hazard, decision, f"Vision_{len(st.session_state.event_history) + 1}"))
                st.toast("✅ Vision detection event successfully logged to trip blackbox audit!")
        with log_col2:
            if enable_fusion:
                st.success("🔀 Multi-Modal Sensor Fusion Active (Camera + Radar)")
            else:
                st.info("📷 Optical Camera Perception Active")

        # Detected Targets Breakdown
        st.markdown("### 🎯 Detected Environmental Hazards")
        hazard_rows = []
        for h in current_hazards:
            h_type_str = h.type.value if hasattr(h.type, "value") else str(h.type)
            if "clear" in h_type_str.lower():
                continue
            hazard_rows.append({
                "Target ID": h.id,
                "Classification": h_type_str.title(),
                "Subtype": (h.subtype or "--").title(),
                "Position": (h.position.value if hasattr(h.position, "value") else str(h.position)).title(),
                "Estimated Distance": f"{h.distance:.1f} m" if h.distance is not None else "--",
                "Confidence": f"{h.confidence * 100:.0f}%",
                "Relative Speed": f"{h.relative_speed_kmh:.1f} km/h" if h.relative_speed_kmh is not None else "--",
                "Sensor": h.sensor
            })

        if hazard_rows:
            st.dataframe(pd.DataFrame(hazard_rows), use_container_width=True, hide_index=True)
        else:
            st.success("🟢 No active roadway hazards detected in field of view.")

        # Analytics / Blackbox History
        render_analytics(st.session_state.event_history)


else:

    # ------------------------------------------------------------
    # SENSOR INGESTION & KINEMATICS EVALUATION
    # ------------------------------------------------------------
    current_hazards: List[HazardEvent] = []
    ego_state = EgoState(speed_kmh=40.0)
    step_desc = ""
    step_id: Any = 1

    if active_mode == "🚗 Live Trip Timeline":
        timeline = SimulationEngine.get_trip_timeline()
        curr_step = timeline[st.session_state.simulation_index]
        step_id = curr_step.get("timestep", st.session_state.simulation_index + 1)
        step_desc = curr_step.get("description", "")
        ego_state = EgoState(speed_kmh=curr_step.get("ego_speed", 40.0))
        current_hazards = [HazardEvent.from_dict(ev) for ev in curr_step.get("events", [])]

    elif active_mode == "🔬 Preset Scenario Explorer":
        sc_data = SimulationEngine.get_scenario(st.session_state.selected_scenario)
        ego_state = sc_data["ego_state"]
        step_desc = sc_data["description"]
        current_hazards = sc_data["events"]
        step_id = st.session_state.selected_scenario

    else:
        ego_state = EgoState(speed_kmh=sandbox_values.get("speed", 45.0))
        step_desc = f"Interactive Sandbox: {sandbox_values.get('hazard_type')} at {sandbox_values.get('distance')}m ({sandbox_values.get('position')})."
        h1 = HazardEvent(
            id=901,
            type=HazardType.from_value(sandbox_values.get("hazard_type", "pedestrian")),
            subtype=sandbox_values.get("hazard_type", "pedestrian"),
            position=Position.from_value(sandbox_values.get("position", "front")),
            distance=sandbox_values.get("distance") if sandbox_values.get("hazard_type") != "clear" else None,
            confidence=0.96,
            sensor_status=SensorStatus.ACTIVE,
            relative_speed_kmh=sandbox_values.get("closing_speed") if sandbox_values.get("hazard_type") != "clear" else None
        )
        current_hazards = [h1]
        if sandbox_values.get("enable_secondary"):
            h2 = HazardEvent(
                id=902,
                type=HazardType.from_value(sandbox_values["secondary_type"]),
                subtype=sandbox_values["secondary_type"],
                position=Position.from_value(sandbox_values["secondary_position"]),
                distance=sandbox_values["secondary_distance"],
                confidence=0.94,
                sensor_status=SensorStatus.ACTIVE,
                relative_speed_kmh=10.0
            )
            current_hazards.append(h2)
        step_id = "Sandbox"

    apply_sensor_faults(current_hazards)

    if len(current_hazards) == 1:
        decision: Decision = make_decision(current_hazards[0], ego_state)
    else:
        decision: Decision = make_decisions(current_hazards, ego_state)

    primary_hazard = current_hazards[0] if current_hazards else HazardEvent(type=HazardType.CLEAR)
    kinematics = compute_kinematics(ego_state.speed_kmh, primary_hazard.distance, primary_hazard.relative_speed_kmh)

    if active_mode == "🚗 Live Trip Timeline":
        step_key = f"step_{st.session_state.simulation_index}_{step_id}"
        if step_key not in st.session_state.processed_steps:
            st.session_state.processed_steps.add(step_key)
            for h in current_hazards:
                record_event(
                    event=h,
                    risk=decision.risk,
                    action=decision.action,
                    speed_kmh=ego_state.speed_kmh,
                    dt_seconds=3.0
                )
            st.session_state.event_history.append(build_history_entry(primary_hazard, decision, step_id))

    # ============================================================
    # 7. MAIN COCKPIT HUD
    # ============================================================
    render_top_hud(ego_state, decision, kinematics)

    # ============================================================
    # 8. CENTRAL SPLIT COCKPIT (BEV MAP + DECISION ENGINE)
    # ============================================================
    render_main_cockpit(current_hazards, decision, ego_state, kinematics, step_desc)

    # ============================================================
    # 9. PERFORMANCE KPIS, CHARTS & BLACKBOX AUDIT LOG
    # ============================================================
    render_analytics(st.session_state.event_history)

    # ============================================================
    # 10. AUTOMATIC TIMELINE PLAYBACK LOOP
    # ============================================================
    run_timeline_loop(playback_speed)