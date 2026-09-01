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

if "sim_mode" not in st.session_state:
    st.session_state.sim_mode = "🚗 Live Trip Timeline"

if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

if "simulation_index" not in st.session_state:
    st.session_state.simulation_index = 0

if "event_history" not in st.session_state:
    st.session_state.event_history = []

if "processed_steps" not in st.session_state:
    st.session_state.processed_steps = set()

if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = list(SCENARIOS.keys())[0]

# Sensor Fault Toggles
if "fault_fog" not in st.session_state:
    st.session_state.fault_fog = False
if "fault_cam_blackout" not in st.session_state:
    st.session_state.fault_cam_blackout = False
if "fault_lidar_noise" not in st.session_state:
    st.session_state.fault_lidar_noise = False


# ============================================================
# 5. SIDEBAR & OPERATIONAL MODE SELECTION
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Command Control Center")

    active_mode = st.radio(
        "Operational Mode",
        [
            "🚗 Live Trip Timeline",
            "🔬 Preset Scenario Explorer",
            "🛠️ Interactive What-If Sandbox"
        ],
        key="sim_mode"
    )

    st.divider()

    # Timeline Controls
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

    # Scenario Explorer
    elif active_mode == "🔬 Preset Scenario Explorer":
        st.subheader("📚 Scenario Catalog")
        sc_keys = SimulationEngine.list_scenarios()
        selected_sc = st.selectbox(
            "Select Scenario",
            options=sc_keys,
            format_func=lambda k: SCENARIOS[k]["title"]
        )
        st.session_state.selected_scenario = selected_sc

    # Sandbox
    else:
        st.subheader("🛠️ Hazard Injector")
        sb_speed = st.slider("Host Speed (km/h)", 0.0, 130.0, 45.0, 5.0)
        sb_type = st.selectbox("Primary Hazard", ["pedestrian", "vehicle", "static_obstacle", "cyclist", "clear"])
        sb_dist = st.slider("Distance (m)", 2.0, 60.0, 14.0, 1.0)
        sb_pos = st.selectbox("Position", ["front", "left", "right"])
        sb_rel_speed = st.slider("Closing Speed (km/h)", 0.0, 80.0, 20.0, 5.0)

        enable_pinch = st.checkbox("Dual Hazard (Swerve Conflict Matrix)", value=False)
        if enable_pinch:
            sec_type = st.selectbox("Secondary Hazard", ["cyclist", "static_obstacle", "vehicle"])
            sec_pos = "right" if sb_pos == "left" else "left"
            sec_dist = st.slider("Secondary Distance (m)", 3.0, 30.0, 10.0, 1.0)

    st.divider()
    st.subheader("📡 Subsystem Telemetry")
    st.markdown("🟢 **Perception Fusion:** `ONLINE`")
    st.markdown("🧠 **Brain Engine:** `ACTIVE`")
    st.markdown("📊 **Blackbox Audit:** `LOGGING`")


# ============================================================
# 6. DATA INGESTION & SENSOR OVERLAYS
# ============================================================

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

    for ev in curr_step.get("events", []):
        current_hazards.append(HazardEvent.from_dict(ev))

elif active_mode == "🔬 Preset Scenario Explorer":
    sc_data = SimulationEngine.get_scenario(st.session_state.selected_scenario)
    ego_state = sc_data["ego_state"]
    step_desc = sc_data["description"]
    current_hazards = sc_data["events"]
    step_id = st.session_state.selected_scenario

else: # Sandbox
    ego_state = EgoState(speed_kmh=sb_speed)
    step_desc = f"Interactive Sandbox: {sb_type} at {sb_dist}m ({sb_pos})."
    h1 = HazardEvent(
        id=901,
        type=HazardType.from_value(sb_type),
        subtype=sb_type,
        position=Position.from_value(sb_pos),
        distance=sb_dist if sb_type != "clear" else None,
        confidence=0.96,
        sensor_status=SensorStatus.ACTIVE,
        relative_speed_kmh=sb_rel_speed if sb_type != "clear" else None
    )
    current_hazards = [h1]
    if enable_pinch:
        h2 = HazardEvent(
            id=902,
            type=HazardType.from_value(sec_type),
            subtype=sec_type,
            position=Position.from_value(sec_pos),
            distance=sec_dist,
            confidence=0.94,
            sensor_status=SensorStatus.ACTIVE,
            relative_speed_kmh=10.0
        )
        current_hazards.append(h2)
    step_id = "Sandbox"

# Apply Sensor Fault Injections
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


# Compute Brain Decision
if len(current_hazards) == 1:
    decision: Decision = make_decision(current_hazards[0], ego_state)
else:
    decision: Decision = make_decisions(current_hazards, ego_state)

primary_hazard = current_hazards[0] if current_hazards else HazardEvent(type=HazardType.CLEAR)
kinematics = compute_kinematics(ego_state.speed_kmh, primary_hazard.distance, primary_hazard.relative_speed_kmh)


# Record Trip Telemetry for Timeline
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

        dist_str = f"{primary_hazard.distance:.1f} m" if primary_hazard.distance is not None else "--"
        h_type_str = primary_hazard.type.value if hasattr(primary_hazard.type, "value") else str(primary_hazard.type)
        pos_str = primary_hazard.position.value if hasattr(primary_hazard.position, "value") else str(primary_hazard.position)

        hist_entry = {
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
            "Reason": decision.reason
        }
        st.session_state.event_history.append(hist_entry)


# ============================================================
# 7. MAIN COCKPIT HUD
# ============================================================

st.markdown('<div class="hud-title">🚗 Intelligent Navigation & Decision-Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="hud-subtitle">Explainable Autonomous Driving Assistance, 2D BEV Perception, & Real-Time Kinematics</div>', unsafe_allow_html=True)

# Top Telemetry HUD Strip
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


# ============================================================
# 8. CENTRAL SPLIT COCKPIT (BEV MAP + DECISION ENGINE)
# ============================================================

if step_desc:
    st.info(f"📍 **Drive Context:** {step_desc}")

col_left, col_right = st.columns([1.05, 1.0])

with col_left:
    st.markdown('<div class="section-label">🗺️ Bird\'s-Eye View (BEV) Road Perception</div>', unsafe_allow_html=True)
    # Render component safely in HTML iframe
    render_bev_road_component(current_hazards, decision, ego_state.speed_kmh)

    # Spatial Awareness Columns
    s_left, s_front, s_right = st.columns(3)

    left_hazards = [h for h in current_hazards if (h.position == Position.LEFT or str(h.position).lower() == "left")]
    front_hazards = [h for h in current_hazards if (h.position == Position.FRONT or str(h.position).lower() == "front")]
    right_hazards = [h for h in current_hazards if (h.position == Position.RIGHT or str(h.position).lower() == "right")]

    with s_left:
        st.markdown("##### ⬅️ Left Sector")
        if left_hazards:
            for lh in left_hazards:
                dist_txt = f"{lh.distance:.1f}m" if lh.distance is not None else "N/A"
                lh_name = lh.subtype.title() if lh.subtype else (lh.type.value if hasattr(lh.type, "value") else str(lh.type).title())
                st.warning(f"⚠️ **{lh_name}** ({dist_txt})")
        else:
            st.success("🟢 Clear")

    with s_front:
        st.markdown("##### ⬆️ Front Sector")
        if front_hazards:
            for fh in front_hazards:
                fh_type_str = fh.type.value if hasattr(fh.type, "value") else str(fh.type).lower()
                if "clear" in fh_type_str:
                    st.success("🟢 Clear")
                else:
                    dist_txt = f"{fh.distance:.1f}m" if fh.distance is not None else "N/A"
                    fh_name = fh.subtype.title() if fh.subtype else fh_type_str.title()
                    st.error(f"⚠️ **{fh_name}** ({dist_txt})")
        elif any(h.type in [HazardType.SENSOR_FAILURE, "sensor_failure"] for h in current_hazards):
            st.error("📡 Sensor Gap")
        else:
            st.success("🟢 Clear")

    with s_right:
        st.markdown("##### ➡️ Right Sector")
        if right_hazards:
            for rh in right_hazards:
                dist_txt = f"{rh.distance:.1f}m" if rh.distance is not None else "N/A"
                rh_name = rh.subtype.title() if rh.subtype else (rh.type.value if hasattr(rh.type, "value") else str(rh.type).title())
                st.warning(f"⚠️ **{rh_name}** ({dist_txt})")
        else:
            st.success("🟢 Clear")


with col_right:
    st.markdown('<div class="section-label">🧠 Brain Decision & Explainability Console</div>', unsafe_allow_html=True)

    act_str = str(decision.action).upper()
    act_cls = "act-continue"
    if "BRAKE" in act_str:
        act_cls = "act-brake"
    elif "STOP" in act_str:
        act_cls = "act-stop"
    elif "SLOW" in act_str:
        act_cls = "act-slow"
    elif "MOVE" in act_str:
        act_cls = "act-swerve"

    st.markdown(
        f'<div class="act-box {act_cls}">'
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
        decel_disp = f"{kinematics['required_decel_ms2']} m/s²" if kinematics['required_decel_ms2'] else "0.0 m/s²"
        st.metric("Required Decel", decel_disp)

    st.divider()

    # Sensor Fusion & Fault Injectors
    st.markdown("#### 🔬 Sensor Diagnostics & Fault Injection:")
    sf1, sf2, sf3 = st.columns(3)
    with sf1:
        st.session_state.fault_fog = st.checkbox("🌫️ Severe Fog (Degraded)", value=st.session_state.fault_fog)
    with sf2:
        st.session_state.fault_cam_blackout = st.checkbox("🔌 Camera Disconnect (Failed)", value=st.session_state.fault_cam_blackout)
    with sf3:
        st.session_state.fault_lidar_noise = st.checkbox("🌧️ LiDAR Glare", value=st.session_state.fault_lidar_noise)


# ============================================================
# 9. PERFORMANCE KPIS, CHARTS & BLACKBOX AUDIT LOG
# ============================================================

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

history = st.session_state.event_history

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


# ============================================================
# 10. AUTOMATIC TIMELINE PLAYBACK LOOP
# ============================================================

if active_mode == "🚗 Live Trip Timeline" and st.session_state.simulation_running:
    time.sleep(playback_speed)
    if st.session_state.simulation_index < len(TRIP_TIMELINE) - 1:
        st.session_state.simulation_index += 1
    else:
        st.session_state.simulation_running = False
    st.rerun()