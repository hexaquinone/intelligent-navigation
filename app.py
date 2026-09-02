# ============================================================
# INTELLIGENT NAVIGATION & DECISION-SUPPORT SYSTEM
# Autonomous Vehicle Cockpit & Multi-Sensor Perception Suite
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np
import cv2
import tempfile
import os
from PIL import Image

from brain import (
    make_decision,
    make_decisions,
    fuse_sensor_streams,
    evaluate_scene,
    compute_kinematics,
    get_sector_occupancy,
    HazardEvent,
    HazardType,
    Position,
    SensorStatus,
    RiskLevel,
    Action,
    EgoState,
    Decision,
    KinematicsTelemetry,
    SectorOccupancy,
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
    reset_metrics,
    get_event_history
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


# ============================================================
# 1. PAGE CONFIGURATION & CACHED ENGINES
# ============================================================

st.set_page_config(
    page_title="AI Nav-Pilot • Autonomous Cockpit",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_cached_vision_engine(model_name: str = "yolov8n.pt", enable_lanes: bool = True) -> VisionPerceptionEngine:
    """Caches the YOLOv8 perception engine in memory to avoid redundant model initialization."""
    return VisionPerceptionEngine(model_name=model_name, enable_lanes=enable_lanes)


# ============================================================
# 2. CYBERNETIC COCKPIT CSS THEME & GLASSMORPHISM
# ============================================================

st.markdown("""
<style>
/* Header & Navbar clearance */
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 10 !important;
}

/* Base Dark Theme */
.stApp {
    background-color: #06090e;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Generous Top Padding for 100% Unobscured Visibility */
.block-container {
    max-width: 1650px;
    padding-top: 4.6rem !important;
    padding-bottom: 2.5rem;
}

/* Prevent Streamlit DOM Ghosting & Fading during Auto-Rerun */
div[data-testid="stVerticalBlock"] > div.element-container,
div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
    opacity: 1 !important;
    transition: opacity 0.1s ease-in-out !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #050a12;
    border-right: 1px solid #1e293b;
}

.sidebar-brand-card {
    background: linear-gradient(135deg, #091224 0%, #111e38 100%);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 18px rgba(56, 189, 248, 0.12);
}

.sidebar-brand-title {
    font-size: 1.15rem;
    font-weight: 850;
    letter-spacing: 0.5px;
    color: #38bdf8;
    margin: 0;
}

.sidebar-brand-sub {
    font-size: 0.74rem;
    color: #94a3b8;
    font-weight: 600;
    margin-top: 2px;
}

.sidebar-box {
    background: #090e17;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}

.sidebar-box-title {
    font-size: 0.80rem;
    font-weight: 800;
    color: #f1f5f9;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Top Hero Banner */
.hud-header {
    background: linear-gradient(135deg, #0a1329 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px 24px;
    margin-top: 0.3rem;
    margin-bottom: 1.0rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}

.hud-title {
    font-size: 1.85rem;
    font-weight: 850;
    letter-spacing: -0.5px;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hud-subtitle {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Mode Purpose Hero Banner */
.mode-hero-banner {
    background: linear-gradient(135deg, #091326 0%, #0d1b36 100%);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.08);
}

.mode-hero-title {
    font-size: 1.20rem;
    font-weight: 850;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 5px;
}

.mode-hero-desc {
    font-size: 0.90rem;
    color: #cbd5e1;
    line-height: 1.45;
    margin-bottom: 10px;
}

.mode-pillars-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 10px;
    margin-top: 8px;
    margin-bottom: 6px;
}

.mode-pillar-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid #1e293b;
    border-radius: 9px;
    padding: 9px 12px;
}

.mode-pillar-header {
    font-size: 0.76rem;
    font-weight: 800;
    color: #7dd3fc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 5px;
}

.mode-pillar-text {
    font-size: 0.80rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.35;
}

.mode-hero-tags {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    margin-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 8px;
}

.mode-tag {
    background: rgba(56, 189, 248, 0.10);
    color: #bae6fd;
    border: 1px solid rgba(56, 189, 248, 0.22);
    font-size: 0.72rem;
    padding: 3px 8px;
    border-radius: 5px;
    font-weight: 650;
}

/* Glass Section Container */
.glass-panel {
    background: #090e17;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 1rem;
}

.section-label {
    font-size: 0.88rem;
    font-weight: 800;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Action Boxes */
.act-box {
    text-align: center;
    padding: 14px;
    border-radius: 12px;
    font-size: 1.75rem;
    font-weight: 900;
    letter-spacing: 0.8px;
    margin: 4px 0 10px 0;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.35);
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
    padding: 10px 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-size: 1.30rem;
    font-weight: 800;
}

/* Badges */
.badge-active { background: #064e3b; color: #34d399; padding: 4px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 750; border: 1px solid #059669; }
.badge-degraded { background: #78350f; color: #fbbf24; padding: 4px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 750; border: 1px solid #d97706; }
.badge-failed { background: #7f1d1d; color: #f87171; padding: 4px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 750; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. 2D BIRD'S-EYE VIEW (BEV) ROAD MAP GENERATOR
# ============================================================

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
        icon = "🚶" if "pedestrian" in h_type_str else ("🐕" if "animal" in h_type_str else ("🚙" if "vehicle" in h_type_str else ("🚴" if "cyclist" in h_type_str else ("🚧" if "obstacle" in h_type_str else "⚠️"))))

        hazard_markers += f"""
        <g transform="translate({h_x}, {h_y})">
            <circle cx="0" cy="0" r="22" fill="{color}" opacity="0.25"/>
            <circle cx="0" cy="0" r="16" fill="#1e293b" stroke="{color}" stroke-width="2.5"/>
            <text x="0" y="5" font-size="14" text-anchor="middle">{icon}</text>
            <rect x="-24" y="-30" width="48" height="15" rx="4" fill="#0f172a" stroke="{color}" stroke-width="1"/>
            <text x="0" y="-19" font-size="9" fill="#f8fafc" font-weight="bold" text-anchor="middle">{h_dist:.1f}m</text>
        </g>
        """

    svg_content = f"""
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
    """

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
        "last_active_mode": "🚗 Live Trip Timeline",
        "simulation_running": True,   # Auto-run by default on entering trip mode
        "simulation_index": 0,
        "trip_loop_count": 0,
        "trip_auto_loop": True,
        "event_history": [],
        "processed_steps": set(),
        "sandbox_mode_type": "📚 Preset Benchmark Catalog",
        "selected_scenario": list(SCENARIOS.keys())[0],
        "vision_anim_running": True,  # Auto-play by default on entering vision mode
        "vision_frame_idx": 0,
        "video_anim_running": True,   # Auto-play by default for video files
        "video_frame_idx": 0,
        "video_temp_path": None,
        "last_uploaded_name": None,
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
        st.markdown(
            """
            <div class="sidebar-brand-card">
                <div class="sidebar-brand-title">🚗 AI NAV-PILOT</div>
                <div class="sidebar-brand-sub">Autonomous Decision Cockpit • HUD v2.5</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("##### 🎛️ Operational Mode")

        active_mode = st.radio(
            "Operational Mode",
            [
                "🚗 Live Trip Timeline",
                "👁️ Live Vision & YOLO Perception",
                "🔬 Scenarios & What-If Sandbox",
                "🎮 3D Road Simulator (road.py)"
            ],
            key="sim_mode",
            label_visibility="collapsed"
        )

        # Detect tab switch and auto-activate loop
        if "last_active_mode" not in st.session_state or st.session_state.last_active_mode != active_mode:
            st.session_state.last_active_mode = active_mode
            if active_mode == "🚗 Live Trip Timeline":
                st.session_state.simulation_running = True
            elif active_mode == "👁️ Live Vision & YOLO Perception":
                st.session_state.vision_anim_running = True
                st.session_state.video_anim_running = True

        st.divider()

        playback_speed = 2.5
        sandbox_values = {}

        if active_mode == "🚗 Live Trip Timeline":
            st.markdown("##### 🎮 Drive Timeline Controls")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶️ Resume" if not st.session_state.simulation_running else "🟢 Driving", use_container_width=True, key="resume_drive_btn"):
                    st.session_state.simulation_running = True
            with c2:
                if st.button("⏸️ Pause", use_container_width=True, key="pause_drive_btn"):
                    st.session_state.simulation_running = False

            c3, c4 = st.columns(2)
            with c3:
                if st.button("⏭️ Next Step", use_container_width=True, key="next_step_btn"):
                    st.session_state.simulation_running = False
                    st.session_state.simulation_index = (st.session_state.simulation_index + 1) % len(TRIP_TIMELINE)
                    st.rerun()
            with c4:
                if st.button("🔄 Reset Drive", use_container_width=True, key="reset_drive_btn"):
                    st.session_state.simulation_running = True
                    st.session_state.simulation_index = 0
                    st.session_state.trip_loop_count = 0
                    st.session_state.event_history = []
                    st.session_state.processed_steps = set()
                    reset_metrics()
                    st.rerun()

            current_step_idx = st.slider(
                "Timeline Step",
                min_value=0,
                max_value=len(TRIP_TIMELINE) - 1,
                value=st.session_state.simulation_index,
                format="Step %d",
                key="timeline_step_slider"
            )
            if current_step_idx != st.session_state.simulation_index:
                st.session_state.simulation_index = current_step_idx
                st.session_state.simulation_running = False
                st.rerun()

            st.session_state.trip_auto_loop = st.checkbox("🔁 Auto-Loop Trip Timeline", value=st.session_state.trip_auto_loop, key="auto_loop_chk")
            playback_speed = st.slider("Step Interval (sec)", min_value=1.0, max_value=5.0, value=2.5, step=0.5, key="step_interval_slider")

        elif active_mode == "👁️ Live Vision & YOLO Perception":
            st.markdown("##### 👁️ Perception Configuration")
            sandbox_values["vision_source"] = st.radio(
                "Input Source",
                ["Preset Animated Driving Scenes", "Upload Dashcam Video / Image", "Live Camera Snapshot (Webcam)"],
                key="vision_source_radio"
            )
            sandbox_values["vision_conf"] = st.slider("YOLO Confidence Threshold", 0.10, 0.95, 0.35, 0.05, key="vision_conf_slider")
            sandbox_values["vision_speed"] = st.slider("Host Ego Speed (km/h)", 0.0, 120.0, 40.0, 5.0, key="vision_speed_slider")
            sandbox_values["enable_lanes"] = st.checkbox("Enable OpenCV Lane Tracking", value=True, key="vision_lane_chk")
            sandbox_values["enable_fusion"] = st.checkbox("🔀 Multi-Sensor Fusion (Radar/LiDAR)", value=False, key="vision_fusion_chk")

            if sandbox_values["vision_source"] == "Preset Animated Driving Scenes":
                st.divider()
                st.markdown("##### 🎬 Animation Playback")
                vc1, vc2 = st.columns(2)
                with vc1:
                    if st.button("▶️ Play" if not st.session_state.vision_anim_running else "🟢 Playing", use_container_width=True, key="play_anim_btn"):
                        st.session_state.vision_anim_running = True
                with vc2:
                    if st.button("⏸️ Pause", use_container_width=True, key="pause_anim_btn"):
                        st.session_state.vision_anim_running = False

                if st.button("🔄 Rewind Animation", use_container_width=True, key="rewind_anim_btn"):
                    st.session_state.vision_frame_idx = 0
                    st.session_state.vision_anim_running = True
                    st.rerun()

            elif sandbox_values["vision_source"] == "Upload Dashcam Video / Image":
                st.divider()
                st.markdown("##### 🎬 Video Playback Controls")
                p1, p2 = st.columns(2)
                with p1:
                    if st.button("▶️ Play Video" if not st.session_state.video_anim_running else "🟢 Playing", use_container_width=True, key="play_video_btn"):
                        st.session_state.video_anim_running = True
                with p2:
                    if st.button("⏸️ Pause Video", use_container_width=True, key="pause_video_btn"):
                        st.session_state.video_anim_running = False

                if st.button("🔄 Rewind Video", use_container_width=True, key="rewind_video_btn"):
                    st.session_state.video_frame_idx = 0
                    st.session_state.video_anim_running = True
                    st.rerun()

        elif active_mode == "🔬 Scenarios & What-If Sandbox":
            st.markdown("##### 🔬 Evaluation Mode")
            sandbox_values["sub_mode"] = st.radio(
                "Select Mode",
                ["📚 Preset Benchmark Catalog", "🛠️ Custom What-If Hazard Injector"],
                label_visibility="collapsed",
                key="sandbox_submode_radio"
            )

            if sandbox_values["sub_mode"] == "📚 Preset Benchmark Catalog":
                st.markdown("##### 📚 Benchmark Catalog")
                sc_keys = SimulationEngine.list_scenarios()
                selected_sc = st.selectbox(
                    "Select Benchmark Scenario",
                    options=sc_keys,
                    format_func=lambda k: SCENARIOS[k]["title"],
                    key="scenario_select_box"
                )
                st.session_state.selected_scenario = selected_sc
            else:
                st.markdown("##### 🛠️ Custom Hazard Injection")
                sandbox_values["speed"] = st.slider("Host Speed (km/h)", 0.0, 130.0, 45.0, 5.0, key="custom_speed_slider")
                sandbox_values["hazard_type"] = st.selectbox("Primary Hazard", ["pedestrian", "vehicle", "static_obstacle", "cyclist", "clear"], key="custom_hazard_select")
                sandbox_values["distance"] = st.slider("Distance (m)", 2.0, 60.0, 14.0, 1.0, key="custom_dist_slider")
                sandbox_values["position"] = st.selectbox("Position", ["front", "left", "right"], key="custom_pos_select")
                sandbox_values["closing_speed"] = st.slider("Closing Speed (km/h)", 0.0, 80.0, 20.0, 5.0, key="custom_closing_slider")

                enable_pinch = st.checkbox("Dual Hazard (Swerve Conflict Matrix)", value=False, key="custom_dual_chk")
                if enable_pinch:
                    sandbox_values["secondary_type"] = st.selectbox("Secondary Hazard", ["cyclist", "static_obstacle", "vehicle"], key="custom_sec_type_select")
                    sandbox_values["secondary_position"] = "right" if sandbox_values["position"] == "left" else "left"
                    sandbox_values["secondary_distance"] = st.slider("Secondary Distance (m)", 3.0, 30.0, 10.0, 1.0, key="custom_sec_dist_slider")
                sandbox_values["enable_secondary"] = enable_pinch

        else:
            st.markdown("##### 🎮 3D Simulator Viewport")
            st.caption("Powered by `road.py` & `simulation.html`")
            sim_h = st.slider("Canvas Viewport Height (px)", min_value=650, max_value=1200, value=850, step=50, key="canvas_h_slider")
            sandbox_values["sim_height"] = sim_h

        st.markdown(
            """
            <div class="sidebar-box">
                <div class="sidebar-box-title">📡 Subsystem Telemetry</div>
                <div style="font-size:0.75rem; color:#94a3b8; line-height:1.6;">
                    <div>🟢 <b>Fusion Engine:</b> <span style="color:#34d399;">ONLINE</span></div>
                    <div>🧠 <b>Decision Brain:</b> <span style="color:#34d399;">ACTIVE (100 Hz)</span></div>
                    <div>👁️ <b>YOLOv8 Neural:</b> <span style="color:#38bdf8;">READY</span></div>
                    <div>📊 <b>Blackbox Audit:</b> <span style="color:#a78bfa;">LOGGING</span></div>
                    <div>🚗 <b>3D Physics:</b> <span style="color:#fbbf24;">60 FPS LINKED</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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


def render_top_hud(ego_state: EgoState, decision: Decision, kinematics: Union[Dict[str, Any], KinematicsTelemetry]) -> None:
    st.markdown(
        """
        <div class="hud-header">
            <div>
                <div class="hud-title">🚗 Intelligent Navigation & Autonomous Decision-Support Cockpit</div>
                <div class="hud-subtitle">Explainable Multi-Modal Perception, Dynamic Kinematic Safety Envelopes & Deterministic Risk Arbitration</div>
            </div>
            <div style="display:flex; gap:8px; align-items:center;">
                <span class="badge-active">● MULTI-SENSOR FUSION ONLINE</span>
                <span class="badge-active">● XAI RATIONALE ENGINE ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Executive Overview & Engineering Architecture Guide
    with st.expander("🎯 Project Vision & System Architecture (What is Being Solved & How?)", expanded=False):
        p_col1, p_col2 = st.columns([1.1, 1.0])
        with p_col1:
            st.markdown("##### 🚨 The Critical Challenge in Autonomous Driving")
            st.markdown("""
            * **Black-Box Failures:** Traditional ADAS models act without explainability, making it impossible to verify why a sudden brake or swerve occurred.
            * **Swerve Conflict Blindspots:** Evasive steering without full 360° sector awareness frequently leads to secondary side-swipe collisions.
            * **Weather & Sensor Degradation:** Sensor dropouts (fog, glare, hardware gaps) often produce false alarms or silent catastrophic failures.
            """)
        with p_col2:
            st.markdown("##### 💡 Our Multi-Modal Engineering Solution")
            st.markdown("""
            * **Sensor Fusion (Camera + Radar + LiDAR):** Fuses optical deep learning semantics with radar Doppler velocities and LiDAR depth.
            * **Deterministic 5-Tier Priority Hierarchy:** Strict safety arbitration ($P_5$ Emergency $\\to$ $P_1$ Nominal) ensuring safety over convenience.
            * **Dynamic Physics & Kinematic Envelopes:** Real-time stopping distance calculation factoring in host speed ($v$) and road friction ($\\mu$).
            * **Explainable AI (XAI) & Audit Log:** Transparent natural language explanations and immutable blackbox CSV logs for every actuation.
            """)

        st.divider()

        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("##### 🚦 5-Tier Priority Hierarchy")
            st.markdown("""
            * **P5 (Emergency Intervention):** Collision path with $TTC < 1.0\\text{s}$ $\\to$ **EMERGENCY BRAKE / STOP**.
            * **P4 (Urgent Avoidance):** Fast closing obstacle ($TTC < 2.5\\text{s}$) $\\to$ Urgent deceleration or safe swerve.
            * **P3 (Active Caution):** Perimeter hazard or degraded sensors $\\to$ **SLOW DOWN**.
            * **P2 (Lateral Maneuver):** Clear adjacent lane $\\to$ **MOVE LEFT / MOVE RIGHT**.
            * **P1 (Nominal Cruise):** Unobstructed roadway $\\to$ **CONTINUE** cruising.
            """)
        with g2:
            st.markdown("##### 📐 Kinematics & Stopping Math")
            st.markdown("""
            * **Stopping Envelope:** $d_{\\text{stop}} = d_{\\text{reaction}} + d_{\\text{brake}}$
            * **Reaction Dist:** $d_{\\text{react}} = v \\cdot t_{\\text{react}}$ ($t_{\\text{react}} = 1.0\\text{s}$)
            * **Braking Dist:** $d_{\\text{brake}} = \\frac{v^2}{2 \\mu g}$ (Dry $\\mu=0.75$, Rain $\\mu=0.55$)
            * **Time-to-Collision (TTC):** $TTC = \\frac{d}{v_{\\text{closing}}}$
            * **Safety Margin:** $\\text{Margin} = d - d_{\\text{stop}}$
            """)
        with g3:
            st.markdown("##### 🛡️ Swerve Safety Matrix")
            st.markdown("""
            * **Left Obstacle Detected:** Check Right Sector. If clear $\\to$ `MOVE_RIGHT`. If occupied $\\to$ `SLOW_DOWN` in-lane.
            * **Right Obstacle Detected:** Check Left Sector. If clear $\\to$ `MOVE_LEFT`. If occupied $\\to$ `SLOW_DOWN` in-lane.
            * **Degraded Sensors:** Automatically reduce cruise speed and increase stopping safety margin.
            """)

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
    sectors = get_sector_occupancy(current_hazards)

    with s_left:
        st.markdown("##### ⬅️ Left Sector")
        if not sectors.is_left_clear:
            for lh in sectors.left_hazards:
                dist_txt = f"{lh.distance:.1f}m" if lh.distance is not None else "N/A"
                st.warning(f"⚠️ **{get_hazard_label(lh)}** ({dist_txt})")
        else:
            st.success("🟢 Clear")

    with s_front:
        st.markdown("##### ⬆️ Front Sector")
        if not sectors.is_front_clear:
            for fh in sectors.front_hazards:
                dist_txt = f"{fh.distance:.1f}m" if fh.distance is not None else "N/A"
                st.error(f"⚠️ **{get_hazard_label(fh)}** ({dist_txt})")
        elif any(getattr(h.type, "value", str(h.type)).lower() == "sensor_failure" for h in current_hazards):
            st.error("📡 Sensor Gap")
        else:
            st.success("🟢 Clear")

    with s_right:
        st.markdown("##### ➡️ Right Sector")
        if not sectors.is_right_clear:
            for rh in sectors.right_hazards:
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


def render_main_cockpit(current_hazards: List[HazardEvent], decision: Decision, ego_state: EgoState, kinematics: Union[Dict[str, Any], KinematicsTelemetry], step_desc: str) -> None:
    if step_desc:
        st.info(f"📍 **Drive Context:** {step_desc}")

    col_left, col_right = st.columns([1.05, 1.0])

    with col_left:
        st.markdown('<div class="section-label">🗺️ Bird\'s-Eye View (BEV) Road Perception</div>', unsafe_allow_html=True)
        render_bev_road_component(current_hazards, decision, ego_state.speed_kmh)
        render_sector_cards(current_hazards)

    with col_right:
        st.markdown('<div class="section-label">🧠 Brain Decision & Explainability Console</div>', unsafe_allow_html=True)

        p_lvl = getattr(decision, "priority_level", 1)
        p_names = {1: "Nominal Cruising", 2: "Lateral Maneuver", 3: "Active Caution", 4: "Urgent Avoidance", 5: "Emergency Intervention"}
        p_badge = f'<span style="float: right; font-size: 0.75rem; background: rgba(255,255,255,0.15); padding: 4px 8px; border-radius: 6px;">Priority {p_lvl}/5: {p_names.get(p_lvl, "Standard")}</span>'

        st.markdown(
            f'<div class="act-box {action_class_for(decision.action)}">'
            f'🚦 {decision.action.replace("_", " ")}'
            f'{p_badge}'
            f'</div>',
            unsafe_allow_html=True
        )

        arb_meta = getattr(decision, "metadata", {}) or {}
        if arb_meta.get("arbitration") in ["blocked_swerve_right", "blocked_swerve_left"]:
            st.warning("⚠️ **Swerve Conflict Arbitration:** Evasive lane change blocked by adjacent sector hazard. Brain arbitrated to safe in-lane deceleration.")

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

        if kinematics.get("safety_margin_m") is not None:
            margin = kinematics["safety_margin_m"]
            margin_txt = f"{margin:+.1f} m"
            if margin > 5.0:
                st.caption(f"🛡️ **Safety Stopping Margin:** `{margin_txt}` (Adequate Buffer)")
            elif margin >= 0:
                st.caption(f"⚠️ **Safety Stopping Margin:** `{margin_txt}` (Tight Threshold)")
            else:
                st.caption(f"🚨 **Safety Stopping Margin:** `{margin_txt}` (Negative Margin - Emergency Intervention Triggered)")

        st.divider()
        st.markdown("#### 🔬 Sensor Diagnostics & Fault Injection:")
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            st.session_state.fault_fog = st.checkbox("🌫️ Severe Fog (Degraded)", value=st.session_state.fault_fog, key="fault_fog_chk")
        with sf2:
            st.session_state.fault_cam_blackout = st.checkbox("🔌 Camera Disconnect (Failed)", value=st.session_state.fault_cam_blackout, key="fault_cam_chk")
        with sf3:
            st.session_state.fault_lidar_noise = st.checkbox("🌧️ LiDAR Glare", value=st.session_state.fault_lidar_noise, key="fault_lidar_chk")


def render_analytics(history: List[Dict[str, Any]]) -> None:
    """Renders the single consolidated blackbox audit analytics panel inside an isolated container."""
    with st.container():
        st.divider()
        st.markdown('<div class="section-label">📊 Cumulative Performance & Blackbox Audit</div>', unsafe_allow_html=True)

        live_metrics = get_metrics()
        score = live_metrics.get("safety_score", 100)
        score_color = "#10b981" if score >= 85 else ("#f59e0b" if score >= 65 else "#ef4444")

        p0, p1, p2, p3, p4, p5 = st.columns([1.2, 1.0, 1.0, 1.0, 1.0, 1.0])
        with p0:
            st.markdown(
                f"""
                <div style="background:#0d1522; border:1px solid #1e293b; border-radius:12px; padding:10px 14px; text-align:center;">
                    <div style="font-size:0.72rem; color:#94a3b8; font-weight:750; text-transform:uppercase;">Safety Score</div>
                    <div style="font-size:1.55rem; font-weight:850; color:{score_color};">{score}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
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
            # Build clean structured records with guaranteed unique sequential index
            records = []
            for i, h in enumerate(history, 1):
                records.append({
                    "#": i,
                    "Time": h.get("Time", ""),
                    "Step / Context": str(h.get("Step", f"Step {i}")),
                    "Hazard": str(h.get("Hazard", "Clear")),
                    "Position": str(h.get("Position", "Front")),
                    "Distance": str(h.get("Distance", "--")),
                    "Confidence": str(h.get("Confidence", "100%")),
                    "Risk": str(h.get("Risk", "LOW")),
                    "Action": str(h.get("Action", "CONTINUE")),
                    "TTC": str(h.get("TTC", "--")),
                    "Target Speed": str(h.get("Target Speed", "--")),
                    "Reason": str(h.get("Reason", ""))
                })

            full_df = pd.DataFrame(records)

            # Mapping for charts
            risk_num_map = {"UNCERTAIN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            full_df["_risk_num"] = full_df["Risk"].map(risk_num_map).fillna(0)

            def parse_dist(d_val):
                if not isinstance(d_val, str):
                    return float(d_val) if d_val is not None else np.nan
                cleaned = d_val.replace(" m", "").replace("--", "").replace("N/A", "").strip()
                try:
                    return float(cleaned)
                except (ValueError, TypeError):
                    return np.nan

            full_df["_dist_num"] = full_df["Distance"].apply(parse_dist)
            full_df["_chart_label"] = full_df.apply(lambda r: f"#{r['#']} {r['Step / Context']}", axis=1)

            # Charts Section
            ch1, ch2 = st.columns(2)
            with ch1:
                st.subheader("⚠️ Risk Timeline")
                risk_chart_data = full_df.set_index("_chart_label")[["_risk_num"]].rename(columns={"_risk_num": "Risk Severity (0-4)"})
                st.line_chart(risk_chart_data, color="#ef4444")
                st.caption("0: Uncertain | 1: Low | 2: Medium | 3: High | 4: Critical")

            with ch2:
                st.subheader("📏 Hazard Distance Timeline (m)")
                dist_chart_data = full_df.dropna(subset=["_dist_num"]).set_index("_chart_label")[["_dist_num"]].rename(columns={"_dist_num": "Distance (m)"})
                if not dist_chart_data.empty:
                    st.line_chart(dist_chart_data, color="#38bdf8")
                else:
                    st.caption("No distance obstacles recorded for clear highway cruising.")

            # Table & Controls Section
            st.subheader("📋 Blackbox Decision Audit History")
            
            filter_col, clear_col, dl_col = st.columns([0.5, 0.25, 0.25])
            with filter_col:
                risk_filter = st.selectbox(
                    "Filter Audit by Risk Level",
                    ["ALL", "HIGH & CRITICAL", "MEDIUM", "LOW", "UNCERTAIN"],
                    key="audit_risk_filter_box"
                )

            clean_cols = ["#", "Time", "Step / Context", "Hazard", "Position", "Distance", "Confidence", "Risk", "Action", "TTC", "Target Speed", "Reason"]
            display_table = full_df[clean_cols].copy()

            if risk_filter == "HIGH & CRITICAL":
                display_table = display_table[display_table["Risk"].isin(["HIGH", "CRITICAL"])]
            elif risk_filter != "ALL":
                display_table = display_table[display_table["Risk"] == risk_filter]

            with clear_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Audit Log", use_container_width=True, key="audit_clear_action_btn"):
                    st.session_state.event_history = []
                    st.session_state.processed_steps = set()
                    reset_metrics()
                    st.rerun()

            with dl_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                csv_data = display_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Audit CSV",
                    data=csv_data,
                    file_name="blackbox_telemetry_audit.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="audit_export_csv_btn"
                )

            st.dataframe(display_table, use_container_width=True, hide_index=True, key="audit_table_grid")

        else:
            st.info("💡 Start the trip timeline, run Computer Vision, or inject hazards to populate telemetry analytics and the blackbox audit history.")


def run_timeline_loop(playback_speed: float) -> None:
    if st.session_state.simulation_running and st.session_state.sim_mode == "🚗 Live Trip Timeline":
        time.sleep(playback_speed)
        if st.session_state.simulation_index < len(TRIP_TIMELINE) - 1:
            st.session_state.simulation_index += 1
        elif st.session_state.get("trip_auto_loop", True):
            st.session_state.simulation_index = 0
            st.session_state.trip_loop_count = st.session_state.get("trip_loop_count", 0) + 1
        else:
            st.session_state.simulation_running = False
        st.rerun()


active_mode, playback_speed, sandbox_values = render_sidebar_controls()


# ============================================================
# 7. MODE DISPATCH & COCKPIT / SIMULATOR RENDERING
# ============================================================

if active_mode == "🎮 3D Road Simulator (road.py)":
    st.markdown(
        """
        <div class="mode-hero-banner">
            <div class="mode-hero-title">🎮 3D WebGL Road Safety Simulator (Three.js Physics Engine)</div>
            <div class="mode-hero-desc">
                Interactive real-time 3D road environment featuring dynamic pedestrian and animal crossing animations, 
                traffic car-following with anti-clipping ACC, multi-camera cockpit/BEV views, and adjustable cruise speed controls.
            </div>
            <div class="mode-pillars-grid">
                <div class="mode-pillar-card">
                    <div class="mode-pillar-header">🎯 What This Tests</div>
                    <p class="mode-pillar-text">Dynamic spatial obstacle avoidance, realistic 4-legged trot gait, perpendicular sidewalk crossings.</p>
                </div>
                <div class="mode-pillar-card">
                    <div class="mode-pillar-header">🔬 Active Physics Models</div>
                    <p class="mode-pillar-text">Adaptive Cruise Control (7.5m hard buffer), Weather road friction (μ=0.75 dry to μ=0.45 storm).</p>
                </div>
                <div class="mode-pillar-card">
                    <div class="mode-pillar-header">🎮 Live Controls</div>
                    <p class="mode-pillar-text">Cruise Speed Slider (10–130 km/h), Quick ±10 km/h buttons, WASD / Arrow keys & Cockpit camera mode.</p>
                </div>
            </div>
            <div class="mode-hero-tags">
                <span class="mode-tag">● Three.js WebGL Engine</span>
                <span class="mode-tag">● Cockpit POV Driver Eye View</span>
                <span class="mode-tag">● 4-Legged Diagonal Trot Kinematics</span>
                <span class="mode-tag">● High-Altitude Cloud Shading</span>
                <span class="mode-tag">● Standalone Mode (`road.py`)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    info_c1, info_c2, info_c3 = st.columns(3)
    with info_c1:
        st.info("🎮 **Controls:** Use onscreen buttons, Speed slider, or WASD / Arrow keys.")
    with info_c2:
        st.success("🟢 **Status:** 3D Canvas Graphics & Physics Active")
    with info_c3:
        st.warning("⚡ **Direct Access:** Also available via `streamlit run road.py`")

    canvas_h = sandbox_values.get("sim_height", 850)
    render_road_simulation_component(height=canvas_h)

elif active_mode == "👁️ Live Vision & YOLO Perception":
    st.markdown(
        """
        <div class="mode-hero-banner">
            <div class="mode-hero-title">👁️ Computer Vision & YOLOv8 Neural Perception Suite</div>
            <div class="mode-hero-desc">
                Real-time video & animation perception stream: Deep learning object detection (YOLOv8) automatically tracks 
                dynamic objects frame-by-frame, combining OpenCV Hough lane tracking, monocular distance estimation, and closing velocity dynamics.
            </div>
            <div class="mode-pillars-grid">
                <div class="mode-pillar-card">
                    <div class="mode-pillar-header">🎯 What This Tests</div>
                    <p class="mode-pillar-text">Continuous video object detection, real-time bounding box tracking, and lane departure warnings.</p>
                </div>
                <div class="mode-pillar-card">
                    <div class="mode-pillar-header">🔬 Active AI Models</div>
                    <p class="mode-pillar-text">YOLOv8n neural weights, Canny edge detection, Hough Line Transform, and monocular focal depth estimation.</p>
                </div>
                <div class="mode-pillar-card">
                    <div class="mode-pillar-header">🎮 Live Controls</div>
                    <p class="mode-pillar-text">Auto-playing video animation, frame scrubber slider, video upload, webcam capture, and sensor fusion.</p>
                </div>
            </div>
            <div class="mode-hero-tags">
                <span class="mode-tag">● YOLOv8 Deep Learning</span>
                <span class="mode-tag">● OpenCV Hough Lane Assist</span>
                <span class="mode-tag">● Real-Time Video Frame Stepping</span>
                <span class="mode-tag">● Monocular Distance Model (d = f·H / h)</span>
                <span class="mode-tag">● Auto-Playing 20 FPS Stream</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    source_choice = sandbox_values.get("vision_source", "Preset Animated Driving Scenes")
    conf_val = sandbox_values.get("vision_conf", 0.35)
    speed_val = sandbox_values.get("vision_speed", 40.0)
    enable_lanes = sandbox_values.get("enable_lanes", True)
    enable_fusion = sandbox_values.get("enable_fusion", False)

    ego_state = EgoState(speed_kmh=speed_val, lane="center")
    frame_to_process = None
    override_boxes = None
    is_video_mode = False
    total_video_frames = 160
    current_frame_id = 0

    if source_choice == "Preset Animated Driving Scenes":
        is_video_mode = True
        total_video_frames = 160
        current_frame_id = st.session_state.vision_frame_idx

        sc_col1, sc_col2 = st.columns([1.2, 1.0])
        with sc_col1:
            sc_choice = st.selectbox(
                "Select Driving Scenario",
                [
                    "🚶 Urban Pedestrian Crossing (Center Lane Risk)",
                    "🚙 Highway Lead Vehicle Rapid Deceleration",
                    "🚧 Dual Hazard Pinch (Left Barrier + Right Cyclist)"
                ],
                key="vision_scene_select"
            )
            sc_key = "urban_pedestrian" if "Pedestrian" in sc_choice else ("highway_lead_vehicle" if "Highway" in sc_choice else "dual_hazard_pinch")
        with sc_col2:
            anim_status = "🟢 Auto-Playing" if st.session_state.vision_anim_running else "⏸️ Paused"
            frame_slider = st.slider(
                f"🎬 Timeline Frame: {st.session_state.vision_frame_idx}/160 ({anim_status})",
                min_value=0,
                max_value=160,
                value=st.session_state.vision_frame_idx,
                step=1,
                key="vision_frame_scrubber"
            )
            if frame_slider != st.session_state.vision_frame_idx and not st.session_state.vision_anim_running:
                st.session_state.vision_frame_idx = frame_slider
                current_frame_id = frame_slider

        # Generate the animated driving frame
        frame_to_process, override_boxes = generate_animated_driving_frame(
            scenario=sc_key,
            frame_idx=st.session_state.vision_frame_idx
        )

    elif source_choice == "Upload Dashcam Video / Image":
        uploaded_file = st.file_uploader(
            "Upload Dashcam Video or Image (MP4, AVI, MOV, JPG, PNG)",
            type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "webp"],
            key="vision_uploader"
        )
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                image_pil = Image.open(uploaded_file).convert("RGB")
                frame_to_process = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            else:
                # Video processing mode
                is_video_mode = True

                # Cache video to temp file if new upload
                if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
                    tfile.write(uploaded_file.read())
                    tfile.flush()
                    tfile.close()
                    st.session_state.video_temp_path = tfile.name
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.video_frame_idx = 0
                    st.session_state.video_anim_running = True

                temp_vid_path = st.session_state.get("video_temp_path")
                if temp_vid_path and os.path.exists(temp_vid_path):
                    cap = cv2.VideoCapture(temp_vid_path)
                    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    current_frame_id = st.session_state.video_frame_idx

                    v_col1, v_col2 = st.columns([1.2, 1.0])
                    with v_col1:
                        status_str = "🟢 Auto-Playing Video" if st.session_state.video_anim_running else "⏸️ Paused"
                        st.caption(f"🎥 Video Loaded: {total_video_frames} frames ({fps} FPS) — **{status_str}**")
                    with v_col2:
                        v_slider = st.slider(
                            f"🎬 Video Frame Scrubber ({st.session_state.video_frame_idx}/{max(0, total_video_frames - 1)})",
                            min_value=0,
                            max_value=max(0, total_video_frames - 1),
                            value=min(st.session_state.video_frame_idx, max(0, total_video_frames - 1)),
                            step=1,
                            key="vid_scrubber_slider"
                        )
                        if v_slider != st.session_state.video_frame_idx and not st.session_state.video_anim_running:
                            st.session_state.video_frame_idx = v_slider
                            current_frame_id = v_slider

                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_frame_idx)
                    ret, frame_read = cap.read()
                    cap.release()
                    if ret:
                        frame_to_process = frame_read
                    else:
                        st.error("Could not extract frame from uploaded video.")
        else:
            st.info("👆 Please upload a dashcam video clip (.mp4 / .avi / .mov) or image above to start real-time YOLO detection.")

    elif source_choice == "Live Camera Snapshot (Webcam)":
        cam_snap = st.camera_input("Capture Dashcam Frame from Webcam", key="webcam_capture")
        if cam_snap is not None:
            bytes_data = cam_snap.getvalue()
            frame_to_process = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if frame_to_process is not None:
        engine = get_cached_vision_engine(model_name="yolov8n.pt", enable_lanes=enable_lanes)
        annotated_frame, current_hazards, decision, lane_info = engine.process_frame(
            frame_to_process, ego_state=ego_state, conf_threshold=conf_val, override_boxes=override_boxes
        )

        radar_sim = None
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

        decision, kinematics, _ = evaluate_scene(
            hazards=current_hazards,
            ego_state=ego_state,
            lane_info=lane_info,
            radar_hazards=radar_sim
        )

        # Dynamic Auto-Telemetry Streaming into Blackbox & Charts
        primary_hazard = current_hazards[0] if current_hazards else HazardEvent(type=HazardType.CLEAR)
        v_step_tag = f"Frame {current_frame_id}"
        
        # Log to live rolling metrics if new or on periodic stride (smooth graph updates)
        if (current_frame_id % 3 == 0) or (primary_hazard.type != HazardType.CLEAR and (len(st.session_state.event_history) == 0 or st.session_state.event_history[-1].get("Step") != v_step_tag)):
            record_event(
                event=primary_hazard,
                risk=decision.risk,
                action=decision.action,
                speed_kmh=ego_state.speed_kmh,
                dt_seconds=0.10
            )
            st.session_state.event_history.append(build_history_entry(primary_hazard, decision, v_step_tag))
            if len(st.session_state.event_history) > 100:
                st.session_state.event_history.pop(0)

        # Render Top HUD
        render_top_hud(ego_state, decision, kinematics)

        # Central Split Cockpit (Identical richness to Live Trip)
        c_vis, c_bev = st.columns([1.1, 1.0])
        with c_vis:
            st.markdown('<div class="section-label">📸 AR Cockpit Vision Perception (OpenCV + YOLO)</div>', unsafe_allow_html=True)
            rgb_disp = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_disp, use_container_width=True)

            if lane_info.departure_warning:
                st.error(f"⚠️ **{lane_info.warning_message}** (Offset: {lane_info.offset_from_center_px:.1f}px)")
            elif lane_info.left_line and lane_info.right_line:
                st.success("🟢 **Lane Keeping Assist:** Vehicle Centered in Host Lane")

            # Detected Targets Breakdown
            st.markdown("##### 🎯 Detected Targets Breakdown")
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
                st.success("🟢 No active roadway hazards detected in camera view.")

        with c_bev:
            st.markdown('<div class="section-label">🗺️ Synchronized 2D Bird\'s-Eye View (BEV)</div>', unsafe_allow_html=True)
            render_bev_road_component(current_hazards, decision, ego_state.speed_kmh)
            render_sector_cards(current_hazards)

            p_lvl = getattr(decision, "priority_level", 1)
            p_names = {1: "Nominal Cruising", 2: "Lateral Maneuver", 3: "Active Caution", 4: "Urgent Avoidance", 5: "Emergency Intervention"}
            p_badge = f'<span style="float: right; font-size: 0.75rem; background: rgba(255,255,255,0.15); padding: 4px 8px; border-radius: 6px;">Priority {p_lvl}/5: {p_names.get(p_lvl, "Standard")}</span>'

            st.markdown(
                f'<div class="act-box {action_class_for(decision.action)}">'
                f'🚦 {decision.action.replace("_", " ")}'
                f'{p_badge}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.info(f"💡 **AI Rationale:** {decision.reason}")

            # Kinematic Safety Telemetry metrics in Vision Mode
            st.markdown("#### 📐 Kinematic Safety Telemetry:")
            kn1, kn2, kn3 = st.columns(3)
            with kn1:
                st.metric("Reaction Dist", f"{kinematics['reaction_dist_m']} m")
            with kn2:
                st.metric("Braking Dist", f"{kinematics['braking_dist_m']} m")
            with kn3:
                decel_disp = f"{kinematics['required_decel_ms2']} m/s²" if kinematics['required_decel_ms2'] is not None else "0.0 m/s²"
                st.metric("Required Decel", decel_disp)

            if kinematics.get("safety_margin_m") is not None:
                margin = kinematics["safety_margin_m"]
                margin_txt = f"{margin:+.1f} m"
                if margin > 5.0:
                    st.caption(f"🛡️ **Safety Stopping Margin:** `{margin_txt}` (Adequate Buffer)")
                elif margin >= 0:
                    st.caption(f"⚠️ **Safety Stopping Margin:** `{margin_txt}` (Tight Threshold)")
                else:
                    st.caption(f"🚨 **Safety Stopping Margin:** `{margin_txt}` (Negative Margin - Emergency Intervention Triggered)")

        # Performance KPIs, Live Graphs & Blackbox Audit History
        render_analytics(st.session_state.event_history)

        # Auto-advance for Preset Animated Driving Scenes
        if source_choice == "Preset Animated Driving Scenes" and st.session_state.vision_anim_running:
            time.sleep(0.06)
            st.session_state.vision_frame_idx = (st.session_state.vision_frame_idx + 3) % 160
            st.rerun()

        # Auto-advance for Uploaded Video Streams
        elif source_choice == "Upload Dashcam Video / Image" and is_video_mode and st.session_state.video_anim_running:
            time.sleep(0.04)
            step_stride = 2 if total_video_frames > 60 else 1
            st.session_state.video_frame_idx = (st.session_state.video_frame_idx + step_stride) % max(1, total_video_frames)
            st.rerun()

else:
    # ------------------------------------------------------------
    # SENSOR INGESTION & KINEMATICS EVALUATION
    # ------------------------------------------------------------
    current_hazards: List[HazardEvent] = []
    ego_state = EgoState(speed_kmh=40.0)
    step_desc = ""
    step_id: Any = 1

    if active_mode == "🚗 Live Trip Timeline":
        st.markdown(
            f"""
            <div class="mode-hero-banner">
                <div class="mode-hero-title">🚗 Sequential Autonomous Drive Simulation (Live Timeline)</div>
                <div class="mode-hero-desc">
                    Simulates a continuous 8-step driving journey across highway, urban, and adverse conditions. 
                    Watch the autonomous vehicle automatically adapt its speed, execute emergency braking for pedestrians, 
                    navigate construction barriers, handle dense fog, and recover from sensor dropouts.
                </div>
                <div class="mode-pillars-grid">
                    <div class="mode-pillar-card">
                        <div class="mode-pillar-header">🎯 What This Tests</div>
                        <p class="mode-pillar-text">Dynamic speed adaptation, pedestrian crossing reactions, obstacle evasion, and sensor failsafes.</p>
                    </div>
                    <div class="mode-pillar-card">
                        <div class="mode-pillar-header">🔬 Active Kinematics</div>
                        <p class="mode-pillar-text">Dynamic reaction distance, stopping distance envelopes (d_stop), TTC thresholds, and safety margins.</p>
                    </div>
                    <div class="mode-pillar-card">
                        <div class="mode-pillar-header">🎮 Live Controls</div>
                        <p class="mode-pillar-text">Auto-driving loop, play/pause toggles, step scrubbing, and automatic 0–100% safety scoring.</p>
                    </div>
                </div>
                <div class="mode-hero-tags">
                    <span class="mode-tag">● Step {st.session_state.simulation_index + 1} of {len(TRIP_TIMELINE)}</span>
                    <span class="mode-tag">● Dynamic Stopping Envelopes</span>
                    <span class="mode-tag">● Real-Time Safety Scoring (0–100%)</span>
                    <span class="mode-tag">● Continuous Blackbox Audit Log</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        timeline = SimulationEngine.get_trip_timeline()
        curr_step = timeline[st.session_state.simulation_index]
        step_id = curr_step.get("timestep", st.session_state.simulation_index + 1)
        step_desc = curr_step.get("description", "")
        ego_state = EgoState(speed_kmh=curr_step.get("ego_speed", 40.0))
        current_hazards = [HazardEvent.from_dict(ev) for ev in curr_step.get("events", [])]

    else:
        # ============================================================
        # COMBINED: 🔬 Scenarios & What-If Sandbox
        # ============================================================
        st.markdown(
            """
            <div class="mode-hero-banner">
                <div class="mode-hero-title">🔬 Scenario Catalog Explorer & What-If Sandbox</div>
                <div class="mode-hero-desc">
                    Combined testing suite: Choose from pre-configured benchmark safety edge-cases (Urban Pedestrian, Highway Braking, Fog, Sensor Gap) 
                    or switch to the Custom What-If Injector to test live parameter tuning, speed variations, and dual-hazard swerve conflict resolution.
                </div>
                <div class="mode-pillars-grid">
                    <div class="mode-pillar-card">
                        <div class="mode-pillar-header">🎯 What This Tests</div>
                        <p class="mode-pillar-text">Deterministic edge-case verification, lateral swerve vs in-lane braking arbitration, and custom obstacle geometry.</p>
                    </div>
                    <div class="mode-pillar-card">
                        <div class="mode-pillar-header">🔬 Active Decision Models</div>
                        <p class="mode-pillar-text">5-Tier Priority Hierarchy, Swerve Conflict Resolution Matrix, and degraded sensor fallback logic.</p>
                    </div>
                    <div class="mode-pillar-card">
                        <div class="mode-pillar-header">🎮 Live Controls</div>
                        <p class="mode-pillar-text">Switch between Preset Scenarios and What-If Injector via top tabs; adjust speed, distance, and secondary obstacles.</p>
                    </div>
                </div>
                <div class="mode-hero-tags">
                    <span class="mode-tag">● Benchmark Scenario Catalog</span>
                    <span class="mode-tag">● Custom Parameter Injector</span>
                    <span class="mode-tag">● Swerve Conflict Matrix</span>
                    <span class="mode-tag">● Instant Stopping Margin Math</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        sub_tab = sandbox_values.get("sub_mode", "📚 Preset Benchmark Catalog")

        if sub_tab == "📚 Preset Benchmark Catalog":
            sc_data = SimulationEngine.get_scenario(st.session_state.selected_scenario)
            ego_state = sc_data["ego_state"]
            step_desc = f"📚 Preset Benchmark: {sc_data['title']} — {sc_data['description']}"
            current_hazards = sc_data["events"]
            step_id = st.session_state.selected_scenario
        else:
            ego_state = EgoState(speed_kmh=sandbox_values.get("speed", 45.0))
            step_desc = f"🛠️ What-If Custom Injection: {sandbox_values.get('hazard_type')} at {sandbox_values.get('distance')}m ({sandbox_values.get('position')}) with host speed {ego_state.speed_kmh:.0f} km/h."
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
    decision, kinematics, _ = evaluate_scene(current_hazards, ego_state)
    primary_hazard = current_hazards[0] if current_hazards else HazardEvent(type=HazardType.CLEAR)

    if active_mode == "🚗 Live Trip Timeline":
        loop_cnt = st.session_state.get("trip_loop_count", 0)
        step_key = f"loop_{loop_cnt}_step_{st.session_state.simulation_index}_{step_id}"
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
            step_label = f"Step {st.session_state.simulation_index + 1}" if loop_cnt == 0 else f"Step {st.session_state.simulation_index + 1} (Loop {loop_cnt + 1})"
            st.session_state.event_history.append(build_history_entry(primary_hazard, decision, step_label))

    # ============================================================
    # 7. MAIN COCKPIT HUD
    # ============================================================
    render_top_hud(ego_state, decision, kinematics)

    # ============================================================
    # 8. CENTRAL SPLIT COCKPIT (BEV MAP + DECISION ENGINE)
    # ============================================================
    render_main_cockpit(current_hazards, decision, ego_state, kinematics, step_desc)

    # Optional Blackbox Logger Button for Scenario / What-If
    if active_mode == "🔬 Scenarios & What-If Sandbox":
        sc_log_col1, sc_log_col2 = st.columns([0.6, 0.4])
        with sc_log_col1:
            if st.button("💾 Record Scenario Decision to Blackbox Audit Log", use_container_width=True, key="record_scenario_audit_btn"):
                for h in current_hazards:
                    record_event(
                        event=h,
                        risk=decision.risk,
                        action=decision.action,
                        speed_kmh=ego_state.speed_kmh,
                        dt_seconds=3.0
                    )
                log_tag = f"Scenario_{st.session_state.selected_scenario}" if sandbox_values.get("sub_mode") == "📚 Preset Benchmark Catalog" else f"WhatIf_{len(st.session_state.event_history) + 1}"
                st.session_state.event_history.append(build_history_entry(primary_hazard, decision, log_tag))
                st.toast("✅ Scenario decision logged to Blackbox Audit!")

    # ============================================================
    # 9. PERFORMANCE KPIS, CHARTS & BLACKBOX AUDIT LOG
    # ============================================================
    render_analytics(st.session_state.event_history)

    # ============================================================
    # 10. AUTOMATIC TIMELINE PLAYBACK LOOP
    # ============================================================
    run_timeline_loop(playback_speed)
