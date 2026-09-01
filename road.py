# ============================================================
# AI ROAD SAFETY SIMULATOR • AUTONOMOUS COCKPIT
# Integrated with brain.py Decision & metrics.py Performance Tracker
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

from brain import (
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
    SectorOccupancy
)
from metrics import (
    record_event,
    get_metrics,
    reset_metrics,
    get_event_history
)


# ============================================================
# 1. CACHED SIMULATION ASSET LOADER
# ============================================================

@st.cache_data
def get_cached_simulation_html() -> str:
    """Caches simulation.html in memory for instant subsequent renders."""
    html_file = Path(__file__).parent / "simulation.html"
    if not html_file.exists():
        return ""
    return html_file.read_text(encoding="utf-8")


def render_road_simulation_component(
    height: int = 780,
    initial_speed_kmh: float = 50.0,
    sensor_health: str = "active",
    initial_scenario: str = "normal"
) -> None:
    """
    Renders the interactive 3D/Canvas Road Safety Simulator iframe component
    tightly integrated with the brain.py decision engine.
    """
    html_code = get_cached_simulation_html()
    if not html_code:
        st.error(
            "⚠️ `simulation.html` was not found. "
            "Please ensure `simulation.html` exists in the same directory as `road.py`."
        )
        return

    # Inject runtime initialization parameters into HTML head
    init_script = f"""
    <script>
        window.INITIAL_SPEED = {initial_speed_kmh};
        window.INITIAL_SENSOR_STATUS = '{sensor_health}';
        window.INITIAL_SCENARIO = '{initial_scenario}';
    </script>
    """
    injected_html = html_code.replace("<head>", f"<head>{init_script}", 1)

    components.html(
        injected_html,
        height=height,
        scrolling=False
    )


# ============================================================
# 2. STANDALONE COCKPIT & METRICS RUNNER
# ============================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Road Safety Simulation • Autonomous Cockpit",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Cybernetic cockpit CSS
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            .stApp {
                background-color: #06090e;
                color: #e2e8f0;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            .block-container {
                padding: 0.8rem 1.2rem !important;
                max-width: 100% !important;
            }

            iframe {
                width: 100% !important;
                border-radius: 14px;
                box-shadow: 0 12px 35px rgba(0,0,0,0.6);
                border: 1px solid #1e293b !important;
            }

            .badge-act-continue { background: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-slow { background: #78350f; color: #fbbf24; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-brake { background: #881337; color: #fb7185; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-stop { background: #7f1d1d; color: #f87171; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-swerve { background: #1e1b4b; color: #818cf8; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }

            .safety-score-card {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .score-val-high { color: #10b981; font-size: 2.2rem; font-weight: 850; }
            .score-val-med { color: #f59e0b; font-size: 2.2rem; font-weight: 850; }
            .score-val-low { color: #ef4444; font-size: 2.2rem; font-weight: 850; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Preset Scenarios Catalog with exact brain.py models
    PRESET_SCENARIOS: Dict[str, List[HazardEvent]] = {
        "🟢 Nominal Cruising": [
            HazardEvent(type=HazardType.CLEAR)
        ],
        "🧍 Pedestrian Crossing": [
            HazardEvent(type=HazardType.PEDESTRIAN, subtype="pedestrian", position=Position.FRONT, distance=6.0, confidence=0.98, relative_speed_kmh=40.0)
        ],
        "🐕 Animal Crossing": [
            HazardEvent(type=HazardType.ANIMAL, subtype="animal", position=Position.FRONT, distance=7.0, confidence=0.95, relative_speed_kmh=40.0)
        ],
        "🚧 Road Obstruction (Left Lane)": [
            HazardEvent(type=HazardType.STATIC_OBSTACLE, subtype="construction_barrier", position=Position.LEFT, distance=12.0, confidence=0.92)
        ],
        "🚗 Lead Vehicle Deceleration": [
            HazardEvent(type=HazardType.VEHICLE, subtype="sedan", position=Position.FRONT, distance=16.0, confidence=0.96, relative_speed_kmh=35.0)
        ],
        "🌧️ Heavy Rain (Degraded Sensors)": [
            HazardEvent(type=HazardType.VEHICLE, subtype="truck", position=Position.FRONT, distance=18.0, confidence=0.40, sensor_status=SensorStatus.DEGRADED)
        ],
        "⚠️ Swerve Conflict (Dual Lateral Hazards)": [
            HazardEvent(type=HazardType.STATIC_OBSTACLE, subtype="debris", position=Position.LEFT, distance=10.0),
            HazardEvent(type=HazardType.CYCLIST, subtype="cyclist", position=Position.RIGHT, distance=8.0)
        ]
    }

    # Session state for scenario selection
    if "road_selected_scenario" not in st.session_state:
        st.session_state.road_selected_scenario = "🟢 Nominal Cruising"

    # Sidebar: Brain controls and live parameters
    with st.sidebar:
        st.title("🧠 AI Brain Telemetry")
        st.caption("Real-Time Decision Engine & Metrics")

        selected_scenario_name = st.selectbox(
            "Select Scenario Preset",
            list(PRESET_SCENARIOS.keys()),
            index=list(PRESET_SCENARIOS.keys()).index(st.session_state.road_selected_scenario) if st.session_state.road_selected_scenario in PRESET_SCENARIOS else 0,
            key="sb_scenario_select"
        )
        st.session_state.road_selected_scenario = selected_scenario_name

        ego_speed = st.slider("Host Vehicle Speed (km/h)", min_value=10, max_value=120, value=50, step=5)
        sensor_status_str = st.selectbox("Sensor Health", ["active", "degraded", "failed"])
        current_lane_str = st.selectbox("Host Lane Position", ["right", "center", "left"])

        st.divider()
        st.markdown("### 🎛️ Evasive & Safety Tuning")
        t_react = st.slider("Driver/ADAS Reaction Time (s)", min_value=0.5, max_value=2.5, value=1.2, step=0.1)
        mu_friction = st.slider("Road Friction Coefficient (μ)", min_value=0.2, max_value=0.9, value=0.75, step=0.05)

        st.divider()
        if st.button("↻ Reset Trip Metrics", use_container_width=True):
            reset_metrics()
            st.toast("Trip metrics & history have been reset.", icon="🔄")

    # Quick Scenario Action Ribbon
    st.markdown("##### ⚡ Quick Scenario Triggers:")
    sc_cols = st.columns(7)
    scenario_buttons = [
        ("🟢 Cruising", "🟢 Nominal Cruising"),
        ("🧍 Pedestrian", "🧍 Pedestrian Crossing"),
        ("🐕 Animal", "🐕 Animal Crossing"),
        ("🚧 Obstacle", "🚧 Road Obstruction (Left Lane)"),
        ("🚗 Collision", "🚗 Lead Vehicle Deceleration"),
        ("🌧️ Rain", "🌧️ Heavy Rain (Degraded Sensors)"),
        ("⚠️ Swerve Conflict", "⚠️ Swerve Conflict (Dual Lateral Hazards)")
    ]
    for i, (label, s_name) in enumerate(scenario_buttons):
        with sc_cols[i]:
            if st.button(label, use_container_width=True, key=f"quick_sc_{i}"):
                st.session_state.road_selected_scenario = s_name
                st.rerun()

    # Run real-time Brain evaluation
    active_hazards = PRESET_SCENARIOS[st.session_state.road_selected_scenario]
    ego_state = EgoState(
        speed_kmh=float(ego_speed),
        lane=current_lane_str,
        sensor_status=SensorStatus.from_value(sensor_status_str)
    )

    decision, kinematics, sectors = evaluate_scene(
        hazards=active_hazards,
        ego_state=ego_state
    )

    # Ingest event into metrics tracker
    primary_hazard = active_hazards[0] if active_hazards else HazardEvent(type=HazardType.CLEAR)
    pos_str = getattr(primary_hazard.position, "value", str(primary_hazard.position))
    record_event(
        event=primary_hazard,
        risk=decision.risk,
        action=decision.action,
        speed_kmh=float(ego_speed),
        dt_seconds=2.0,
        reason=decision.reason,
        distance_m=primary_hazard.distance,
        position=pos_str
    )

    # Determine CSS styling for decision action
    act_str = str(decision.action).upper()
    if "STOP" in act_str:
        act_css = "badge-act-stop"
    elif "BRAKE" in act_str:
        act_css = "badge-act-brake"
    elif "SLOW" in act_str:
        act_css = "badge-act-slow"
    elif "MOVE" in act_str:
        act_css = "badge-act-swerve"
    else:
        act_css = "badge-act-continue"

    # Top Brain Telemetry Ribbon
    col_hud1, col_hud2, col_hud3, col_hud4, col_hud5 = st.columns([1.2, 1.0, 1.0, 1.2, 2.2])

    with col_hud1:
        st.markdown(
            f'<div class="{act_css}" style="text-align:center;font-size:0.95rem;padding:6px 12px;">'
            f'🚦 {decision.action.replace("_", " ")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_hud2:
        risk_icon = "🔴" if decision.risk in ["HIGH", "CRITICAL"] else ("🟡" if decision.risk == "MEDIUM" else "🟢")
        st.metric("Risk Level", f"{risk_icon} {decision.risk}")

    with col_hud3:
        ttc_str = f"⚡ {decision.ttc_seconds}s" if decision.ttc_seconds else "🛡️ Safe"
        st.metric("Time-To-Collision", ttc_str)

    with col_hud4:
        st.metric("Stopping Dist", f"{kinematics.total_stopping_dist_m} m")

    with col_hud5:
        st.info(f"💡 **AI Rationale:** {decision.reason}")

    # Roadway Sector Clearance Ribbon
    sec_c1, sec_c2, sec_c3 = st.columns(3)
    with sec_c1:
        left_status = "🟢 Clear" if sectors.is_left_clear else f"🔴 Occupied ({len(sectors.left_hazards)} Hazard)"
        st.caption(f"◀️ **Left Sector:** `{left_status}`")
    with sec_c2:
        front_status = "🟢 Clear" if sectors.is_front_clear else f"🔴 Hazard Ahead ({primary_hazard.distance}m)" if primary_hazard.distance else "🔴 Occupied"
        st.caption(f"⬆️ **Front Corridor:** `{front_status}`")
    with sec_c3:
        right_status = "🟢 Clear" if sectors.is_right_clear else f"🔴 Occupied ({len(sectors.right_hazards)} Hazard)"
        st.caption(f"▶️ **Right Sector:** `{right_status}`")

    scenario_map = {
        "🟢 Nominal Cruising": "normal",
        "🧍 Pedestrian Crossing": "human",
        "🐕 Animal Crossing": "animal",
        "🚧 Road Obstruction (Left Lane)": "obstacle",
        "🚗 Lead Vehicle Deceleration": "collision",
        "🌧️ Heavy Rain (Degraded Sensors)": "rain",
        "⚠️ Swerve Conflict (Dual Lateral Hazards)": "obstacle"
    }
    init_scen = scenario_map.get(st.session_state.road_selected_scenario, "normal")


    # Render interactive 3D Simulator canvas
    render_road_simulation_component(
        height=780,
        initial_speed_kmh=float(ego_speed),
        sensor_health=sensor_status_str,
        initial_scenario=init_scen
    )

    # ============================================================
    # 3. TRIP PERFORMANCE & SAFETY ANALYTICS (METRICS INTEGRATION)
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Trip Performance & Safety Analytics")

    live_metrics = get_metrics()
    score = live_metrics.get("safety_score", 100)
    score_cls = "score-val-high" if score >= 85 else ("score-val-med" if score >= 65 else "score-val-low")

    tab_summary, tab_history, tab_diagnostics = st.tabs([
        "📈 Trip Overview & KPIs",
        "📋 Blackbox Audit Log",
        "🔬 Sensor & Reliability Diagnostics"
    ])

    with tab_summary:
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns([1.3, 1.0, 1.0, 1.0, 1.0, 1.1])

        with m_col1:
            st.markdown(
                f"""
                <div class="safety-score-card">
                    <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Safety Score</div>
                    <div class="{score_cls}">{score}%</div>
                    <div style="font-size:0.75rem;color:#64748b;">Autonomous Drive Quality</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m_col2:
            st.metric("Total Distance", f"{live_metrics['trip_distance_km']:.2f} km")
            st.metric("Total Events", live_metrics["total_events"])

        with m_col3:
            st.metric("Hazards Detected", live_metrics["hazards_detected"])
            st.metric("High Risk Events", live_metrics["high_risk_events"])

        with m_col4:
            st.metric("Warning Actions", live_metrics["warnings_count"])
            st.metric("Emergency Brakes", live_metrics["brake_events"])

        with m_col5:
            conf_pct = int(round(live_metrics["average_confidence"] * 100))
            st.metric("Sensor Health", sensor_status_str.upper())
            st.metric("Avg Confidence", f"{conf_pct}%")

        with m_col6:
            margin = kinematics.safety_margin_m
            margin_str = f"{margin:+.1f} m" if margin is not None else "Safe"
            st.metric("Stopping Margin", margin_str)
            st.caption("Buffer between stop distance and hazard")

    with tab_history:
        history = get_event_history()
        if history:
            df_history = pd.DataFrame(history)
            st.dataframe(
                df_history,
                use_container_width=True,
                column_config={
                    "timestamp": "Time",
                    "event_type": "Event Hazard",
                    "position": "Sector",
                    "distance_m": "Distance",
                    "speed_kmh": "Speed",
                    "risk": "Risk Level",
                    "action": "Brain Decision",
                    "reason": "AI Rationale"
                },
                hide_index=True
            )

            # CSV Download
            csv_data = df_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Trip Audit Log (CSV)",
                data=csv_data,
                file_name="autonomous_trip_telemetry.csv",
                mime="text/csv"
            )
        else:
            st.info("No events recorded yet in this drive session.")

    with tab_diagnostics:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("#### 📡 Multi-Modal Perception Health")
            st.write(f"- **Camera Stream:** {'🟢 Nominal' if sensor_status_str == 'active' else ('🟡 Degraded' if sensor_status_str == 'degraded' else '🔴 Disconnected')}")
            st.write(f"- **Radar Telemetry:** {'🟢 Online (Long Range)' if sensor_status_str != 'failed' else '🔴 Offline'}")
            st.write(f"- **LiDAR Point Cloud:** {'🟢 Calibrated' if sensor_status_str == 'active' else '🟡 Partial Attenuation'}")
            st.write(f"- **Lane Keeping Assist (LKA):** {'🟢 Active Centering' if sectors.is_front_clear else '🟡 Lane Deviation Caution'}")

        with d_col2:
            st.markdown("#### 📐 Kinematic Stopping Envelope")
            st.write(f"- **Host Speed:** `{ego_speed} km/h` (`{kinematics.speed_ms} m/s`)")
            st.write(f"- **Reaction Distance:** `{kinematics.reaction_dist_m} m` (Reaction time: `{t_react}s`)")
            st.write(f"- **Braking Distance:** `{kinematics.braking_dist_m} m` (Road friction: `μ={mu_friction}`)")
            st.write(f"- **Total Stopping Distance:** `{kinematics.total_stopping_dist_m} m`")




