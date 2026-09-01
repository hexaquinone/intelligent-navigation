# ============================================================
# AI ROAD SAFETY SIMULATOR • AUTONOMOUS COCKPIT
# Integrated with brain.py Decision & Kinematics Engine
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
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
from metrics import record_event


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
    height: int = 950,
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
# 2. STANDALONE COCKPIT RUNNER
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

            .brain-hud-card {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 12px 18px;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 12px;
            }

            .badge-act-continue { background: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-slow { background: #78350f; color: #fbbf24; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-brake { background: #881337; color: #fb7185; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-stop { background: #7f1d1d; color: #f87171; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
            .badge-act-swerve { background: #1e1b4b; color: #818cf8; padding: 4px 10px; border-radius: 8px; font-weight: 750; font-size: 0.85rem; }
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

    # Sidebar: Brain controls and live parameters
    with st.sidebar:
        st.title("🧠 AI Brain Telemetry")
        st.caption("Real-Time Decision Engine & Kinematics")

        selected_scenario_name = st.selectbox(
            "Select Scenario Preset",
            list(PRESET_SCENARIOS.keys())
        )

        ego_speed = st.slider("Host Vehicle Speed (km/h)", min_value=10, max_value=120, value=50, step=5)
        sensor_status_str = st.selectbox("Sensor Health", ["active", "degraded", "failed"])
        current_lane_str = st.selectbox("Host Lane Position", ["right", "center", "left"])

        st.divider()
        st.markdown("### 🎛️ Evasive & Safety Tuning")
        t_react = st.slider("Driver/ADAS Reaction Time (s)", min_value=0.5, max_value=2.5, value=1.2, step=0.1)
        mu_friction = st.slider("Road Friction Coefficient (μ)", min_value=0.2, max_value=0.9, value=0.75, step=0.05)

    # Run real-time Brain evaluation
    active_hazards = PRESET_SCENARIOS[selected_scenario_name]
    ego_state = EgoState(
        speed_kmh=float(ego_speed),
        lane=current_lane_str,
        sensor_status=SensorStatus.from_value(sensor_status_str)
    )

    decision, kinematics, sectors = evaluate_scene(
        hazards=active_hazards,
        ego_state=ego_state
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

    scenario_map = {
        "🟢 Nominal Cruising": "normal",
        "🧍 Pedestrian Crossing": "human",
        "🐕 Animal Crossing": "animal",
        "🚧 Road Obstruction (Left Lane)": "obstacle",
        "🚗 Lead Vehicle Deceleration": "collision",
        "🌧️ Heavy Rain (Degraded Sensors)": "rain",
        "⚠️ Swerve Conflict (Dual Lateral Hazards)": "obstacle"
    }
    init_scen = scenario_map.get(selected_scenario_name, "normal")

    # Render interactive 3D Simulator canvas
    render_road_simulation_component(
        height=900,
        initial_speed_kmh=float(ego_speed),
        sensor_health=sensor_status_str,
        initial_scenario=init_scen
    )



