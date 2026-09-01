import streamlit as st
import pandas as pd
import time
import random
from datetime import datetime


# ============================================================
                       # PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Navigation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL APP
   ========================================================= */

.stApp {
    background-color: #0b0f14;
    color: #e8edf2;
}


/* =========================================================
   MAIN CONTENT
   ========================================================= */

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   HEADER
   ========================================================= */

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f5f7fa;
    margin-bottom: 0;
}

.subtitle {
    color: #8d99a6;
    font-size: 1rem;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border: none;
    height: 1px;
    background: #202833;
    margin: 1.5rem 0;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

div[data-testid="stMetric"] {
    background: #111720;
    border: 1px solid #202a35;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 110px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}

div[data-testid="stMetricLabel"] {
    color: #8995a3;
    font-size: 0.85rem;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #f5f7fa;
    font-size: 1.55rem;
    font-weight: 750;
}


/* =========================================================
   HAZARD CARD
   ========================================================= */

.hazard-box {
    background: #111720;
    border: 1px solid #2a3541;
    border-radius: 16px;
    padding: 24px;
    min-height: 250px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}


/* =========================================================
   DECISION CARD
   ========================================================= */

.decision-box {
    background: #111720;
    border: 1px solid #2a3541;
    border-radius: 16px;
    padding: 24px;
    min-height: 250px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}


/* =========================================================
   BIG ACTION
   ========================================================= */

.big-action {
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    padding: 24px;
    letter-spacing: 1px;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background-color: #080c11;
    border-right: 1px solid #202833;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f5f7fa;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #303b48;
    background-color: #151c25;
    color: #f1f5f9;
    font-weight: 650;
    padding: 0.65rem 1rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6b7785;
    background-color: #1c2530;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

div[data-testid="stDataFrame"] {
    border: 1px solid #202a35;
    border-radius: 12px;
    overflow: hidden;
}


/* =========================================================
   ALERT BOXES
   ========================================================= */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================================================
   SELECTBOX / INPUTS
   ========================================================= */

div[data-baseweb="select"] > div {
    background-color: #111720;
    border-color: #303b48;
    border-radius: 10px;
}


/* =========================================================
   PROGRESS BAR
   ========================================================= */

div[data-testid="stProgress"] > div {
    border-radius: 10px;
}


/* =========================================================
   PIPELINE CARDS
   ========================================================= */

.pipeline-card {
    background: #111720;
    border: 1px solid #202a35;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    min-height: 100px;
}

.pipeline-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #f1f5f9;
}

.pipeline-description {
    font-size: 0.75rem;
    color: #7f8b98;
    margin-top: 5px;
}


/* =========================================================
   SIMULATION AREA
   ========================================================= */

.simulation-container {
    background: #0e141b;
    border: 1px solid #26313d;
    border-radius: 18px;
    padding: 25px;
    margin: 10px 0 20px 0;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}


/* =========================================================
   VEHICLE DISPLAY
   ========================================================= */

.vehicle-display {
    text-align: center;
    font-size: 4rem;
    padding: 30px;
}


/* =========================================================
   STATUS BADGE
   ========================================================= */

.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background-color: #16202a;
    border: 1px solid #303c49;
    font-size: 0.8rem;
    font-weight: 700;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #66717e;
    font-size: 0.75rem;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)
# ============================================================
# DISPLAY HELPERS
# ============================================================

hazard_icons = {
    "pedestrian": "🚶",
    "vehicle": "🚙",
    "static_obstacle": "🌳",
    "clear": "🟢",
    "sensor_failure": "📡"
}


hazard_names = {
    "pedestrian": "Pedestrian",
    "vehicle": "Vehicle",
    "static_obstacle": "Static Obstacle",
    "clear": "Clear Environment",
    "sensor_failure": "Sensor Failure"
}
# ============================================================
# SIMULATION ENGINE
# ============================================================

SIMULATION_EVENTS = [

    {
        "id": 1,
        "type": "clear",
        "subtype": None,
        "position": "front",
        "distance": None,
        "confidence": 1.0,
        "sensor_status": "active"
    },

    {
        "id": 2,
        "type": "vehicle",
        "subtype": None,
        "position": "front",
        "distance": 15,
        "confidence": 0.94,
        "sensor_status": "active"
    },

    {
        "id": 3,
        "type": "pedestrian",
        "subtype": None,
        "position": "front",
        "distance": 6,
        "confidence": 0.98,
        "sensor_status": "active"
    },

    {
        "id": 4,
        "type": "static_obstacle",
        "subtype": "tree",
        "position": "left",
        "distance": 10,
        "confidence": 0.90,
        "sensor_status": "active"
    },

    {
        "id": 5,
        "type": "static_obstacle",
        "subtype": "building",
        "position": "right",
        "distance": 12,
        "confidence": 0.92,
        "sensor_status": "active"
    },

    {
        "id": 6,
        "type": "sensor_failure",
        "subtype": None,
        "position": None,
        "distance": None,
        "confidence": 0.0,
        "sensor_status": "failed"
    },

    {
        "id": 7,
        "type": "clear",
        "subtype": None,
        "position": "front",
        "distance": None,
        "confidence": 1.0,
        "sensor_status": "active"
    }
]


# ============================================================
# DECISION ENGINE
# ============================================================
#
# TEMPORARY LOCAL VERSION
#
# Later Nano's actual decision engine can replace this.
#
# ============================================================

def generate_decision(event):

    hazard_type = event["type"]

    position = event["position"]

    distance = event["distance"]


    # --------------------------------------------------------
    # CLEAR ENVIRONMENT
    # --------------------------------------------------------

    if hazard_type == "clear":

        return {
            "risk": "LOW",
            "action": "CONTINUE",
            "reason": "No hazards are currently detected."
        }


    # --------------------------------------------------------
    # SENSOR FAILURE
    # --------------------------------------------------------

    if hazard_type == "sensor_failure":

        return {
            "risk": "UNCERTAIN",
            "action": "SLOW_DOWN",
            "reason": (
                "Sensor data is unavailable, reducing "
                "environmental awareness. The vehicle "
                "slows down as a precaution."
            )
        }


    # --------------------------------------------------------
    # DISTANCE BASED RISK
    # --------------------------------------------------------

    if distance is not None:

        if distance < 8:

            risk = "HIGH"

        elif distance <= 20:

            risk = "MEDIUM"

        else:

            risk = "LOW"

    else:

        risk = "UNCERTAIN"


    # --------------------------------------------------------
    # FRONT HAZARD
    # --------------------------------------------------------

    if position == "front":

        if risk == "HIGH":

            action = "BRAKE"

        elif risk == "MEDIUM":

            action = "SLOW_DOWN"

        else:

            action = "CONTINUE"


    # --------------------------------------------------------
    # LEFT HAZARD
    # --------------------------------------------------------

    elif position == "left":

        if risk == "HIGH":

            action = "SLOW_DOWN"

        else:

            action = "MOVE_RIGHT"


    # --------------------------------------------------------
    # RIGHT HAZARD
    # --------------------------------------------------------

    elif position == "right":

        if risk == "HIGH":

            action = "SLOW_DOWN"

        else:

            action = "MOVE_LEFT"


    else:

        action = "SLOW_DOWN"


    # --------------------------------------------------------
    # HAZARD NAME
    # --------------------------------------------------------

    if hazard_type == "pedestrian":

        hazard_name = "Pedestrian"

    elif hazard_type == "vehicle":

        hazard_name = "Vehicle"

    elif hazard_type == "static_obstacle":

        hazard_name = event["subtype"].title()

    else:

        hazard_name = hazard_type.replace(
            "_",
            " "
        ).title()


    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    if distance is not None:

        reason = (
            f"{hazard_name} detected "
            f"{position} at {distance}m. "
            f"The system classifies this as "
            f"{risk.lower()} risk."
        )

    else:

        reason = (
            f"{hazard_name} detected. "
            f"The system recommends {action.replace('_', ' ').lower()}."
        )


    return {
        "risk": risk,
        "action": action,
        "reason": reason
    }


# ============================================================
# SESSION STATE
# ============================================================

if "simulation_running" not in st.session_state:

    st.session_state.simulation_running = False


if "simulation_index" not in st.session_state:

    st.session_state.simulation_index = 0


if "event_history" not in st.session_state:

    st.session_state.event_history = []


if "trip_started" not in st.session_state:

    st.session_state.trip_started = False


# ============================================================
# CURRENT EVENT
# ============================================================

event = SIMULATION_EVENTS[
    st.session_state.simulation_index
]


decision = generate_decision(event)
# ============================================================
# RECORD CURRENT EVENT
# ============================================================

current_event_id = event["id"]


if (
    len(st.session_state.event_history) == 0
    or
    st.session_state.event_history[-1]["ID"]
    != current_event_id
):

    history_entry = {

        "ID":
            event["id"],

        "Event":
            event["id"],

        "Time":
            datetime.now().strftime("%H:%M:%S"),

        "Hazard":
            (
                hazard_names.get(
                    event["type"],
                    event["type"]
                )
            ),

        "Position":
            (
                event["position"].title()
                if event["position"]
                else "--"
            ),

        "Distance":
            (
                f"{event['distance']} m"
                if event["distance"] is not None
                else "--"
            ),

        "Confidence":
            f"{event['confidence'] * 100:.0f}%",

        "Risk":
            decision["risk"],

        "Action":
            decision["action"]

    }


    st.session_state.event_history.append(
        history_entry
    )




# ============================================================
                  # TRIP METRICS
# ============================================================
#
# Temporary values.
# Later these can come from the simulation.
#
# ============================================================

trip_distance = 4.8
hazard_count = 3
warning_count = 2
brake_count = 1
average_confidence = 0.92


# ============================================================
# HAZARD DISPLAY HELPERS
# ============================================================

hazard_icons = {

    "pedestrian": "🚶",

    "vehicle": "🚙",

    "static_obstacle": "🌳",

    "clear": "🟢",

    "sensor_failure": "📡"

}


hazard_names = {

    "pedestrian": "Pedestrian",

    "vehicle": "Vehicle",

    "static_obstacle": "Static Obstacle",

    "clear": "Clear Environment",

    "sensor_failure": "Sensor Failure"

}


# ============================================================
                          # HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚗 Intelligent Navigation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Decision-Support & Hazard Awareness System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
                            # SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Control Panel")

    monitoring = st.toggle(
        "Enable monitoring",
        value=True
    )

    st.subheader("System")

    st.write("🌍 Environmental Awareness")

    if event is not None:
        st.success("CONNECTED")
    else:
        st.warning("WAITING")


    st.write("🧠 Decision Engine")

    if decision is not None:
        st.success("CONNECTED")
    else:
        st.warning("WAITING")


    st.write("🖥️ Dashboard")

    st.success("ACTIVE")


    st.divider()

    st.subheader("Trip")

    st.metric(
        "Distance",
        f"{trip_distance:.1f} km"
    )


    st.divider()

    st.caption(
        "Detect → Understand → Assess → Decide → Explain"
    )

    st.header("⚙️ Control Panel")
    # ========================================================
    # SIMULATION CONTROLS
    # ========================================================

    st.subheader("🎮 Simulation")

    start_simulation = st.button(
        "▶️ Start Trip",
        use_container_width=True
    )

    pause_simulation = st.button(
        "⏸️ Pause",
        use_container_width=True
    )

    reset_simulation = st.button(
        "🔄 Reset",
        use_container_width=True
    )


    if start_simulation:

        st.session_state.simulation_running = True
        st.session_state.trip_started = True


    if pause_simulation:

        st.session_state.simulation_running = False


    if reset_simulation:

        st.session_state.simulation_running = False

        st.session_state.simulation_index = 0

        st.session_state.event_history = []

        st.session_state.trip_started = False

        st.rerun()


    st.write(
        f"Event {st.session_state.simulation_index + 1}"
        f" / {len(SIMULATION_EVENTS)}"
    )



# ============================================================
                    # TOP STATUS CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
                    # SYSTEM STATUS
# ------------------------------------------------------------

with col1:

    if monitoring:

        st.metric(
            "System Status",
            "🟢 ACTIVE"
        )

    else:

        st.metric(
            "System Status",
            "⚪ PAUSED"
        )


# ------------------------------------------------------------
                     # CURRENT RISK
# ------------------------------------------------------------

with col2:

    risk = decision["risk"]

    if risk == "LOW":

        display_risk = "🟢 LOW"

    elif risk == "MEDIUM":

        display_risk = "🟡 MEDIUM"

    elif risk == "HIGH":

        display_risk = "🔴 HIGH"

    else:

        display_risk = "⚪ UNCERTAIN"


    st.metric(
        "Current Risk",
        display_risk
    )


# ------------------------------------------------------------
                  # RECOMMENDED ACTION
# ------------------------------------------------------------

with col3:

    action = decision["action"]

    action_display = action.replace(
        "_",
        " "
    )

    st.metric(
        "Recommended Action",
        action_display
    )


# ------------------------------------------------------------
                          # CONFIDENCE
# ------------------------------------------------------------

with col4:

    confidence = event["confidence"]

    st.metric(
        "Detection Confidence",
        f"{confidence * 100:.0f}%"
    )


# ============================================================
                 # VEHICLE ENVIRONMENT VIEW
# ============================================================

st.markdown(
    '<div class="section-title">🗺️ Vehicle Environment</div>',
    unsafe_allow_html=True
)

st.caption(
    "Current hazard position relative to the vehicle"
)


left_col, front_col, right_col = st.columns(3)


# ============================================================
                          # LEFT
# ============================================================

with left_col:

    st.subheader("⬅️ LEFT")

    if event["position"] == "left":

        icon = hazard_icons.get(
            event["type"],
            "⚠️"
        )

        name = hazard_names.get(
            event["type"],
            event["type"]
        )

        st.warning(
            f"{icon} {name}"
        )

        if event["distance"] is not None:

            st.write(
                f"Distance: **{event['distance']} m**"
            )

    else:

        st.success("Clear")


# ============================================================
                           # FRONT
# ============================================================

with front_col:

    st.subheader("⬆️ FRONT")

    if event["position"] == "front":

        icon = hazard_icons.get(
            event["type"],
            "⚠️"
        )

        name = hazard_names.get(
            event["type"],
            event["type"]
        )

        st.warning(
            f"{icon} {name}"
        )

        if event["distance"] is not None:

            st.write(
                f"Distance: **{event['distance']} m**"
            )

    else:

        st.success("Clear")


# ============================================================
                          # RIGHT
# ============================================================

with right_col:

    st.subheader("➡️ RIGHT")

    if event["position"] == "right":

        icon = hazard_icons.get(
            event["type"],
            "⚠️"
        )

        name = hazard_names.get(
            event["type"],
            event["type"]
        )

        st.warning(
            f"{icon} {name}"
        )

        if event["distance"] is not None:

            st.write(
                f"Distance: **{event['distance']} m**"
            )

    else:

        st.success("Clear")


# ============================================================
              # CURRENT HAZARD + DECISION
# ============================================================

st.divider()


hazard_col, decision_col = st.columns(2)


# ============================================================
                      # CURRENT HAZARD
# ============================================================

with hazard_col:

    st.markdown(
        '<div class="section-title">🚨 Current Hazard</div>',
        unsafe_allow_html=True
    )


    icon = hazard_icons.get(
        event["type"],
        "⚠️"
    )


    name = hazard_names.get(
        event["type"],
        event["type"]
    )


    st.markdown(
        '<div class="hazard-box">',
        unsafe_allow_html=True
    )


    st.subheader(
        f"{icon} {name}"
    )


    if event["subtype"] is not None:

        st.write(
            f"Subtype: **{event['subtype'].title()}**"
        )


    st.write(
        f"📍 Position: **{event['position'].title()}**"
    )


    if event["distance"] is not None:

        st.write(
            f"📏 Distance: **{event['distance']} m**"
        )

    else:

        st.write(
            "📏 Distance: **N/A**"
        )


    st.write(
        f"🎯 Confidence: **{event['confidence'] * 100:.0f}%**"
    )


    if event["sensor_status"] == "active":

        st.success("📡 Sensor Active")

    else:

        st.error("📡 Sensor Failed")


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
                         # DECISION
# ============================================================

with decision_col:

    st.markdown(
        '<div class="section-title">🧠 Decision Engine</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="decision-box">',
        unsafe_allow_html=True
    )


    risk = decision["risk"]


    if risk == "LOW":

        st.success(
            f"RISK: {risk}"
        )

    elif risk == "MEDIUM":

        st.warning(
            f"RISK: {risk}"
        )

    elif risk == "HIGH":

        st.error(
            f"RISK: {risk}"
        )

    else:

        st.info(
            f"RISK: {risk}"
        )


    st.markdown(
        f'<div class="big-action">'
        f'🚦 {decision["action"].replace("_", " ")}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.write("### Why?")


    st.write(
        decision["reason"]
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
                       # SENSOR STATUS
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">📡 Sensor Status</div>',
    unsafe_allow_html=True
)


sensor_col1, sensor_col2, sensor_col3 = st.columns(3)


with sensor_col1:

    if event["sensor_status"] == "active":

        st.success(
            "🟢 Environmental Sensor — ACTIVE"
        )

    else:

        st.error(
            "🔴 Environmental Sensor — FAILED"
        )


with sensor_col2:

    st.metric(
        "Detection Confidence",
        f"{event['confidence'] * 100:.1f}%"
    )


with sensor_col3:

    st.metric(
        "Event ID",
        event["id"]
    )
# ============================================================
# LIVE ANALYTICS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Live Trip Analytics</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREPARE GRAPH DATA
# ============================================================

history = st.session_state.event_history


if len(history) > 0:

    graph_df = pd.DataFrame(history)


    # --------------------------------------------------------
    # GRAPH 1 — RISK OVER TIME
    # --------------------------------------------------------

    risk_mapping = {

        "LOW": 1,

        "MEDIUM": 2,

        "HIGH": 3,

        "UNCERTAIN": 0
    }


    graph_df["Risk Level"] = graph_df[
        "Risk"
    ].map(risk_mapping)


    # --------------------------------------------------------
    # GRAPH 2 — DISTANCE
    # --------------------------------------------------------

    graph_df["Distance Numeric"] = pd.to_numeric(
        graph_df["Distance"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # GRAPH 3 — CONFIDENCE
    # --------------------------------------------------------

    graph_df["Confidence Numeric"] = (
        graph_df["Confidence"]
        .str.replace("%", "")
        .astype(float)
    )


    # --------------------------------------------------------
    # GRAPH LAYOUT
    # --------------------------------------------------------

    graph1, graph2 = st.columns(2)


    # ========================================================
    # RISK GRAPH
    # ========================================================

    with graph1:

        st.subheader(
            "⚠️ Risk Level Over Time"
        )

        risk_chart = graph_df[
            ["Event", "Risk Level"]
        ].set_index("Event")


        st.line_chart(
            risk_chart,
            y="Risk Level"
        )


        st.caption(
            "0 = Uncertain | 1 = Low | "
            "2 = Medium | 3 = High"
        )


    # ========================================================
    # DISTANCE GRAPH
    # ========================================================

    with graph2:

        st.subheader(
            "📏 Hazard Distance"
        )

        distance_chart = graph_df[
            ["Event", "Distance Numeric"]
        ].dropna().set_index("Event")


        if len(distance_chart) > 0:

            st.line_chart(
                distance_chart,
                y="Distance Numeric"
            )

        else:

            st.info(
                "No measurable hazard distance yet."
            )


    # ========================================================
    # CONFIDENCE GRAPH
    # ========================================================

    st.subheader(
        "🎯 Sensor Confidence"
    )


    confidence_chart = graph_df[
        ["Event", "Confidence Numeric"]
    ].set_index("Event")


    st.line_chart(
        confidence_chart,
        y="Confidence Numeric"
    )


else:

    st.info(
        "Start the trip simulation to generate live analytics."
    )

# ============================================================
                      # TRIP PERFORMANCE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">📊 Trip Performance</div>',
    unsafe_allow_html=True
)


metric1, metric2, metric3, metric4, metric5 = st.columns(5)


with metric1:

    st.metric(
        "Distance",
        f"{trip_distance:.1f} km"
    )


with metric2:

    st.metric(
        "Hazards Detected",
        hazard_count
    )


with metric3:

    st.metric(
        "Warnings",
        warning_count
    )


with metric4:

    st.metric(
        "Brake Events",
        brake_count
    )


with metric5:

    st.metric(
        "Avg Confidence",
        f"{average_confidence * 100:.0f}%"
    )


# ============================================================
# EVENT HISTORY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📋 Event History</div>',
    unsafe_allow_html=True
)


if len(st.session_state.event_history) > 0:

    history_display = pd.DataFrame(
        st.session_state.event_history
    )


    # Remove internal ID column

    history_display = history_display.drop(
        columns=["ID"],
        errors="ignore"
    )


    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No events recorded yet."
    )

# ============================================================
                      # SYSTEM PIPELINE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">🔄 Decision Pipeline</div>',
    unsafe_allow_html=True
)


pipe1, pipe2, pipe3, pipe4, pipe5 = st.columns(5)


with pipe1:

    st.info(
        "🌍\n\nEnvironmental\nAwareness"
    )


with pipe2:

    st.info(
        "⚠️\n\nHazard\nEvent"
    )


with pipe3:

    st.info(
        "🧠\n\nRisk\nAssessment"
    )


with pipe4:

    st.info(
        "🚦\n\nDecision\nEngine"
    )


with pipe5:

    st.success(
        "🖥️\n\nDashboard\nOutput"
    )


# ============================================================
                        # FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Navigation & Decision-Support System | "
    "Detect → Understand → Assess → Decide → Explain"
)
# ============================================================
# AUTOMATIC SIMULATION LOOP
# ============================================================

if st.session_state.simulation_running:

    # Wait between events

    time.sleep(3)


    # Move to next event

    if (
        st.session_state.simulation_index
        < len(SIMULATION_EVENTS) - 1
    ):

        st.session_state.simulation_index += 1

    else:

        # Trip finished

        st.session_state.simulation_running = False


    st.rerun()