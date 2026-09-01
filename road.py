# roaddd.py

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="AI Road Safety Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

html_path = Path(__file__).parent / "simulation.html"

if not html_path.exists():
    st.error("simulation.html must be in the same folder as roaddd.py")
else:
    html = html_path.read_text(encoding="utf-8")

    components.html(
        html,
        height=900,
        scrolling=False,
    )