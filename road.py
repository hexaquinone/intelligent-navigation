import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="AI Road Safety Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit's default UI so the simulation gets the whole screen.
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            margin: 0;
            padding: 0;
        }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }

        iframe {
            width: 100% !important;
            height: 100vh !important;
            border: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

html_file = Path(__file__).parent / "simulation.html"

if not html_file.exists():
    st.error(
        "simulation.html was not found. "
        "Make sure simulation.html is in the same folder as road.py."
    )
    st.stop()

html_code = html_file.read_text(encoding="utf-8")

components.html(
    html_code,
    height=1000,
    scrolling=False
)