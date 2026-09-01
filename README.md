# intelligent-navigation

# 🚗 Intelligent Navigation & Decision-Support System

An intelligent driving assistance prototype that converts simulated sensor detections into **risk-aware, explainable driving recommendations**.

## 🎯 How It Works

**Detect → Assess Risk → Decide → Explain → Visualize → Log**

The system:

* Detects environmental hazards such as pedestrians, vehicles, and obstacles
* Assesses their position and risk level
* Recommends actions such as **CONTINUE, SLOW DOWN, BRAKE, or STOP**
* Explains the reasoning behind every recommendation
* Displays hazards and decisions through a live interface
* Tracks trip metrics such as trip time and hazard alerts
* Handles edge cases such as sensor gaps or conflicting detections


## 📁 Structure

```text
vision.py       # Computer Vision perception (YOLOv8/11 + OpenCV lane tracking & AR HUD)
simulation.py   # Sensor simulation & scenario catalog
brain.py        # Decision engine, kinematic TTC, & multi-hazard arbitration
app.py          # Live Cockpit HUD, 2D BEV Perception, & Telemetry Dashboard
road.py         # Canvas simulator runner & bridge component for app.py
simulation.html # Interactive canvas-based 2D/3D driving simulation
metrics.py      # Performance tracking & blackbox audit
test_brain.py   # Comprehensive verification test suite
```

## 🚀 Running the System

- **Full Cockpit HUD & Decision Dashboard** (Includes embedded 3D Road Simulator):
  ```bash
  streamlit run app.py
  ```

- **Standalone 3D Road Simulation**:
  ```bash
  streamlit run road.py
  ```

> **Our goal:** Don't just detect hazards—understand what they mean, decide what to do, and explain why.

