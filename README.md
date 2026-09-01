# 🚗 Intelligent Navigation & Decision-Support System

An end-to-end autonomous driving perception and decision-support platform that transforms **Computer Vision (YOLOv8/11 + OpenCV)**, simulated multi-modal sensors (Radar/LiDAR), and real-time telemetry into **risk-aware, physics-grounded, explainable driving decisions**.

---

## 🎯 System Architecture

```text
📸 Input (Camera / Video / Sim) 
  │
  ▼
👁️ Vision Perception Layer (vision.py)
  ├── Real-time YOLOv8 Object Detection (COCO -> ADAS Hazard Classes)
  ├── Monocular Pinhole Distance Estimation & Closing Velocity (TTC)
  └── OpenCV Canny / Hough Lane Tracking & Departure Detection
  │
  ▼
🧠 Core Decision Engine (brain.py)
  ├── Multi-Hazard Spatial Arbitration (Front / Left / Right Sectors)
  ├── Swerve Conflict Resolution Matrix (Evasive Lane Safety)
  ├── Multi-Modal Sensor Fusion (Camera + Radar + LiDAR)
  ├── Lane Keeping Safety (evaluate_lane_departure_safety)
  └── Explainable AI Rationale & Safety Arbitration Priority (1-5)
  │
  ▼
🚗 Visualization & Cockpit HUD (app.py, road.py, simulation.html)
  ├── Augmented Reality (AR) Cockpit HUD Video Player
  ├── Synchronized 2D Bird's-Eye View (BEV) Perception Canvas
  ├── 3D Physics Road Simulator Canvas
  └── Blackbox Decision Audit History & Telemetry Analytics (metrics.py)
```

---

## 📁 Repository Structure

```text
app.py          # Unified Cockpit Dashboard (AR HUD, BEV Map, What-If Sandbox, Vision Suite)
vision.py       # Computer Vision engine (YOLOv8/11, Monocular Distance, OpenCV Lane Tracking)
brain.py        # Autonomous decision engine, kinematics, sensor fusion, & arbitration
main.py         # Unified CLI launcher for sensor trip simulation or live vision perception
road.py         # Bridge component & standalone runner for the 3D Canvas driving simulator
simulation.html # Interactive canvas-based driving physics simulation
simulation.py   # Multi-sensor simulation engine & scenario catalog
metrics.py      # Blackbox audit log, trip performance tracking, & KPI analytics
test_brain.py   # Comprehensive verification test suite (100% automated test coverage)
```

---

## 🚀 Quickstart & Execution Modes

### 1. Launch the Full Cockpit HUD Web Dashboard
```bash
streamlit run app.py
```
* **Operational Modes:**
  * 🚗 **Live Trip Timeline:** Automated multi-step driving trip with real-time risk graphs.
  * 👁️ **Live Vision & YOLO Perception:** Video/Image dashcam ingestion, animated scenarios, and webcam perception.
  * 🔬 **Preset Scenario Explorer:** Urban pedestrian crossings, highway braking, tight pinch hazards, sensor dropouts.
  * 🛠️ **Interactive What-If Sandbox:** Custom multi-hazard builder and sensor fault injection.
  * 🎮 **3D Road Simulator:** Interactive canvas-based driving physics with keyboard controls.

### 2. Run the Interactive 30 FPS Animated Vision CLI
```bash
# Animated driving simulator with live hotkeys (1, 2, 3, W, Space, Q)
python vision.py

# Or run directly with your webcam:
python vision.py --source webcam
```

### 3. Run the Unified Command Line Entry Point
```bash
# Sensor simulation trip:
python main.py --mode trip

# Computer vision perception pipeline:
python main.py --mode vision
```

### 4. Run the Automated Test Suite
```bash
python test_brain.py
```

---

> **Our Goal:** Don't just detect hazards—understand their spatial kinematics, arbitrate safety conflicts, decide optimal actions, and explain why.


