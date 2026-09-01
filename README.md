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
simulation.py   # Sensor simulation
brain.py        # Decision engine
app.py          # Live dashboard
metrics.py      # Performance tracking
```

> **Our goal:** Don't just detect hazards—understand what they mean, decide what to do, and explain why.
