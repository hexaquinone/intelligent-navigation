# ============================================================
# INTELLIGENT NAVIGATION SYSTEM • AUTONOMOUS CLI ENGINE
# Multi-Modal Perception, Decision Support & Safety Analytics
# ============================================================

import sys
import os
import time
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List

# Ensure UTF-8 output on Windows terminals
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from brain import (
    evaluate_scene,
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
from simulation import SimulationEngine, SCENARIOS, TRIP_TIMELINE
from metrics import record_event, get_metrics, reset_metrics, get_event_history


# ============================================================
# 1. ANSI COLOR & FORMATTING UTILITIES
# ============================================================

class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background
    BG_RED = "\033[41m\033[97m"
    BG_GREEN = "\033[42m\033[97m"
    BG_YELLOW = "\033[43m\033[30m"
    BG_BLUE = "\033[44m\033[97m"
    BG_MAGENTA = "\033[45m\033[97m"


def print_banner():
    """Prints the cybernetic ASCII header banner."""
    print(f"""
{Color.CYAN}{Color.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗
║         🚗 AI INTELLIGENT NAVIGATION & ROAD SAFETY SYSTEM                    ║
║         Autonomous Multi-Modal Decision, Kinematics & XAI Engine             ║
╚══════════════════════════════════════════════════════════════════════════════╝{Color.RESET}
""")


def get_risk_badge(risk_str: str) -> str:
    """Formats risk level with colored terminal badges."""
    r = str(risk_str).upper()
    if r in ["CRITICAL", "HIGH"]:
        return f"{Color.BG_RED} 🔴 {r} {Color.RESET}"
    elif r == "MEDIUM":
        return f"{Color.BG_YELLOW} 🟡 {r} {Color.RESET}"
    elif r in ["LOW", "FINE"]:
        return f"{Color.BG_GREEN} 🟢 {r} {Color.RESET}"
    else:
        return f"{Color.BG_MAGENTA} 🟣 {r} {Color.RESET}"


def get_action_badge(action_str: str) -> str:
    """Formats decision actions with high-contrast colored badges."""
    act = str(action_str).upper()
    if "STOP" in act:
        return f"{Color.RED}{Color.BOLD}🛑 {act}{Color.RESET}"
    elif "BRAKE" in act:
        return f"{Color.RED}{Color.BOLD}⚡ {act}{Color.RESET}"
    elif "SLOW" in act:
        return f"{Color.YELLOW}{Color.BOLD}⚠️  {act}{Color.RESET}"
    elif "MOVE" in act:
        return f"{Color.BLUE}{Color.BOLD}↪  {act}{Color.RESET}"
    else:
        return f"{Color.GREEN}{Color.BOLD}✓  {act}{Color.RESET}"


def get_priority_label(risk: str, action: str) -> str:
    act = str(action).upper()
    r = str(risk).upper()
    if r in ["CRITICAL", "HIGH"] and ("STOP" in act or "BRAKE" in act):
        return "P5 (Emergency)"
    elif r in ["CRITICAL", "HIGH"]:
        return "P4 (High Risk)"
    elif r == "MEDIUM" or "SLOW" in act:
        return "P3 (Caution)"
    elif "MOVE" in act:
        return "P2 (Maneuver)"
    else:
        return "P1 (Nominal)"


# ============================================================
# 2. SEQUENTIAL TRIP SIMULATION RUNNER
# ============================================================

def run_trip(delay_sec: float = 0.6, export_csv: Optional[str] = None):
    """Executes a chronological multi-step drive simulation with live telemetry."""
    print_banner()
    print(f"{Color.BOLD}▶ LAUNCHING FULL MULTI-MODAL TRIP SIMULATION...{Color.RESET}\n")

    reset_metrics()
    timeline = TRIP_TIMELINE

    for idx, step in enumerate(timeline, start=1):
        ego_speed = float(step.get("ego_speed", 40.0))
        ego_state = EgoState(speed_kmh=ego_speed, lane="center")
        raw_events = step.get("events", [])
        hazards = [HazardEvent.from_dict(e) for e in raw_events]

        # Run unified brain evaluation
        decision, kinematics, sectors = evaluate_scene(hazards, ego_state)

        # Ingest into metrics
        primary = hazards[0] if hazards else HazardEvent(type=HazardType.CLEAR)
        pos_str = getattr(primary.position, "value", str(primary.position))
        record_event(
            event=primary,
            risk=decision.risk,
            action=decision.action,
            speed_kmh=ego_speed,
            dt_seconds=4.0,
            reason=decision.reason,
            distance_m=primary.distance,
            position=pos_str
        )

        # Render Step Telemetry Card
        print(f"{Color.CYAN}{'═' * 78}{Color.RESET}")
        print(f"{Color.BOLD}⏱️  STEP {idx}/{len(timeline)}: {step.get('description')}{Color.RESET}")
        print(f"{Color.CYAN}{'─' * 78}{Color.RESET}")

        print(f"🚗 {Color.BOLD}Vehicle State:{Color.RESET} Speed = {Color.YELLOW}{ego_speed:.0f} km/h{Color.RESET} | Lane = {Color.WHITE}{ego_state.lane.upper()}{Color.RESET}")

        if hazards and hazards[0].type != HazardType.CLEAR:
            print(f"\n📡 {Color.BOLD}Perception Hazards Detected ({len(hazards)}):{Color.RESET}")
            for h in hazards:
                dist_str = f"{h.distance:.1f}m" if h.distance is not None else "--"
                conf_str = f"{h.confidence * 100:.0f}%"
                sensor_str = getattr(h.sensor_status, "value", str(h.sensor_status)).upper()
                pos_disp = getattr(h.position, "value", str(h.position)).upper()
                print(f"   • {Color.WHITE}{h.type.value.upper()}{Color.RESET} ({h.subtype or 'generic'}) | Pos: {Color.CYAN}{pos_disp}{Color.RESET} | Dist: {Color.YELLOW}{dist_str}{Color.RESET} | Sensor: {sensor_str} (Conf: {conf_str})")
        else:
            print(f"\n📡 {Color.BOLD}Perception:{Color.RESET} {Color.GREEN}No roadway hazards detected (Roadway Clear){Color.RESET}")

        # Kinematics Envelope
        ttc_disp = f"{decision.ttc_seconds}s" if decision.ttc_seconds else "Safe / No Collision Trajectory"
        margin_disp = f"{kinematics.safety_margin_m:+.1f} m" if kinematics.safety_margin_m is not None else "Safe"
        print(f"\n📐 {Color.BOLD}Kinematics:{Color.RESET} TTC: {Color.CYAN}{ttc_disp}{Color.RESET} | Stopping Dist: {Color.YELLOW}{kinematics.total_stopping_dist_m}m{Color.RESET} | Safety Margin: {Color.GREEN if (kinematics.safety_margin_m or 0) >= 0 else Color.RED}{margin_disp}{Color.RESET}")

        # AI Decision & Explainability
        p_label = get_priority_label(decision.risk, decision.action)
        print(f"\n🧠 {Color.BOLD}Autonomous AI Decision:{Color.RESET}")
        print(f"   Action:    {get_action_badge(decision.action)}")
        print(f"   Risk:      {get_risk_badge(decision.risk)} (Priority: {Color.BOLD}{p_label}{Color.RESET})")
        print(f"   Rationale: {Color.WHITE}{decision.reason}{Color.RESET}")

        if delay_sec > 0:
            time.sleep(delay_sec)

    # Render Trip Performance & Safety KPI Summary
    render_trip_summary(export_csv)


def render_trip_summary(export_csv: Optional[str] = None):
    """Renders formatted trip KPI table, safety score gauge, and export summary."""
    metrics = get_metrics()
    score = metrics.get("safety_score", 100)

    print(f"\n{Color.CYAN}{'═' * 78}{Color.RESET}")
    print(f"{Color.BOLD}📊 TRIP PERFORMANCE & AUTONOMOUS SAFETY AUDIT{Color.RESET}")
    print(f"{Color.CYAN}{'═' * 78}{Color.RESET}")

    # Visual Safety Score Gauge
    score_bars = int(score / 5)
    gauge = f"{'█' * score_bars}{'░' * (20 - score_bars)}"
    gauge_color = Color.GREEN if score >= 85 else (Color.YELLOW if score >= 65 else Color.RED)
    print(f"\n🛡️  {Color.BOLD}Safety Score:{Color.RESET} {gauge_color}[{gauge}] {score}%{Color.RESET}")

    # Metrics Table
    print(f"\n┌{'─' * 36}┬{'─' * 39}┐")
    print(f"│ {Color.BOLD}Performance Metric{Color.RESET}{' ' * 18} │ {Color.BOLD}Recorded Value{Color.RESET}{' ' * 24} │")
    print(f"├{'─' * 36}┼{'─' * 39}┤")
    print(f"│ Total Accumulated Distance         │ {metrics['trip_distance_km']:.2f} km{' ' * max(0, 32 - len(f'{metrics["trip_distance_km"]:.2f} km'))}│")
    print(f"│ Total Roadway Events Evaluated     │ {metrics['total_events']}{' ' * max(0, 35 - len(str(metrics['total_events'])))}│")
    print(f"│ Active Hazards Detected            │ {metrics['hazards_detected']}{' ' * max(0, 35 - len(str(metrics['hazards_detected'])))}│")
    print(f"│ High / Critical Risk Events        │ {metrics['high_risk_events']}{' ' * max(0, 35 - len(str(metrics['high_risk_events'])))}│")
    print(f"│ Autonomous Warning Interventions   │ {metrics['warnings_count']}{' ' * max(0, 35 - len(str(metrics['warnings_count'])))}│")
    print(f"│ Emergency Braking Events           │ {metrics['brake_events']}{' ' * max(0, 35 - len(str(metrics['brake_events'])))}│")
    print(f"│ Average Sensor Confidence          │ {int(metrics['average_confidence'] * 100)}%{' ' * 32}│")
    print(f"└{'─' * 36}┴{'─' * 39}┘")

    # Optional CSV Export
    if export_csv:
        try:
            import pandas as pd
            history = get_event_history()
            df = pd.DataFrame(history)
            df.to_csv(export_csv, index=False)
            print(f"\n{Color.GREEN}✓ Blackbox audit telemetry successfully exported to: {export_csv}{Color.RESET}")
        except Exception as e:
            print(f"\n{Color.RED}⚠️ Failed to export CSV: {e}{Color.RESET}")


# ============================================================
# 3. INDIVIDUAL SCENARIO EVALUATOR
# ============================================================

def run_single_scenario(scenario_key: str):
    """Executes and inspects a single isolated scenario catalog preset."""
    if scenario_key not in SCENARIOS:
        print(f"{Color.RED}Error: Scenario '{scenario_key}' not found.{Color.RESET}")
        print(f"Available scenarios: {list(SCENARIOS.keys())}")
        return

    sc = SCENARIOS[scenario_key]
    print_banner()
    print(f"{Color.BOLD}🛡️  SCENARIO INSPECTION: {sc['title']}{Color.RESET}")
    print(f"Description: {sc['description']}")
    print(f"{Color.CYAN}{'─' * 78}{Color.RESET}")

    ego = sc["ego_state"]
    events = sc["events"]
    decision, kinematics, sectors = evaluate_scene(events, ego)

    print(f"\n🚗 {Color.BOLD}Vehicle State:{Color.RESET} Speed: {ego.speed_kmh} km/h | Lane: {ego.lane}")
    print(f"📡 {Color.BOLD}Hazards Detected ({len(events)}):{Color.RESET}")
    for h in events:
        dist_str = f"{h.distance:.1f}m" if h.distance is not None else "--"
        print(f"   • {h.type.value.upper()} ({h.subtype or 'generic'}) | Position: {h.position.value.upper()} | Distance: {dist_str}")

    p_label = get_priority_label(decision.risk, decision.action)
    print(f"\n🧠 {Color.BOLD}AI Decision:{Color.RESET}")
    print(f"   Action:    {get_action_badge(decision.action)}")
    print(f"   Risk:      {get_risk_badge(decision.risk)} (Priority: {p_label})")
    print(f"   TTC:       {decision.ttc_seconds}s" if decision.ttc_seconds else "   TTC:       Safe")
    print(f"   Margin:    {kinematics.safety_margin_m:+.1f} m" if kinematics.safety_margin_m is not None else "   Margin:    Safe")
    print(f"   Rationale: {Color.WHITE}{decision.reason}{Color.RESET}")
    print(f"{Color.CYAN}{'═' * 78}{Color.RESET}\n")


# ============================================================
# 4. COMPUTER VISION & PERCEPTION LAUNCHER
# ============================================================

def run_vision_perception(source: str = "demo"):
    """Launches the real-time OpenCV + YOLOv8 vision perception pipeline."""
    from vision import run_vision_cli
    sys.argv = ["vision.py", "--source", source]
    run_vision_cli()


# ============================================================
# 5. GUI & WEB APP LAUNCHER
# ============================================================

def launch_web_gui(target: str = "road"):
    """Launches the Streamlit interactive 3D Simulator or Cockpit Suite."""
    script_name = "road.py" if target == "road" else "app.py"
    target_path = os.path.join(os.path.dirname(__file__), script_name)
    print_banner()
    print(f"{Color.GREEN}{Color.BOLD}🚀 Launching Streamlit Web Cockpit ({script_name})...{Color.RESET}")
    print(f"Running command: streamlit run \"{target_path}\"\n")
    subprocess.run(["streamlit", "run", target_path])


# ============================================================
# 6. SELF-DIAGNOSTIC TEST RUNNER
# ============================================================

def run_diagnostics():
    """Executes the complete 4-tier unit and integration test suite."""
    test_path = os.path.join(os.path.dirname(__file__), "test_brain.py")
    print_banner()
    print(f"{Color.CYAN}{Color.BOLD}🧪 RUNNING SELF-DIAGNOSTIC TEST SUITE (test_brain.py)...{Color.RESET}\n")
    subprocess.run([sys.executable, test_path])


# ============================================================
# 7. INTERACTIVE CLI MENU
# ============================================================

def interactive_menu():
    """Interactive command-line menu for easy navigation."""
    while True:
        print_banner()
        print(f"{Color.BOLD}MAIN NAVIGATION MENU:{Color.RESET}")
        print("  1. 🚗 Run Full Trip Simulation (Sequential Drive Timeline)")
        print("  2. 🛡️  Inspect Specific Scenario Preset")
        print("  3. 👁️  Launch Computer Vision Perception (Demo / Synthetic Stream)")
        print("  4. 📹 Launch Live Webcam Perception Pipeline")
        print("  5. 🌐 Launch 3D WebGL Road Simulator (road.py)")
        print("  6. 🎛️  Launch Autonomous Navigation Cockpit (app.py)")
        print("  7. 🧪 Run Self-Diagnostic & Test Suite (test_brain.py)")
        print("  8. ❌ Exit")
        print()

        choice = input(f"{Color.CYAN}Select an option [1-8]: {Color.RESET}").strip()

        if choice == "1":
            run_trip(delay_sec=0.5)
            input(f"\n{Color.DIM}Press Enter to return to menu...{Color.RESET}")
        elif choice == "2":
            sc_keys = list(SCENARIOS.keys())
            print(f"\n{Color.BOLD}Available Scenarios:{Color.RESET}")
            for idx, k in enumerate(sc_keys, start=1):
                print(f"  {idx}. {SCENARIOS[k]['title']} ({k})")
            sc_idx = input(f"\nSelect scenario [1-{len(sc_keys)}]: ").strip()
            try:
                sel_key = sc_keys[int(sc_idx) - 1]
                run_single_scenario(sel_key)
            except (ValueError, IndexError):
                print(f"{Color.RED}Invalid scenario selection.{Color.RESET}")
            input(f"\n{Color.DIM}Press Enter to return to menu...{Color.RESET}")
        elif choice == "3":
            run_vision_perception("demo")
            input(f"\n{Color.DIM}Press Enter to return to menu...{Color.RESET}")
        elif choice == "4":
            run_vision_perception("webcam")
            input(f"\n{Color.DIM}Press Enter to return to menu...{Color.RESET}")
        elif choice == "5":
            launch_web_gui("road")
            break
        elif choice == "6":
            launch_web_gui("app")
            break
        elif choice == "7":
            run_diagnostics()
            input(f"\n{Color.DIM}Press Enter to return to menu...{Color.RESET}")
        elif choice == "8" or choice.lower() in ["exit", "q", "quit"]:
            print(f"\n{Color.GREEN}Safe travels! Autonomous system exiting.{Color.RESET}\n")
            sys.exit(0)
        else:
            print(f"{Color.RED}Invalid choice. Please choose 1-8.{Color.RESET}")
            time.sleep(1)


# ============================================================
# 8. COMMAND LINE DISPATCHER
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚗 AI Intelligent Navigation & Road Safety System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Launch interactive CLI menu
  python main.py --mode trip               # Run multi-step trip simulation
  python main.py --mode trip --delay 0.8   # Run trip with 0.8s pacing between steps
  python main.py --mode trip --export trip_log.csv # Export trip metrics to CSV
  python main.py --scenario urban_pedestrian # Run isolated scenario
  python main.py --mode vision             # Run YOLO + OpenCV perception demo
  python main.py --mode webcam             # Run live webcam perception
  python main.py --launch road             # Launch 3D simulator web app
  python main.py --launch app              # Launch full cockpit suite web app
  python main.py --test                    # Run complete test suite
        """
    )

    parser.add_argument(
        "--mode",
        choices=["trip", "vision", "webcam", "menu"],
        default="menu" if len(sys.argv) == 1 else "trip",
        help="Execution mode (default: 'menu' if no args, 'trip' otherwise)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a specific scenario by key (e.g., 'urban_pedestrian', 'highway_lead_vehicle')"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Step delay in seconds for trip simulation (default: 0.5)"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export trip audit log to CSV filepath (e.g. --export drive_log.csv)"
    )
    parser.add_argument(
        "--launch",
        choices=["road", "app"],
        default=None,
        help="Launch Streamlit Web App directly ('road' or 'app')"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run self-diagnostic unit and integration test suite"
    )

    args = parser.parse_args()

    if args.test:
        run_diagnostics()
    elif args.launch:
        launch_web_gui(args.launch)
    elif args.scenario:
        run_single_scenario(args.scenario)
    elif args.mode == "menu":
        interactive_menu()
    elif args.mode == "trip":
        run_trip(delay_sec=args.delay, export_csv=args.export)
    elif args.mode == "webcam":
        run_vision_perception("webcam")
    elif args.mode == "vision":
        run_vision_perception("demo")
    else:
        interactive_menu()
