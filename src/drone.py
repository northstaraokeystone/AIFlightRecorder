"""Simulated Autonomous AI Decision Generator

The "flight brain" - generates AI decisions at 10Hz.
Each decision captures complete context: telemetry, perception, reasoning, action.
"""

import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .core import emit_receipt, dual_hash

# Action types for the drone
ACTION_TYPES = ["CONTINUE", "AVOID", "ENGAGE", "ABORT", "HOVER", "RTB"]

# Model version for tracking
MODEL_VERSION = "drone-brain-v1.0.0"


@dataclass
class DroneState:
    """Complete drone state at a given moment."""
    # Position
    lat: float = 37.7749  # San Francisco default
    lon: float = -122.4194
    alt: float = 100.0  # meters

    # Velocity (m/s)
    vx: float = 5.0
    vy: float = 0.0
    vz: float = 0.0

    # Status
    battery_pct: float = 100.0
    heading_deg: float = 0.0
    speed_mps: float = 5.0

    # Mission
    mission_id: str = "mission-001"
    cycle_number: int = 0
    waypoint_index: int = 0

    # Chain tracking
    prev_decision_hash: str = ""


@dataclass
class Obstacle:
    """Detected obstacle in environment."""
    type: str
    distance_m: float
    bearing_deg: float
    confidence: float


@dataclass
class Target:
    """Mission target to engage/observe."""
    id: str
    lat: float
    lon: float
    alt: float
    priority: int


@dataclass
class Threat:
    """Detected threat."""
    type: str
    severity: float  # 0-1


def generate_telemetry(state: DroneState, seed: Optional[int] = None) -> dict:
    """Generate current telemetry snapshot.

    Args:
        state: Current drone state
        seed: Optional random seed for reproducibility

    Returns:
        Telemetry dict with GPS, battery, velocity
    """
    if seed is not None:
        random.seed(seed + state.cycle_number)

    # Add slight noise to simulate sensor readings
    noise_lat = random.gauss(0, 0.00001)
    noise_lon = random.gauss(0, 0.00001)
    noise_alt = random.gauss(0, 0.5)

    telemetry = {
        "gps": {
            "lat": state.lat + noise_lat,
            "lon": state.lon + noise_lon,
            "alt": state.alt + noise_alt
        },
        "battery_pct": max(0, state.battery_pct - random.uniform(0, 0.1)),
        "velocity": {
            "vx": state.vx + random.gauss(0, 0.1),
            "vy": state.vy + random.gauss(0, 0.1),
            "vz": state.vz + random.gauss(0, 0.05)
        },
        "heading_deg": state.heading_deg,
        "speed_mps": state.speed_mps,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return telemetry


def perceive_environment(telemetry: dict, seed: Optional[int] = None) -> dict:
    """Simulate sensor fusion to detect obstacles, targets, threats.

    Args:
        telemetry: Current telemetry data
        seed: Optional random seed for reproducibility

    Returns:
        Perception dict with obstacles, targets, threats
    """
    if seed is not None:
        random.seed(seed)

    obstacles = []
    targets = []
    threats = []

    # 20% chance of detecting an obstacle
    if random.random() < 0.2:
        obstacle_types = ["building", "tree", "drone", "bird", "power_line"]
        obstacles.append({
            "type": random.choice(obstacle_types),
            "distance_m": random.uniform(10, 200),
            "bearing_deg": random.uniform(0, 360),
            "confidence": random.uniform(0.7, 0.99)
        })

    # 10% chance of detecting a target
    if random.random() < 0.1:
        lat = telemetry["gps"]["lat"] + random.uniform(-0.001, 0.001)
        lon = telemetry["gps"]["lon"] + random.uniform(-0.001, 0.001)
        targets.append({
            "id": f"target-{uuid.uuid4().hex[:8]}",
            "position": {"lat": lat, "lon": lon, "alt": telemetry["gps"]["alt"]},
            "priority": random.randint(1, 5)
        })

    # 5% chance of detecting a threat
    if random.random() < 0.05:
        threat_types = ["jamming", "radar_lock", "restricted_airspace", "weather"]
        threats.append({
            "type": random.choice(threat_types),
            "severity": random.uniform(0.3, 0.9)
        })

    return {
        "obstacles": obstacles,
        "targets": targets,
        "threats": threats,
        "visibility": random.uniform(0.8, 1.0),
        "sensor_health": random.uniform(0.95, 1.0)
    }


def make_decision(perception: dict, state: DroneState,
                  mission: Optional[dict] = None,
                  seed: Optional[int] = None) -> dict:
    """AI decision engine: given perception and mission, decide action.

    Args:
        perception: Current environmental perception
        state: Current drone state
        mission: Optional mission parameters
        seed: Optional random seed for reproducibility

    Returns:
        Decision dict with action, confidence, reasoning, alternatives
    """
    if seed is not None:
        random.seed(seed)

    decision_id = str(uuid.uuid4())
    action_type = "CONTINUE"
    confidence = 0.95
    reasoning = "No obstacles detected, continuing on mission path"
    alternatives = []
    parameters = {
        "heading_deg": state.heading_deg,
        "altitude_delta_m": 0.0,
        "speed_mps": state.speed_mps
    }

    # Decision logic based on perception
    obstacles = perception.get("obstacles", [])
    threats = perception.get("threats", [])
    targets = perception.get("targets", [])

    # Priority 1: Threats
    if threats:
        highest_threat = max(threats, key=lambda t: t["severity"])
        if highest_threat["severity"] > 0.7:
            action_type = "ABORT"
            confidence = 0.99
            reasoning = f"High severity threat detected: {highest_threat['type']} (severity: {highest_threat['severity']:.2f})"
            parameters["speed_mps"] = 0
            alternatives = [
                {"action": "RTB", "reason": "Return to base as alternative"},
                {"action": "HOVER", "reason": "Hold position for assessment"}
            ]
        else:
            action_type = "AVOID"
            confidence = 0.85
            reasoning = f"Moderate threat detected: {highest_threat['type']}, adjusting course"
            parameters["heading_deg"] = (state.heading_deg + 90) % 360
            alternatives = [
                {"action": "CONTINUE", "reason": "Threat may be tolerable"},
                {"action": "RTB", "reason": "Abort mission entirely"}
            ]

    # Priority 2: Obstacles
    elif obstacles:
        closest = min(obstacles, key=lambda o: o["distance_m"])
        if closest["distance_m"] < 50:
            action_type = "AVOID"
            confidence = closest["confidence"]
            reasoning = f"Obstacle ({closest['type']}) at {closest['distance_m']:.1f}m, bearing {closest['bearing_deg']:.0f}°"

            # Calculate avoidance heading
            avoidance_heading = (closest["bearing_deg"] + 90) % 360
            parameters["heading_deg"] = avoidance_heading
            parameters["altitude_delta_m"] = 10.0 if closest["type"] in ["building", "tree"] else 0

            alternatives = [
                {"action": "HOVER", "reason": "Wait for obstacle to clear"},
                {"action": "CONTINUE", "reason": f"Obstacle confidence only {closest['confidence']:.2f}"}
            ]

    # Priority 3: Targets
    elif targets:
        highest_priority = min(targets, key=lambda t: t["priority"])
        action_type = "ENGAGE"
        confidence = 0.9
        reasoning = f"Target {highest_priority['id']} detected with priority {highest_priority['priority']}"

        # Calculate heading to target
        target_pos = highest_priority["position"]
        delta_lat = target_pos["lat"] - state.lat
        delta_lon = target_pos["lon"] - state.lon
        target_heading = math.degrees(math.atan2(delta_lon, delta_lat)) % 360
        parameters["heading_deg"] = target_heading

        alternatives = [
            {"action": "CONTINUE", "reason": "Target not mission-critical"},
            {"action": "HOVER", "reason": "Observe target first"}
        ]

    # Battery check
    if state.battery_pct < 20:
        action_type = "RTB"
        confidence = 0.99
        reasoning = f"Low battery ({state.battery_pct:.1f}%), returning to base"
        alternatives = [
            {"action": "HOVER", "reason": "Conserve power while awaiting pickup"}
        ]

    # Build the decision
    decision = {
        "decision_id": decision_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle_number": state.cycle_number,
        "telemetry_snapshot": generate_telemetry(state, seed),
        "perception": perception,
        "action": {
            "type": action_type,
            "parameters": parameters
        },
        "confidence": confidence,
        "reasoning": reasoning,
        "model_version": MODEL_VERSION,
        "alternative_actions_considered": alternatives,
        "mission_id": state.mission_id,
        "prev_decision_hash": state.prev_decision_hash
    }

    return decision


def apply_decision(state: DroneState, decision: dict) -> DroneState:
    """Apply decision to update drone state.

    Args:
        state: Current drone state
        decision: The decision to apply

    Returns:
        New drone state after applying decision
    """
    action = decision["action"]
    params = action["parameters"]

    # Update position based on velocity and heading
    dt = 0.1  # 10Hz = 0.1s per cycle

    heading_rad = math.radians(params.get("heading_deg", state.heading_deg))
    speed = params.get("speed_mps", state.speed_mps)

    # Calculate new velocity
    new_vx = speed * math.cos(heading_rad)
    new_vy = speed * math.sin(heading_rad)
    new_vz = params.get("altitude_delta_m", 0) / dt if action["type"] == "AVOID" else 0

    # Update position (simplified flat-earth approximation)
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * math.cos(math.radians(state.lat))

    new_lat = state.lat + (new_vx * dt) / meters_per_degree_lat
    new_lon = state.lon + (new_vy * dt) / meters_per_degree_lon
    new_alt = state.alt + new_vz * dt

    # Handle special actions
    if action["type"] in ["HOVER", "ABORT"]:
        new_vx, new_vy, new_vz = 0, 0, 0
        speed = 0
    elif action["type"] == "RTB":
        # Head back to origin
        speed = 10.0  # Faster return

    # Battery drain
    new_battery = state.battery_pct - 0.01 * (speed / 5.0)

    return DroneState(
        lat=new_lat,
        lon=new_lon,
        alt=max(0, min(500, new_alt)),  # Clamp altitude
        vx=new_vx,
        vy=new_vy,
        vz=new_vz,
        battery_pct=max(0, new_battery),
        heading_deg=params.get("heading_deg", state.heading_deg),
        speed_mps=speed,
        mission_id=state.mission_id,
        cycle_number=state.cycle_number + 1,
        waypoint_index=state.waypoint_index,
        prev_decision_hash=dual_hash(str(decision))
    )


def run_cycle(state: dict | DroneState, seed: Optional[int] = None) -> tuple[DroneState, dict]:
    """Execute one complete sense→decide cycle.

    Args:
        state: Current state (dict or DroneState)
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (new_state, decision_receipt)
    """
    # Convert dict to DroneState if needed
    if isinstance(state, dict):
        if not state:
            drone_state = DroneState()
        else:
            drone_state = DroneState(**{k: v for k, v in state.items()
                                        if k in DroneState.__dataclass_fields__})
    else:
        drone_state = state

    cycle_seed = seed + drone_state.cycle_number if seed is not None else None

    # Sense
    telemetry = generate_telemetry(drone_state, cycle_seed)
    perception = perceive_environment(telemetry, cycle_seed)

    # Decide
    decision = make_decision(perception, drone_state, seed=cycle_seed)

    # Apply
    new_state = apply_decision(drone_state, decision)

    # Emit decision receipt
    decision_receipt = emit_receipt("decision", {
        "decision_id": decision["decision_id"],
        "action_type": decision["action"]["type"],
        "confidence": decision["confidence"],
        "reasoning": decision["reasoning"],
        "model_version": decision["model_version"],
        "cycle_number": decision["cycle_number"],
        "mission_id": decision["mission_id"],
        "decision_hash": dual_hash(str(decision)),
        "full_decision": decision
    }, silent=True, to_file=True)

    return new_state, decision_receipt


def run_mission(n_cycles: int, seed: Optional[int] = None,
                initial_state: Optional[DroneState] = None) -> tuple[list[dict], DroneState]:
    """Run a complete mission with n decision cycles.

    Args:
        n_cycles: Number of decision cycles to run
        seed: Optional random seed for reproducibility
        initial_state: Optional starting state

    Returns:
        Tuple of (list of decision receipts, final state)
    """
    state = initial_state or DroneState()
    decisions = []

    for i in range(n_cycles):
        state, receipt = run_cycle(state, seed)
        decisions.append(receipt)

    return decisions, state
