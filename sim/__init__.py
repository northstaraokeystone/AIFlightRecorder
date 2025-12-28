"""Monte Carlo Simulation Harness for AI Flight Recorder

Provides validation framework with 7 mandatory scenarios:
1. BASELINE - Standard operation
2. STRESS - High decision rate, resource constraints
3. TOPOLOGY - Pattern classification accuracy
4. CASCADE - High-volume decision logging
5. COMPRESSION - Anomaly detection validation
6. SINGULARITY - Long-run stability
7. THERMODYNAMIC - Hash integrity conservation
"""

from .sim import SimConfig, SimState, run_simulation, execute_cycle
from .scenarios import BASELINE, STRESS, TOPOLOGY, CASCADE, COMPRESSION, SINGULARITY, THERMODYNAMIC
