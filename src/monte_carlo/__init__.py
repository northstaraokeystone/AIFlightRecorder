"""Monte Carlo Variance Reduction Module - v2.0

Run N simulated decision variations to assess stability.
High variance = lower confidence, low variance = stable decision.
"""

from .simulate import simulate, SimulationResult
from .variance import calculate_variance, VarianceResult
from .threshold import check_stability, StabilityResult

__all__ = [
    "simulate",
    "SimulationResult",
    "calculate_variance",
    "VarianceResult",
    "check_stability",
    "StabilityResult"
]
