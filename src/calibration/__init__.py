"""
Calibration Module for Fetal Health Classification
Provides functions to assess probability calibration:
- Multi-class Brier Score (overall and per-class)
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Class-wise ECE
- Reliability diagram plotting
"""

from .calibrate import (
    compute_brier_scores,
    compute_ece,
    compute_classwise_ece,
    evaluate_calibration,
    plot_reliability_diagrams
)

__all__ = [
    "compute_brier_scores",
    "compute_ece",
    "compute_classwise_ece",
    "evaluate_calibration",
    "plot_reliability_diagrams"
]
