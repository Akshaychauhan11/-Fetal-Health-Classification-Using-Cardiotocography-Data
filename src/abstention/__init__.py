"""
Selective Classification and Abstention Module.

Provides:
- decide_prediction: Accept/abstain decision logic
- evaluate_selective_abstention: Coverage and selective risk computation
- compute_ae_method_comparison: Method A-E benchmark comparison
- compute_risk_coverage_curves: Risk-coverage curve coordinates
- plot_risk_coverage_curves: Visualization
"""

from src.abstention.selective import (
    decide_prediction,
    evaluate_selective_abstention,
    compute_ae_method_comparison,
    compute_risk_coverage_curves,
    plot_risk_coverage_curves,
    CLASS_NAMES_DEFAULT
)

__all__ = [
    "decide_prediction",
    "evaluate_selective_abstention",
    "compute_ae_method_comparison",
    "compute_risk_coverage_curves",
    "plot_risk_coverage_curves",
    "CLASS_NAMES_DEFAULT"
]
