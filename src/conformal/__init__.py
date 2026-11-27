"""
Conformal Prediction Module for Fetal Health Classification.

Provides:
- Global (marginal) split conformal prediction (LAC and APS)
- Class-conditional (Mondrian) conformal prediction
- Calibration-set size sensitivity experiment (RQ3)
- Evaluation metrics for coverage validity, set size efficiency, and composition
"""

from src.conformal.global_cp import (
    fit_global_conformal_classifier,
    predict_prediction_sets,
    evaluate_conformal_coverage,
    sweep_alpha_conformal,
    plot_global_conformal_comparison,
    CLASS_NAMES_DEFAULT
)

from src.conformal.mondrian_cp import (
    MondrianConformalClassifier,
    evaluate_mondrian_coverage,
    sweep_alpha_mondrian,
    run_calibration_sensitivity_experiment,
    plot_mondrian_comparison,
    plot_sensitivity_curves
)

__all__ = [
    "fit_global_conformal_classifier",
    "predict_prediction_sets",
    "evaluate_conformal_coverage",
    "sweep_alpha_conformal",
    "plot_global_conformal_comparison",
    "CLASS_NAMES_DEFAULT",
    "MondrianConformalClassifier",
    "evaluate_mondrian_coverage",
    "sweep_alpha_mondrian",
    "run_calibration_sensitivity_experiment",
    "plot_mondrian_comparison",
    "plot_sensitivity_curves"
]
