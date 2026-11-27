"""
Explainability Module using SHAP for Fetal Health Classification.

Provides:
- get_tree_explainer: TreeExplainer setup
- compute_shap_values: Compute SHAP explanation object
- get_top_features_for_sample: Extract top contributing features with values
- get_top_feature_names_only: Extract feature name list for API responses
- plot_global_feature_importance: Global multi-class SHAP summary plot
- plot_local_patient_explanation: Individual patient explanation plot
"""

from src.explainability.shap_explain import (
    get_tree_explainer,
    compute_shap_values,
    get_top_features_for_sample,
    get_top_feature_names_only,
    plot_global_feature_importance,
    plot_local_patient_explanation,
    CLASS_NAMES_DEFAULT
)

__all__ = [
    "get_tree_explainer",
    "compute_shap_values",
    "get_top_features_for_sample",
    "get_top_feature_names_only",
    "plot_global_feature_importance",
    "plot_local_patient_explanation",
    "CLASS_NAMES_DEFAULT"
]
