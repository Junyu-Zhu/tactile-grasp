"""Round-8 future-slip evaluation primitives."""

from .core import (
    add_causal_rules,
    apply_shared_platt,
    cumulative_risk_from_hazard,
    event_metrics,
    fit_shared_monotone_platt,
    fit_state_linear_baseline,
    frame_metrics,
    paired_cluster_bootstrap,
    predict_state_linear_baseline,
    select_threshold,
)
from .schema import load_timeline_csv, validate_cross_run_identity

__all__ = [
    "add_causal_rules",
    "apply_shared_platt",
    "cumulative_risk_from_hazard",
    "event_metrics",
    "fit_shared_monotone_platt",
    "fit_state_linear_baseline",
    "frame_metrics",
    "load_timeline_csv",
    "paired_cluster_bootstrap",
    "predict_state_linear_baseline",
    "select_threshold",
    "validate_cross_run_identity",
]
