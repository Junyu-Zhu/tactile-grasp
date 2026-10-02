"""Metrics whose exact definitions are frozen before Round 5 training."""
from __future__ import annotations

import numpy as np


def force_metrics(target_n: np.ndarray, prediction_n: np.ndarray) -> dict:
    target = np.asarray(target_n, dtype=np.float64)
    prediction = np.asarray(prediction_n, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("force arrays must have matching [N,3] shapes")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("force arrays must be finite")
    rmse_xyz = np.sqrt(np.mean(np.square(prediction - target), axis=0))
    return {"rmse_xyz_native_n": rmse_xyz.tolist(), "mean_rmse_xyz_native_n": float(rmse_xyz.mean())}


def partial_tpr_auc_0_0p1(labels: np.ndarray, probabilities: np.ndarray, max_fpr: float = 0.1) -> float:
    """Area under TPR(FPR) on [0,max_fpr], divided by max_fpr.

    This is a direct normalized truncated area in [0,1], rather than sklearn's
    standardized partial AUC transformation.
    """
    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not np.isin(y, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError("labels/scores must be finite aligned binary vectors")
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if not positives or not negatives or not 0 < max_fpr <= 1:
        raise ValueError("both classes and a valid max_fpr are required")
    order = np.argsort(-score, kind="stable")
    sorted_y, sorted_score = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y) - 1]
    tp = np.cumsum(sorted_y)[ends]
    fp = np.cumsum(1 - sorted_y)[ends]
    fpr = np.r_[0.0, fp / negatives]
    tpr = np.r_[0.0, tp / positives]
    within = fpr <= max_fpr
    clipped_fpr, clipped_tpr = fpr[within], tpr[within]
    if clipped_fpr[-1] < max_fpr:
        above_index = int(np.flatnonzero(fpr > max_fpr)[0])
        below_index = above_index - 1
        fraction = (max_fpr - fpr[below_index]) / (fpr[above_index] - fpr[below_index])
        boundary_tpr = tpr[below_index] + fraction * (tpr[above_index] - tpr[below_index])
        clipped_fpr = np.r_[clipped_fpr, max_fpr]
        clipped_tpr = np.r_[clipped_tpr, boundary_tpr]
    widths = np.diff(clipped_fpr)
    area = np.sum(widths * (clipped_tpr[:-1] + clipped_tpr[1:]) * 0.5)
    return float(area / max_fpr)


def force_target_normalization(train_force_n: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train_force_n, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("train force must be finite [N,3]")
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-6)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def fit_force_condition_normalization(train_force_n: np.ndarray) -> dict[str, np.ndarray | float]:
    values = np.asarray(train_force_n, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("train force must be finite [N,3]")
    fn = np.abs(values[:, 2])
    epsilon = max(1e-3, float(np.quantile(fn, 0.01)))
    ft = np.linalg.norm(values[:, :2], axis=1)
    # Deltas are fitted by the caller on strict valid t,t-5 pairs; this helper
    # establishes the physical epsilon only.
    return {"ratio_epsilon_n": epsilon}


def fit_clip_standardize(train_conditions: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train_conditions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("train conditions must be finite [N,5]")
    lower = np.quantile(values, 0.005, axis=0)
    upper = np.quantile(values, 0.995, axis=0)
    clipped = np.clip(values, lower, upper)
    mean, std = clipped.mean(axis=0), clipped.std(axis=0)
    std = np.maximum(std, 1e-6)
    return {"clip_lower": lower.astype(np.float32), "clip_upper": upper.astype(np.float32),
            "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_clip_standardize(values: np.ndarray, normalization: dict[str, np.ndarray]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return ((np.clip(array, normalization["clip_lower"], normalization["clip_upper"])
             - normalization["mean"]) / normalization["std"]).astype(np.float32)
