#!/usr/bin/env python3
"""Post-process six completed NormalFlow runs without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_normalflow as nf


def load_stats(path: Path) -> dict:
    with np.load(path) as z:
        return {key: {stat: z[f"{key}_{stat}"] for stat in ("mean", "std")} for key in ("z", "delta", "motion")}


@torch.no_grad()
def predictions(checkpoint_path: Path, val: dict, stats: dict, device: str):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = nf.DynamicsModel(hidden=payload["config"]["hidden"], auxiliary_motion=payload["variant"] == "D").to(device)
    model.load_state_dict(payload["model"], strict=True); model.eval()
    dataset = nf.WindowDataset(val, stats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    st = nf.tensor_stats(stats, device); feats, motions, indices = [], [], []
    for x, _, _, idx in loader:
        x = x.to(device); delta, motion = model(x)
        feats.append(nf.reconstruct_raw(x, delta, st).cpu().numpy())
        if motion is not None:
            raw_motion = motion.cpu().numpy() * stats["motion"]["std"] + stats["motion"]["mean"]
            motions.append(raw_motion)
        indices.append(idx.numpy())
    order = np.argsort(np.concatenate(indices))
    return np.concatenate(feats)[order], (np.concatenate(motions)[order] if motions else None)


def motion_metrics(pred: np.ndarray, val: dict) -> dict:
    result = {"overall": {}, "by_object": {}}
    for hi, horizon in enumerate(nf.HORIZONS):
        translation = np.mean((pred[:, hi, :3] - val["motion"][:, hi, :3]) ** 2, axis=1)
        rotation = np.mean((pred[:, hi, 3:] - val["motion"][:, hi, 3:]) ** 2, axis=1)
        result["overall"][str(horizon)] = {"translation_mse_m2": float(translation.mean()),
                                            "rotation_log_mse_rad2": float(rotation.mean())}
        for obj in nf.VAL_OBJECTS:
            mask = val["object"] == obj
            result["by_object"].setdefault(obj, {})[str(horizon)] = {
                "translation_mse_m2": float(translation[mask].mean()),
                "rotation_log_mse_rad2": float(rotation[mask].mean())}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); root = args.output
    episodes, _ = nf.load_cache(args.cache, audit_hashes=True)
    val = nf.build_arrays(episodes, nf.VAL_OBJECTS); stats = load_stats(root / "train_only_standardizers.npz")
    baseline = json.loads((root / "baselines.json").read_text())
    persistence = baseline["persistence"]["metrics"]["overall"]
    linear = baseline["linear_ar_ridge_1e-3"]["metrics"]["overall"]
    runs = []
    for metrics_path in sorted((root / "runs").glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text()); feat, motion = predictions(metrics_path.parent / "best.pt", val, stats, args.device)
        row = {"variant": metrics["variant"], "seed": metrics["seed"], "best_epoch": metrics["best_epoch"],
               "feature_metrics": metrics["metrics"], "variance_check": metrics["variance_check"],
               "motion_metrics": motion_metrics(motion, val) if motion is not None else None}
        runs.append(row)
    means = {}
    for variant in ("C", "D"):
        subset = [r for r in runs if r["variant"] == variant]
        means[variant] = {}
        for horizon in nf.HORIZONS:
            key = str(horizon); value = float(np.mean([r["feature_metrics"]["overall"][key] for r in subset]))
            means[variant][key] = {"mse": value,
                                   "improvement_vs_persistence_percent": 100 * (float(persistence[key]) - value) / float(persistence[key]),
                                   "improvement_vs_linear_ar_percent": 100 * (float(linear[key]) - value) / float(linear[key])}
    analysis = {"status": "complete" if len(runs) == 6 else "incomplete", "runs": len(runs),
                "feature_mse": {"persistence": persistence, "linear_ar": linear, **means},
                "motion_auxiliary_validation": [r for r in runs if r["variant"] == "D"],
                "claim_limit": "two validation objects; pooled-feature prediction only; no action inputs or slip labels"}
    nf_path = root / "analysis.json"; nf_path.write_text(json.dumps(analysis, indent=2) + "\n")
    lines = ["# NormalFlow 短期状态预测结论", "",
             "已完成 C（GRU 特征预测）和 D（GRU＋相对运动辅助监督）各 3 个种子，共 6 次训练。MAE 编码器未训练；只使用 56 个训练试次，验证为 table、bead 共 14 个试次，seed、ball 未读取。", "",
             "## 冻结特征 MSE", "", "| 方法 | H=1 | H=3 | H=5 |", "|---|---:|---:|---:|",
             f"| 保持当前状态 | {float(persistence['1']):.7f} | {float(persistence['3']):.7f} | {float(persistence['5']):.7f} |",
             f"| 线性 AR | {float(linear['1']):.7f} | {float(linear['3']):.7f} | {float(linear['5']):.7f} |",
             f"| C：GRU，3种子均值 | {means['C']['1']['mse']:.7f} | {means['C']['3']['mse']:.7f} | {means['C']['5']['mse']:.7f} |",
             f"| D：GRU＋运动监督，3种子均值 | {means['D']['1']['mse']:.7f} | {means['D']['3']['mse']:.7f} | {means['D']['5']['mse']:.7f} |", "",
             "C 相比保持不变基线在三个窗口均有改善，但没有稳定超过线性 AR：H=3 略好，H=1 和 H=5 较差。D 在三个窗口均差于线性 AR，并且较 C 更差，因此本轮没有证据表明运动辅助监督改善了冻结特征预测。", "",
             "预测并非完全输出常数，但预测增量方差仅覆盖目标增量方差的一部分，D 的覆盖尤其低。这提示模型倾向保守变化，低 MSE 不能单独支持‘世界模型已有效’。", "",
             "## 结论边界", "",
             "当前结果只支持：小型因果模块能学到部分短期触觉变化。由于验证只有两个物体、没有动作输入、没有滑移监督，且神经模型没有稳定超过简单线性基线，暂不能声称已获得可靠的轻量级世界模型。", ""]
    (root / "SUMMARY_ZH.md").write_text("\n".join(lines))
    try:
        import matplotlib.pyplot as plt
        x = np.arange(3); width = .2
        series = [("Persistence", [persistence[str(h)] for h in nf.HORIZONS]),
                  ("Linear AR", [linear[str(h)] for h in nf.HORIZONS]),
                  ("GRU C", [means["C"][str(h)]["mse"] for h in nf.HORIZONS]),
                  ("GRU+motion D", [means["D"][str(h)]["mse"] for h in nf.HORIZONS])]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (name, values) in enumerate(series): ax.bar(x + (i - 1.5) * width, values, width, label=name)
        ax.set_xticks(x, [f"H={h}" for h in nf.HORIZONS]); ax.set_ylabel("Validation feature MSE")
        ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(root / "feature_mse.png", dpi=160); plt.close(fig)
    except ImportError: pass
    print(json.dumps({"status": analysis["status"], "runs": len(runs), "output": str(root)}, indent=2))


if __name__ == "__main__": main()

