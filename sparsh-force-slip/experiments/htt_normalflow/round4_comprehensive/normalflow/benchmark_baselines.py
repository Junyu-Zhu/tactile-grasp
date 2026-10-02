#!/usr/bin/env python3
"""Measure existing NormalFlow baselines without fitting or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


HORIZONS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persistence(history: np.ndarray) -> np.ndarray:
    return np.repeat(history[:, -1:, :], HORIZONS, axis=1)


def linear_ar(history: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.einsum("nld,hdl->nhd", history, coefficients, optimize=True).astype(np.float32)


def measure(function, warmup: int, iterations: int, blocks: int) -> dict:
    result = None
    for _ in range(warmup):
        result = function()
    timings = []
    for _ in range(blocks):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            result = function()
        timings.append((time.perf_counter_ns() - start) / iterations / 1e6)
    assert result is not None and np.isfinite(result).all()
    values = np.asarray(timings, dtype=np.float64)
    return {
        "latency_ms_batch1_median": float(np.median(values)),
        "latency_ms_batch1_p10": float(np.quantile(values, 0.1)),
        "latency_ms_batch1_p90": float(np.quantile(values, 0.9)),
        "blocks": blocks,
        "iterations_per_block": iterations,
        "warmup_iterations": warmup,
        "output_checksum": float(np.sum(result, dtype=np.float64)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path, required=True)
    parser.add_argument("--neural-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--blocks", type=int, default=7)
    args = parser.parse_args()

    coefficients = np.load(args.coefficients, allow_pickle=False)
    with np.load(args.sample_cache, allow_pickle=False) as archive:
        history = np.asarray(archive["z"][:4], dtype=np.float32)[None]
    if coefficients.shape != (3, 768, 4) or history.shape != (1, 4, 768):
        raise ValueError(f"unexpected shapes: history={history.shape}, coefficients={coefficients.shape}")

    neural = json.loads(args.neural_summary.read_text())
    neural_rows = [
        {
            "variant": row["variant"],
            "seed": row["seed"],
            "parameters": row["parameters"],
            "module_latency_ms_batch1": row["module_latency_ms_batch1"],
            "device": row["device"],
        }
        for row in neural["runs"]
    ]
    payload = {
        "status": "complete",
        "scope": "module-only single-sample inference; no fit and no training",
        "baselines": {
            "persistence": {
                "learned_parameters": 0,
                "state_bytes": 0,
                **measure(lambda: persistence(history), 1_000, args.iterations, args.blocks),
            },
            "linear_ar_ridge_1e-3": {
                "learned_parameters": int(coefficients.size),
                "state_bytes": int(coefficients.nbytes),
                **measure(lambda: linear_ar(history, coefficients), 1_000, args.iterations, args.blocks),
            },
        },
        "neural_reference": neural_rows,
        "comparability_limit": (
            "Baseline timings use NumPy on CPU and include Python call overhead. Existing C/D timings use "
            "PyTorch CUDA module execution. They were measured on different devices and cannot be used for a "
            "direct speed ranking. None includes image preprocessing or the frozen MAE encoder."
        ),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "provenance": {
            "coefficients": str(args.coefficients.resolve()),
            "coefficients_sha256": sha256(args.coefficients),
            "sample_cache": str(args.sample_cache.resolve()),
            "sample_cache_sha256": sha256(args.sample_cache),
            "neural_summary": str(args.neural_summary.resolve()),
            "neural_summary_sha256": sha256(args.neural_summary),
            "benchmark_source_sha256": sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
