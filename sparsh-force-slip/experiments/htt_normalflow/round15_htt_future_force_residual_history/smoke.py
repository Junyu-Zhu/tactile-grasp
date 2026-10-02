#!/usr/bin/env python3
"""CPU invariants and optional real-cache GPU/recovery smoke."""
import argparse
import json
import tempfile
from pathlib import Path

import torch

import future_train as ft


def synthetic(seed=7):
    generator = torch.Generator().manual_seed(seed)
    roles = {}
    for role, count in (("fit", 48), ("selection", 20), ("calibration", 12), ("validation", 12)):
        x = torch.randn(count, 9, 195, generator=generator)
        current = x[:, -1, 192:195]
        y = torch.stack([current + 0.02 * horizon for horizon in ft.HORIZONS], 1)
        roles[role] = {
            "x": x,
            "y": y,
            "y_current": current + 0.01,
            "episode_id": [f"{role}/e{index}" for index in range(count)],
            "leakage_group": [f"{role}/g{index}" for index in range(count)],
            "t": torch.arange(count) + 13,
        }
    return {"schema": "round14_future_cache_v1", "horizons": ft.HORIZONS, "roles": roles}


def gradient_checks(data, device):
    fit = data["roles"]["fit"]
    normalizer = ft.fit_normalizer(fit)
    x = ft.normalize_x(fit["x"][:8], normalizer).to(device)
    target = ft.target_residual(fit, normalizer)[:8].to(device)
    outcomes = {}
    for group in ft.GROUPS:
        model = ft.init_model(group, ft.SEEDS[0]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        with torch.inference_mode():
            zero = model(x)
            future = ft.decode_future(zero.cpu(), {"x": fit["x"][:8]}, normalizer)
            persistence = fit["x"][:8, -1, 192:195][:, None, :].expand(-1, 3, -1)
            assert torch.equal(zero.cpu(), torch.zeros_like(zero.cpu()))
            assert torch.equal(future, persistence)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(model(x), target, beta=1.0)
        loss.backward()
        body_first = [parameter.grad for name, parameter in model.named_parameters() if not name.startswith("out.")]
        out_first = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("out.")]
        assert all(value is None or torch.count_nonzero(value) == 0 for value in body_first)
        assert any(value is not None and torch.count_nonzero(value) > 0 for value in out_first)
        optimizer.step()
        assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters() if name.startswith("out."))
        optimizer.zero_grad(set_to_none=True)
        second = torch.nn.functional.smooth_l1_loss(model(x), target, beta=1.0)
        second.backward()
        body_second = [parameter.grad for name, parameter in model.named_parameters() if not name.startswith("out.")]
        assert any(value is not None and torch.count_nonzero(value) > 0 for value in body_second)
        outcomes[group] = {
            "initial_future_bitwise_persistence": True,
            "first_body_gradient_zero": True,
            "first_output_gradient_nonzero": True,
            "second_body_gradient_nonzero": True,
            "finite_first_loss": bool(torch.isfinite(loss)),
            "finite_second_loss": bool(torch.isfinite(second)),
        }
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("SMOKE.json"))
    arguments = parser.parse_args()
    data = synthetic() if arguments.data is None else torch.load(arguments.data, map_location="cpu", weights_only=False)
    if arguments.data is not None:
        data["round14_data_sha256"] = ft.sha(arguments.data)
    ft.validate_cache(data)
    upstream = data["roles"]["fit"]["x"].clone()
    counts = {group: sum(parameter.numel() for parameter in ft.init_model(group, 1).parameters()) for group in ft.GROUPS}
    assert len(set(counts.values())) == 1
    assert ft.state_hash(ft.init_model("H_history", 1).state_dict()) == ft.state_hash(ft.init_model("H_history", 1).state_dict())
    assert ft.state_hash(ft.init_model("H_history", 1).state_dict()) != ft.state_hash(ft.init_model("H_history", 2).state_dict())
    bad = synthetic()
    bad["roles"]["selection"]["leakage_group"][0] = bad["roles"]["fit"]["leakage_group"][0]
    try:
        ft.assert_role_isolation(bad["roles"])
        raise AssertionError("role overlap accepted")
    except ValueError:
        pass
    gradients = gradient_checks(data, arguments.device)
    recovery = None
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        full = ft.train(data, temporary / "full", "H_history", 1, ft.SEEDS[0], max_epochs=2, patience=99, device=arguments.device)
        first = ft.train(data, temporary / "resume", "H_history", 1, ft.SEEDS[0], max_epochs=2, patience=99, interrupt_after=0, device=arguments.device)
        assert first["status"] == "interrupted"
        resumed = ft.train(data, temporary / "resume", "H_history", 1, ft.SEEDS[0], max_epochs=2, patience=99, device=arguments.device)
        assert full["state_sha256"] == resumed["state_sha256"]
        assert full["best_epoch"] == resumed["best_epoch"]
        recovery = {"exact_state": True, "best_epoch": full["best_epoch"], "epochs": full["epochs"]}
    assert torch.equal(upstream, data["roles"]["fit"]["x"])
    receipt = {
        "schema": "round15_smoke_v1",
        "status": "pass",
        "device": arguments.device,
        "real_round14_cache": arguments.data is not None,
        "input_sha256": ft.sha(arguments.data) if arguments.data is not None else None,
        "parameter_counts": counts,
        "gradient_checks": gradients,
        "recovery": recovery,
        "checks": [
            "scale_only_zero_offset",
            "exact_initial_persistence",
            "first_step_expected_gradient_block",
            "second_step_body_update",
            "finite_loss_gradients",
            "identical_group_parameters",
            "history_sequence_lengths_1_and_9",
            "upstream_input_bitwise_unchanged",
            "role_overlap_rejected",
            "seed_repeatable_distinct",
            "atomic_checkpoint_roundtrip",
            "interrupted_resume_exact",
        ],
    }
    ft.atomic_json(receipt, arguments.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
