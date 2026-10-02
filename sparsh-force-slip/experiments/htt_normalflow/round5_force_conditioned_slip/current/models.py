"""Round 5 force adapter and identity-initialized slip fusion modules."""
from __future__ import annotations

import copy
from typing import Literal

import torch
from torch import nn


class ForceAdapter(nn.Module):
    """Warm-start the private force representation and predict native HTT z-scores.

    The caller owns train-only target normalization. This module deliberately has
    no tanh because the new output coordinates/range differ from the old task.
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        required = ("force_pooler", "force_trunk", "force_head")
        if not all(hasattr(decoder, name) for name in required):
            raise TypeError("Force adaptation requires the decoupled force branch")
        self.force_pooler = copy.deepcopy(decoder.force_pooler)
        self.force_trunk = copy.deepcopy(decoder.force_trunk)
        old_head = decoder.force_head
        if not isinstance(old_head, nn.Linear) or old_head.out_features != 3:
            raise TypeError("Expected a three-output linear force head")
        self.force_head = nn.Linear(old_head.in_features, 3)

    def forward_features(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.force_trunk(self.force_pooler(tokens).squeeze(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.force_head(self.forward_features(tokens))


class FrozenOldForce(nn.Module):
    """Historical tanh force branch expressed in its recorded Newton scale."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.force_pooler = copy.deepcopy(decoder.force_pooler)
        self.force_trunk = copy.deepcopy(decoder.force_trunk)
        self.force_head = copy.deepcopy(decoder.force_head)
        self.register_buffer("scale_n", torch.tensor([1.5, 1.5, 2.0]))
        self.eval().requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feature = self.force_trunk(self.force_pooler(tokens).squeeze(1))
            return torch.tanh(self.force_head(feature)) * self.scale_n


class FrozenSlipBranch(nn.Module):
    """A copied Round 3-B branch exposing its frozen trunk feature."""

    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        required = ("slip_pooler", "slip_trunk", "slip_head")
        if not all(hasattr(branch, name) for name in required):
            raise TypeError("Expected a complete decoupled slip branch")
        self.slip_pooler = copy.deepcopy(branch.slip_pooler)
        self.slip_trunk = copy.deepcopy(branch.slip_trunk)
        self.slip_head = copy.deepcopy(branch.slip_head)
        self.eval().requires_grad_(False)

    def forward_features(self, tokens: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.slip_trunk(self.slip_pooler(tokens).squeeze(1)).detach()

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.slip_head(features).detach()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.logits_from_features(self.forward_features(tokens))


def zero_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class ForceConditionedSlip(nn.Module):
    """Frozen base slip head plus identity-initialized FiLM and logit residual.

    `V` learns a five-dimensional condition from frozen visual features. Force
    variants receive the same five physical condition scalars from the caller.
    """

    def __init__(self, base_branch: nn.Module, variant: Literal["V", "F-old", "F-adapt"],
                 condition_dim: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        self.variant = variant
        self.base = FrozenSlipBranch(base_branch)
        head = self.base.slip_head
        if not isinstance(head, nn.Linear) or head.out_features != 2:
            raise TypeError("Expected a two-class linear slip head")
        feature_dim = head.in_features
        self.visual_conditioner = nn.Linear(feature_dim, condition_dim) if variant == "V" else None
        self.film = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, feature_dim * 2)
        )
        self.residual = nn.Sequential(
            nn.Linear(feature_dim + condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )
        zero_linear(self.film[-1])
        zero_linear(self.residual[-1])

    def train(self, mode: bool = True) -> "ForceConditionedSlip":
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor | None = None,
                *, return_aux: bool = False):
        feature = self.base.forward_features(tokens)
        base_logits = self.base.logits_from_features(feature)
        if self.variant == "V":
            if condition is not None:
                raise ValueError("V computes its condition from frozen visual features")
            assert self.visual_conditioner is not None
            condition = self.visual_conditioner(feature)
        elif condition is None:
            raise ValueError(f"{self.variant} requires a five-dimensional force condition")
        if condition.ndim != 2 or condition.shape != (len(feature), self.film[0].in_features):
            raise ValueError(f"Invalid condition shape {tuple(condition.shape)}")
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        modulated = feature * (1.0 + gamma) + beta
        logits = base_logits + self.residual(torch.cat((modulated, condition), dim=-1))
        if return_aux:
            return {"logits": logits, "base_logits": base_logits, "condition": condition,
                    "gamma": gamma, "beta": beta, "feature": feature}
        return logits

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def physical_condition(force_now_n: torch.Tensor, force_previous_n: torch.Tensor,
                       ratio_epsilon_n: float) -> torch.Tensor:
    """Return Fn, Ft, Ft/(Fn+eps), delta-Fn and delta-Ft in native units."""
    if force_now_n.shape != force_previous_n.shape or force_now_n.ndim != 2 or force_now_n.shape[1] != 3:
        raise ValueError("Force inputs must both have shape [B,3]")
    if ratio_epsilon_n <= 0:
        raise ValueError("ratio_epsilon_n must be positive")
    fn = force_now_n[:, 2].abs()
    ft = torch.linalg.vector_norm(force_now_n[:, :2], dim=1)
    previous_fn = force_previous_n[:, 2].abs()
    previous_ft = torch.linalg.vector_norm(force_previous_n[:, :2], dim=1)
    return torch.stack((fn, ft, ft / (fn + ratio_epsilon_n), fn - previous_fn, ft - previous_ft), dim=1)
