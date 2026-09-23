"""Frozen terminal rewards for the oracle-proxy GRPO survival experiment.

This module only consumes terminal measurements. It does not perform source
inspection, shape a path, or transform the reference-agreement proxy.
"""

from __future__ import annotations

import math


ARMS = ("terminal", "test", "semantic")
BETA = 0.5


def _fraction(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return float(value)


def terminal_reward(
    arm: str, *, solved: bool, public_pass_fraction: float | None = None,
    reference_agreement: float | None = None,
) -> float:
    """Return Y, Y+0.5 p_T, or Y+0.5 q(P_T), without other shaping."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not isinstance(solved, bool):
        raise TypeError("solved must be a bool from the terminal verifier")
    if arm == "terminal":
        return float(solved)
    if arm == "test":
        return float(solved) + BETA * _fraction(public_pass_fraction, "p_T")
    return float(solved) + BETA * _fraction(reference_agreement, "q(P_T)")
