"""Deterministic portfolio budget constraints."""

from __future__ import annotations

import math
from typing import Any


def apply_long_only(weights: dict[str, float]) -> dict[str, float]:
    """Remove invalid and non-positive proposed budgets without reinvesting cash."""
    return {ticker: float(value) for ticker, value in sorted(weights.items()) if math.isfinite(value) and value > 0.0}


def apply_min_position(weights: dict[str, float], min_weight: float = 0.05) -> dict[str, float]:
    """Drop positive budgets below the configured minimum."""
    if min_weight < 0.0:
        raise ValueError("min_weight must be nonnegative")
    return {ticker: value for ticker, value in sorted(weights.items()) if value >= min_weight}


def apply_max_position(weights: dict[str, float], max_weight: float = 0.30) -> dict[str, float]:
    """Cap each budget without scaling the remaining assets upward."""
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    return {ticker: min(float(value), max_weight) for ticker, value in sorted(weights.items())}


def apply_all_constraints(weights: dict[str, float], config: Any) -> dict[str, float]:
    """Apply positive filtering, cap, minimum drop, then gross-limit scaling."""
    cleaned = apply_long_only(weights)
    capped = apply_max_position(cleaned, float(config.max_position_size))
    filtered = apply_min_position(capped, float(config.min_position_size))
    total = sum(filtered.values())
    gross_limit = float(config.gross_limit)
    if total > gross_limit and total > 0.0:
        scale = gross_limit / total
        filtered = {ticker: value * scale for ticker, value in filtered.items()}
    return filtered


def validate_budget(weights: dict[str, float], *, gross_limit: float, tolerance: float = 1e-9) -> None:
    if any(not math.isfinite(value) or value < -tolerance for value in weights.values()):
        raise ValueError("Budgets must be finite and nonnegative")
    if sum(weights.values()) > gross_limit + tolerance:
        raise ValueError("Budgets exceed the gross limit")
