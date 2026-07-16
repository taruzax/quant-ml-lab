"""
Each function returns a new dict (no mutation) with weights that sum to ~1.0.
This is done intentionaly for debugging and logging
"""
from __future__ import annotations
from lab.core.config import PipelineConfig

def apply_long_only(weights):
    """Safety function against negative weights"""
    if min(weights.values()) >= 0.0:
        if abs(sum(weights.values()) - 1.0) < 1e-6:
            return weights.copy() 
    clipped = {k: max(v, 0.0) for k, v in weights.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}

def apply_min_position(weights, min_weight: float = 0.05):
    """Zero out positions below min_weight"""
    if min_weight<=0:
        return dict(weights)
    
    surviving = {k: v for k,v in weights.items() if v>=min_weight}
    if not surviving:
        # all weights below threshold, return equal weight
        return {k: 1.0 / len(weights) for k in weights}

    total_surviving = sum(surviving.values())
    return {
        k: (v / total_surviving if k in surviving else 0.0) 
        for k, v in weights.items()
    }

def apply_max_position(weights, max_weight: float = 0.3):
    """Cap weights at max_weight and redistribute"""
    n = len(weights)
    if n == 1:
        return {k: 1.0 for k in weights}

    if max_weight < 1.0 / n:
        raise ValueError(f"max_weight={max_weight} < 1/{n}={1.0/n:.4f}")

    if max(weights.values()) <= max_weight + 1e-10:
        return weights.copy()

    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)

    capped = 0
    remaining_budget = 1.0
    remaining_sum = sum(weights.values())

    for _, v in items:
        if v * remaining_budget <= max_weight * remaining_sum + 1e-12:
            break
        capped += 1
        remaining_budget -= max_weight
        remaining_sum -= v

    if remaining_sum <= 0:
        return {k: 1.0 / n for k in weights}

    scale = remaining_budget / remaining_sum
    capped_keys = {k for k, _ in items[:capped]}

    return {
        k: max_weight if k in capped_keys else v * scale
        for k, v in weights.items()
    }


def apply_all_constraints(
    weights: dict[str, float], config: PipelineConfig):
    weights = apply_long_only(weights)
    weights = apply_min_position(weights, min_weight=config.min_position_size)
    weights = apply_max_position(weights, max_weight=config.max_position_size)

    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected ~1.0"

    return weights