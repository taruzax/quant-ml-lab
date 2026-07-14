"""
Each function returns a new dict (no mutation) with weights that sum to ~1.0.
This is done intentionaly for debugging and logging
"""
from __future__ import annotations

from src.core.config import PipelineConfig


class ConstraintConvergenceError(Exception):
    """Raised when iterative constraint application fails to converge."""


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
    removed_weight = sum(v for v in weights.values() if v<min_weight)
    if not surviving:
        # all weights below threshold, return equal weight
        return {k: 1.0 / len(weights) for k in weights}
    total_surviving = sum(surviving.values())
    return {
        k: (v / total_surviving if k in surviving else 0.0) 
        for k, v in weights.items()
    }