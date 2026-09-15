from lab.core.config import TaskConfig


def barrier_prices(entry_price: float, volatility: float, config: TaskConfig) -> tuple[float, float]:
    """Return absolute upper/lower prices anchored at the actual entry open."""
    barrier = config.triple_barrier
    return (
        entry_price * (1.0 + barrier.profit_taking * volatility),
        entry_price * (1.0 - barrier.stop_loss * volatility),
    )


def first_barrier_touch(
    closes: list[float], upper: float, lower: float
) -> tuple[int | None, int]:
    """Find the first close touch, with deterministic upper precedence on ties."""
    for offset, close in enumerate(closes):
        if close >= upper:
            return offset, 1
        if close <= lower:
            return offset, -1
    return None, 0
