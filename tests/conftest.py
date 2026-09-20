"""
Pytest Fixtures for Nexubot Strategy Engine.
Generates synthetic OHLCV pandas DataFrames representing various market conditions.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone


def generate_synthetic_candles(start_price: float, price_deltas: list, wicks: list = None) -> pd.DataFrame:
    """Helper to build consistent pandas DataFrames using cumulative price action."""
    count = len(price_deltas)
    timestamps = [datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(count)]

    closes = []
    current = start_price
    for delta in price_deltas:
        current += delta
        closes.append(current)

    opens = [c - d for c, d in zip(closes, price_deltas)]

    if wicks:
        highs = [max(o, c) + w[0] for o, c, w in zip(opens, closes, wicks)]
        lows = [min(o, c) - w[1] for o, c, w in zip(opens, closes, wicks)]
    else:
        highs = [max(o, c) + 2.0 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 2.0 for o, c in zip(opens, closes)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.uniform(100, 500, count),
        }
    )


@pytest.fixture
def bullish_bos_df() -> pd.DataFrame:
    """Phase 2: Bullish BOS generation."""
    deltas = [5.0] * 10 + [-4.0] * 5 + [6.0] * 10 + [-4.0] * 5 + [2.0] * 4 + [20.0] + [5.0] * 5
    return generate_synthetic_candles(2500.0, deltas)


@pytest.fixture
def bearish_choch_df() -> pd.DataFrame:
    """Phase 2: Bearish CHoCH generation."""
    deltas = [5.0] * 10 + [-4.0] * 5 + [6.0] * 10 + [-4.0] * 5 + [1.0] * 4 + [-20.0] + [-5.0] * 5
    return generate_synthetic_candles(2500.0, deltas)


@pytest.fixture
def ote_retracement_df() -> pd.DataFrame:
    """Phase 3: Bullish displacement followed by a 70.5% OTE retracement."""
    # Start at 2000. 10 candles up $10 each = High of 2100.
    # Displacement range: 2000 to 2100.
    # 70.5% retracement = pulls back down $70.5 to 2029.50.
    deltas = [10.0] * 10 + [-14.1] * 5
    return generate_synthetic_candles(1990.0, deltas)


@pytest.fixture
def liquidity_sweep_df() -> pd.DataFrame:
    """Phase 3: Creates Equal Lows, then sweeps them with a wick and closes back up."""
    # Base pattern ending at exactly 2500 for the close.
    deltas = [-5.0] * 5 + [5.0] * 5 + [-5.0] * 5 + [5.0] * 4
    wicks = [[2.0, 2.0]] * 19

    # Candle 19 (the sweep): price drops slightly, but wick goes deep, then recovers.
    deltas.append(-2.0)
    # High wick = 2, Low wick = 25.0 (sweeps to 2493, clearing the EQL at 2498)
    wicks.append([2.0, 25.0])

    return generate_synthetic_candles(2525.0, deltas, wicks)
