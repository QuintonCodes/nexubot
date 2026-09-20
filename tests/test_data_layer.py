"""
Unit tests for Data Layer components: Normalizer, Interval Parsing, and CandleStore.
Run with: python -m pytest tests/test_data_layer.py -v
"""

import pandas as pd
import pytest
from datetime import datetime, timezone

from data.normalizer import normalize_ohlcv
from data.candle_store import candle_store


def test_normalize_ohlcv_valid_data():
    """Test that valid Twelve Data responses are standardized with UTC timestamps."""
    # Hardcoded mock Twelve Data REST response
    raw_data = {
        "datetime": ["2026-09-17 10:00:00", "2026-09-17 10:05:00"],
        "open": ["2500.50", "2501.00"],
        "high": ["2502.00", "2503.50"],
        "low": ["2499.00", "2500.25"],
        "close": ["2501.00", "2502.75"],
        "volume": ["1050", "1100"],
    }
    df_raw = pd.DataFrame(raw_data)
    df_norm = normalize_ohlcv(df_raw)

    # Check strict schema enforcement
    assert list(df_norm.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    # Check data types
    assert df_norm["open"].dtype == "float64"
    assert df_norm["volume"].dtype == "float64"
    assert isinstance(df_norm["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(df_norm["timestamp"].dt.tz) == "UTC"
    assert df_norm["timestamp"].iloc[0] < df_norm["timestamp"].iloc[1]
    assert df_norm["close"].iloc[1] == 2502.75


def test_normalize_ohlcv_missing_columns():
    """Test that missing required columns raise a clear ValueError."""
    bad_data = pd.DataFrame(
        {
            "datetime": ["2026-09-17 10:00:00"],
            "open": ["2500.50"],
        }
    )

    with pytest.raises(ValueError, match="Normalization Failed: Missing required columns"):
        normalize_ohlcv(bad_data)


def test_normalize_ohlcv_chronological_sorting():
    """Test that out-of-order candles are sorted ascending by timestamp."""
    unordered_data = {
        "datetime": ["2026-09-17 10:10:00", "2026-09-17 10:00:00"],
        "open": [2502.0, 2500.0],
        "high": [2504.0, 2502.0],
        "low": [2501.0, 2499.0],
        "close": [2503.0, 2501.0],
        "volume": [100.0, 100.0],
    }
    df_norm = normalize_ohlcv(pd.DataFrame(unordered_data))
    assert df_norm["timestamp"].iloc[0] < df_norm["timestamp"].iloc[1]


def test_timeframe_interval_parser():
    """Test robust timeframe interval conversion."""

    def _parse_interval_minutes(tf: str) -> int:
        if tf.endswith("min"):
            return int(tf.replace("min", ""))
        if tf.endswith("h"):
            return int(tf.replace("h", "")) * 60
        raise ValueError(f"Unsupported timeframe format: {tf}")

    assert _parse_interval_minutes("5min") == 5
    assert _parse_interval_minutes("15min") == 15
    assert _parse_interval_minutes("1h") == 60
    assert _parse_interval_minutes("4h") == 240

    with pytest.raises(ValueError, match="Unsupported timeframe format"):
        _parse_interval_minutes("1d")


@pytest.mark.asyncio
async def test_candle_store_lifecycle():
    """Test initialization, appending, readiness checks, and in-place timestamp updates."""
    symbol = "XAUUSD"
    tf = "5min"

    # 1. Test Bulk Initialization
    historical_df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 17, 10, 5, tzinfo=timezone.utc),
            ],
            "open": [2500.0, 2501.0],
            "high": [2502.0, 2503.0],
            "low": [2499.0, 2500.0],
            "close": [2501.0, 2502.0],
            "volume": [100.0, 200.0],
        }
    )

    # 1. Bulk initialization
    await candle_store.initialize(symbol, tf, historical_df)
    assert await candle_store.is_ready(symbol, tf, min_candles=2) is True
    assert await candle_store.is_ready(symbol, tf, min_candles=100) is False

    # 2. Retrieval
    df_fetched = await candle_store.get_candles(symbol, tf)
    assert len(df_fetched) == 2

    # 3. Test Appending new candle (simulating WS tick)
    new_candle = pd.Series(
        {
            "timestamp": datetime(2026, 9, 17, 10, 10, tzinfo=timezone.utc),
            "open": 2502.0,
            "high": 2505.0,
            "low": 2501.0,
            "close": 2504.0,
            "volume": 300.0,
        }
    )

    await candle_store.add_candle(symbol, tf, new_candle)
    df_updated = await candle_store.get_candles(symbol, tf)
    assert len(df_updated) == 3
    assert df_updated.iloc[-1]["close"] == 2504.0

    # 4. In-place overwrite for matching timestamp
    updated_candle = new_candle.copy()
    updated_candle["close"] = 2506.0  # Late update on same timestamp
    await candle_store.add_candle(symbol, tf, updated_candle)

    df_final = await candle_store.get_candles(symbol, tf)
    assert len(df_final) == 3  # Did not append a new row
    assert df_final.iloc[-1]["close"] == 2506.0  # Overwrote the previous close
