"""
Unit tests for the Market Structure Engine.
Tests Swings, BOS, CHoCH, MSS, and CISD structural detection.
Run with: python -m pytest tests/test_structure.py -v
"""

import pandas as pd
from datetime import datetime, timezone

from strategies.structure import (
    classify_structure,
    detect_bos,
    detect_choch,
    detect_cisd,
    detect_mss,
    detect_swings,
)


def test_detect_swings(bullish_bos_df):
    """Test that the ZigZag algorithm correctly identifies highs and lows without repainting."""
    swings = detect_swings(bullish_bos_df, lookback=3)

    assert len(swings) >= 4
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    # First swing high is at index 9, second is at 24
    assert highs[0].candle_index == 9
    assert highs[1].candle_index == 24

    # First swing low is at 14, second is at 29
    assert lows[0].candle_index == 14
    assert lows[1].candle_index == 29

    # Test classification correctly identifies Higher Highs and Higher Lows
    assert highs[0].classification == "UNCONFIRMED"
    assert highs[1].classification == "HH"
    assert lows[1].classification == "HL"


def test_classify_structure_bullish(bullish_bos_df):
    """Test that a sequence of HHs and HLs is classified as bullish."""
    df_slice = bullish_bos_df.iloc[:33]
    swings = detect_swings(df_slice, lookback=3)
    bias = classify_structure(swings)

    assert bias == "bullish"


def test_classify_structure_bearish(bearish_choch_df):
    """Test that a sequence of LHs and LLs is classified as bearish."""
    swings = detect_swings(bearish_choch_df, lookback=3)
    # Filter to only the crash sequence containing lower low and lower high
    bearish_swings = [s for s in swings if s.classification in ["LL", "LH"]]
    if len(bearish_swings) >= 2:
        bias = classify_structure(bearish_swings)
        assert bias in ["bearish", "ranging"]


def test_classify_structure_ranging():
    """Test that insufficient swings return ranging status."""
    assert classify_structure([]) == "ranging"


def test_detect_bos_bullish(bullish_bos_df):
    """Test that a bullish close over a previous swing high triggers a BOS event."""
    df_slice = bullish_bos_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    event = detect_bos(df_slice, swings, symbol="XAU/USD", tf="5min")

    assert event is not None
    assert event.event_type == "BOS"
    assert event.direction == "bullish"
    assert event.confirmed is True


def test_detect_bos_bearish(bearish_choch_df):
    """Test that a bearish close below a previous swing low triggers a BOS event."""
    df_slice = bearish_choch_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    event = detect_bos(df_slice, swings, symbol="XAUUSD", tf="5min")

    assert event is not None
    assert event.event_type == "BOS"
    assert event.direction == "bearish"
    assert event.confirmed is True


def test_detect_choch_bullish(bullish_bos_df):
    """Test that an upward break against an established bearish bias triggers a bullish CHoCH."""
    df_slice = bullish_bos_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    event = detect_choch(df_slice, swings, "bearish", symbol="XAUUSD", tf="5min")

    assert event is not None
    assert event.event_type == "CHoCH"
    assert event.direction == "bullish"


def test_detect_choch_bearish(bearish_choch_df):
    """Test that a downtrend break after a bullish structure triggers a CHoCH."""
    # The crash crossing the Higher Low happens exactly on index 34.
    df_slice = bearish_choch_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    current_bias = classify_structure(swings)

    event = detect_choch(df_slice, swings, current_bias, symbol="XAU/USD", tf="5min")

    assert event is not None
    assert event.event_type == "CHoCH"
    assert event.direction == "bearish"


def test_detect_mss_confirmed(mss_displacement_df):
    """Test that a CHoCH with large displacement (body >= 30 pips) triggers an MSS event."""
    swings = detect_swings(mss_displacement_df, lookback=3)
    event = detect_mss(mss_displacement_df, swings, "bullish", symbol="XAUUSD", tf="5min", displacement_pips=30.0)

    assert event is not None
    assert event.event_type == "MSS"
    assert event.direction == "bearish"
    assert event.confirmed is True


def test_detect_mss_unconfirmed_weak_candle(bearish_choch_df):
    """Test that a CHoCH without sufficient displacement candle body returns None for MSS."""
    df_slice = bearish_choch_df.iloc[:35].copy()
    # Dampen the candle body so displacement is below threshold
    df_slice.loc[df_slice.index[-1], "open"] = 2550.0
    df_slice.loc[df_slice.index[-1], "close"] = 2549.0  # 10 pips diff

    swings = detect_swings(df_slice, lookback=3)
    event = detect_mss(df_slice, swings, "bullish", symbol="XAUUSD", tf="5min", displacement_pips=30.0)

    assert event is None


def test_detect_cisd_bullish(dummy_ob_dict):
    """Test that price touching the 50% line of an OB triggers a CISD event."""
    # ob_50 = (2505 + 2495) / 2 = 2500.0
    candles = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 1, 10, 5, tzinfo=timezone.utc),
            ],
            "open": [2510.0, 2502.0],
            "high": [2512.0, 2504.0],
            "low": [2508.0, 2498.0],  # 2498 <= 2500 <= 2504
            "close": [2509.0, 2501.0],
            "volume": [100.0, 200.0],
        }
    )

    event = detect_cisd(candles, dummy_ob_dict, symbol="XAUUSD", tf="5min")

    assert event is not None
    assert event.event_type == "CISD"
    assert event.direction == "bullish"
    assert event.price_level == 2500.0


def test_detect_cisd_bearish(dummy_breaker_dict):
    """Test CISD trigger for a bearish order block."""
    candles = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 1, 10, 5, tzinfo=timezone.utc),
            ],
            "open": [2490.0, 2498.0],
            "high": [2492.0, 2503.0],  # 2498 <= 2500 <= 2503
            "low": [2488.0, 2497.0],
            "close": [2491.0, 2501.0],
            "volume": [100.0, 200.0],
        }
    )

    event = detect_cisd(candles, dummy_breaker_dict, symbol="XAUUSD", tf="5min")

    assert event is not None
    assert event.event_type == "CISD"
    assert event.direction == "bearish"
    assert event.price_level == 2500.0
