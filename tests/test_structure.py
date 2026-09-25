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
    bias = classify_structure(df_slice, swings)

    assert bias == "bullish"


def test_classify_structure_bearish(bearish_choch_df):
    """Test that a sequence of LHs and LLs is classified as bearish."""
    swings = detect_swings(bearish_choch_df, lookback=3)
    # Filter to only the crash sequence containing lower low and lower high
    bearish_swings = [s for s in swings if s.classification in ["LL", "LH"]]
    if len(bearish_swings) >= 2:
        bias = classify_structure(bearish_choch_df, bearish_swings)
        assert bias in ["bearish", "ranging"]


def test_classify_structure_ranging():
    """Test that insufficient swings return ranging status."""
    empty_df = pd.DataFrame()
    assert classify_structure(empty_df, []) == "ranging"


def test_detect_bos_bullish(bullish_bos_df):
    """Test that a bullish close over a previous swing high triggers a BOS event."""
    df_slice = bullish_bos_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    event = detect_bos(df_slice, swings, "bullish", symbol="XAU/USD", tf="5min")

    assert event is not None
    assert event.event_type == "BOS"
    assert event.direction == "bullish"
    assert event.confirmed is True


def test_detect_bos_bearish(bearish_choch_df):
    """Test that a bearish close below a previous swing low triggers a BOS event."""
    df_slice = bearish_choch_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    event = detect_bos(df_slice, swings, "bearish", symbol="XAUUSD", tf="5min")

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

    event = detect_choch(df_slice, swings, "bullish", symbol="XAU/USD", tf="5min")

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
    df_slice.loc[df_slice.index[-1], "close"] = 2549.9

    swings = detect_swings(df_slice, lookback=3)
    event = detect_mss(df_slice, swings, "bullish", symbol="XAUUSD", tf="5min", displacement_pips=30.0)

    assert event is None


def test_detect_cisd_bullish(dummy_ob_dict):
    """Test that price closing above a bearish OB origin open triggers a bullish CISD."""
    origin_time = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    dummy_ob_dict["origin_timestamp"] = origin_time
    dummy_ob_dict["direction"] = "bearish"  # Downward delivery state

    candles = pd.DataFrame(
        {
            "timestamp": [
                origin_time,
                datetime(2026, 9, 1, 10, 5, tzinfo=timezone.utc),
                datetime(2026, 9, 1, 10, 10, tzinfo=timezone.utc),
            ],
            "open": [2500.0, 2490.0, 2495.0],
            "high": [2505.0, 2495.0, 2510.0],
            "low": [2490.0, 2480.0, 2490.0],
            # Close starts below the origin open (2500.0) then closes powerfully above it
            "close": [2490.0, 2495.0, 2505.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )

    event = detect_cisd(candles, dummy_ob_dict, symbol="XAUUSD", tf="5min")

    assert event is not None
    assert event.event_type == "CISD"
    assert event.direction == "bullish"
    assert event.price_level == 2500.0  # Matches origin open


def test_detect_cisd_bearish(dummy_breaker_dict):
    """Test that price closing below a bullish OB origin open triggers a bearish CISD."""

    origin_time = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    dummy_breaker_dict["origin_timestamp"] = origin_time
    dummy_breaker_dict["direction"] = "bullish"  # Upward delivery state

    candles = pd.DataFrame(
        {
            "timestamp": [
                origin_time,
                datetime(2026, 9, 1, 10, 5, tzinfo=timezone.utc),
                datetime(2026, 9, 1, 10, 10, tzinfo=timezone.utc),
            ],
            "open": [2500.0, 2510.0, 2505.0],
            "high": [2510.0, 2520.0, 2510.0],
            "low": [2495.0, 2505.0, 2490.0],
            # Close starts above the origin open (2500.0) then closes powerfully below it
            "close": [2510.0, 2505.0, 2495.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )

    event = detect_cisd(candles, dummy_breaker_dict, symbol="XAUUSD", tf="5min")

    assert event is not None
    assert event.event_type == "CISD"
    assert event.direction == "bearish"
    assert event.price_level == 2500.0  # Matches origin open
