"""
Unit tests for the Market Structure Engine (Phase 2).
Run with: python -m pytest tests/test_structure.py -v
"""

from strategies.structure import detect_swings, classify_structure, detect_bos, detect_choch


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


def test_classify_structure(bullish_bos_df):
    """Test that a sequence of HHs and HLs is classified as bullish."""
    # Slice before the massive breakout so we evaluate the established HH/HL sequence
    df_slice = bullish_bos_df.iloc[:33]
    swings = detect_swings(df_slice, lookback=3)
    bias = classify_structure(swings)

    assert bias == "bullish"


def test_detect_bos_bullish(bullish_bos_df):
    """Test that a bullish close over a previous swing high triggers a BOS event."""
    # The BOS happens exactly on index 34.
    # We pass the DataFrame up to index 34 (length 35) so it detects it on the final candle.
    df_slice = bullish_bos_df.iloc[:35]
    swings = detect_swings(df_slice, lookback=3)

    event = detect_bos(df_slice, swings, symbol="XAU/USD", tf="5min")

    assert event is not None
    assert event.event_type == "BOS"
    assert event.direction == "bullish"
    assert event.confirmed is True


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
