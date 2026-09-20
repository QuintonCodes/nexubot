"""
Unit tests for the SMC Entry Models (Phase 3).
Run with: python -m pytest tests/test_smc.py -v
"""

import pandas as pd

from strategies.structure import detect_swings
from strategies.smc import (
    find_order_blocks,
    is_ob_mitigated,
    calculate_ote_zone,
    detect_liquidity_pools,
    detect_liquidity_sweep,
    generate_trade_signal,
)


def test_find_and_mitigate_order_block(bullish_bos_df):
    """Test OB creation and 50% mitigation invalidation."""
    swings = detect_swings(bullish_bos_df, lookback=3)

    # 1. Find the OB
    obs = find_order_blocks(bullish_bos_df, swings, direction="bullish", symbol="XAU/USD", tf="5min")
    assert len(obs) > 0
    ob = obs[0]

    # Ensure it mapped to a bearish down-candle before the swing low
    assert ob.ob_high > ob.ob_low
    assert ob.is_mitigated is False

    # 2. Test Mitigation
    # Initially not mitigated
    assert is_ob_mitigated(bullish_bos_df, ob) is False

    # Manually append a candle that crashes through the OB 50% line to test mitigation
    crash_row = bullish_bos_df.iloc[-1].copy()
    crash_row["timestamp"] = crash_row["timestamp"] + pd.Timedelta(minutes=5)
    crash_row["close"] = ob.ob_50 - 5.0  # Close below 50% line

    mitigated_df = pd.concat([bullish_bos_df, crash_row.to_frame().T], ignore_index=True)

    assert is_ob_mitigated(mitigated_df, ob) is True


def test_calculate_ote_zone():
    """Test standard Fibonacci retracement mathematics."""
    zone = calculate_ote_zone(swing_low=2000.0, swing_high=2100.0, direction="bullish", symbol="XAU/USD", tf="5min")

    assert zone.ote_entry == 2061.8  # 61.8% from bottom = 38.2% from top
    assert zone.ote_mid == 2070.5  # 70.5%
    assert zone.ote_top == 2078.6  # 78.6%


def test_detect_liquidity_sweep(liquidity_sweep_df):
    """Test that EQH/EQL are identified, and wick-closures confirm sweeps."""
    swings = detect_swings(liquidity_sweep_df.iloc[:-1], lookback=3)

    # 1. Detect the Equal Lows (EQL)
    pools = detect_liquidity_pools(swings, symbol="XAU/USD", tf="5min", pip_tolerance=40.0)

    # Isolate the EQL pool specifically to guarantee stable test assertions
    eql_pools = [p for p in pools if p.pool_type == "EQL"]
    assert len(eql_pools) > 0

    eql_pool = eql_pools[0]
    assert eql_pool.is_swept is False

    # 2. Process the final candle which contains the massive downside wick
    swept_pool = detect_liquidity_sweep(liquidity_sweep_df, eql_pools)

    assert swept_pool is not None
    assert swept_pool.pool_type == "EQL"
    assert swept_pool.is_swept is True


def test_generate_trade_signal():
    """Test accurate RRR mapping for Take Profit calculation."""
    signal = generate_trade_signal(
        symbol="XAU/USD",
        tf="5min",
        direction="buy",
        entry_price=2500.0,
        stop_loss=2490.0,
        factors=["Bullish CHoCH", "OTE Tap"],
        timestamp=pd.Timestamp.now(tz="UTC"),
    )

    # Stop loss difference is $10.00
    # TP1 should be 1:1.5 = +$15.00
    # TP2 should be 1:3.0 = +$30.00
    assert signal.take_profit_1 == 2515.0
    assert signal.take_profit_2 == 2530.0
    assert signal.risk_reward == 3.0
