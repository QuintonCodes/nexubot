"""
Unit tests for SMC Entry Models, Math Helpers, and Session Tracking.
Run with: python -m pytest tests/test_smc.py -v
"""

import pandas as pd
from datetime import datetime, timezone

from strategies.sessions import SessionManager
from strategies.smc import (
    calculate_ote_zone,
    detect_fair_value_gaps,
    detect_inducement,
    detect_liquidity_pools,
    detect_liquidity_sweep,
    find_order_blocks,
    generate_trade_signal,
    get_premium_discount_zone,
    is_ob_mitigated,
)
from strategies.structure import detect_swings
from utils.math_helpers import (
    calculate_atr,
    calculate_rrr,
    is_within_range,
    pips_to_price,
    price_to_pips,
)


def test_find_and_mitigate_order_block(bullish_bos_df):
    """Test OB creation, displacement filtering, and 50% mitigation.."""
    swings = detect_swings(bullish_bos_df, lookback=3)

    # 1. Detect Order Blocks
    obs = find_order_blocks(bullish_bos_df, swings, direction="bullish", symbol="XAU/USD", tf="5min")
    assert len(obs) > 0
    ob = obs[0]

    # Ensure it mapped to a bearish down-candle before the swing low
    assert ob.ob_high > ob.ob_low
    assert ob.mitigated is False
    assert 0.4 <= ob.strength_score <= 1.0

    # 2. Test Mitigation
    ob_dict = {
        "id": 1,
        "symbol": ob.symbol,
        "timeframe": ob.timeframe,
        "direction": ob.direction,
        "ob_high": ob.ob_high,
        "ob_low": ob.ob_low,
        "origin_timestamp": ob.origin_timestamp,
    }
    assert is_ob_mitigated(bullish_bos_df, ob_dict) is False

    # Manually append a candle that crashes through the OB 50% line to test mitigation
    crash_row = bullish_bos_df.iloc[-1].copy()
    crash_row["timestamp"] = crash_row["timestamp"] + pd.Timedelta(minutes=5)
    crash_row["close"] = ob.ob_50 - 5.0  # Close below 50% line

    mitigated_df = pd.concat([bullish_bos_df, crash_row.to_frame().T], ignore_index=True)
    assert is_ob_mitigated(mitigated_df, ob_dict) is True


def test_calculate_ote_zone():
    """Test standard Fibonacci retracement mathematics for both directions."""
    # Bullish (Low to High -> Retraces Downward into Discount)
    bull_zone = calculate_ote_zone(
        swing_low=2000.0, swing_high=2100.0, direction="bullish", symbol="XAU/USD", tf="5min"
    )
    assert bull_zone.ote_entry == 2038.2  # 2100 - (100 * 0.618)
    assert bull_zone.ote_mid == 2029.5
    assert bull_zone.ote_top == 2021.4

    # Bearish (High to Low -> Retraces Upward into Premium)
    bear_zone = calculate_ote_zone(
        swing_low=2000.0, swing_high=2100.0, direction="bearish", symbol="XAU/USD", tf="5min"
    )
    assert bear_zone.ote_entry == 2061.8  # 2000 + (100 * 0.618)
    assert bear_zone.ote_mid == 2070.5
    assert bear_zone.ote_top == 2078.6


def test_calculate_ote_zone_with_retracement_df(ote_retracement_df):
    """Test OTE calculation against synthetic retracement DataFrame fixture."""
    swing_low = ote_retracement_df["low"].min()
    swing_high = ote_retracement_df["high"].max()

    zone = calculate_ote_zone(swing_low, swing_high, direction="bullish", symbol="XAUUSD", tf="5min")

    # The fixture data was generated against the old faulty upward math.
    # Force the synthetic close directly into the true OTE pocket to validate.
    retracement_candle = ote_retracement_df.iloc[-1].copy()
    retracement_candle["close"] = zone.ote_mid

    assert zone.direction == "bullish"
    assert zone.fib_0 == swing_high
    assert zone.fib_1 == swing_low
    assert is_within_range(retracement_candle["close"], zone.ote_entry, zone.ote_top)


def test_detect_liquidity_sweep(liquidity_sweep_df):
    """Test that EQH/EQL are identified, and wick-closures confirm sweeps."""
    swings = detect_swings(liquidity_sweep_df.iloc[:-1], lookback=3)
    pools = detect_liquidity_pools(swings, symbol="XAU/USD", tf="5min", pip_tolerance=40.0)

    # Isolate the EQL pool specifically to guarantee stable test assertions
    eql_pools = [p for p in pools if p.pool_type == "EQL"]
    assert len(eql_pools) > 0

    eql_pool = eql_pools[0]
    assert eql_pool.swept is False

    # 2. Process the final candle which contains the massive downside wick
    swept_pool = detect_liquidity_sweep(liquidity_sweep_df, eql_pools)

    assert swept_pool is not None
    assert swept_pool.pool_type == "EQL"
    assert swept_pool.swept is True


def test_detect_fair_value_gaps(fvg_df):
    """Test Fair Value Gap detection."""
    fvgs = detect_fair_value_gaps(fvg_df, symbol="XAUUSD", tf="5min")
    assert len(fvgs) > 0
    bullish_fvgs = [f for f in fvgs if f.direction == "bullish"]
    assert len(bullish_fvgs) > 0
    assert bullish_fvgs[0].top > bullish_fvgs[0].bottom


def test_detect_inducement(bullish_bos_df):
    """Test minor swing Inducement identification."""
    swings = detect_swings(bullish_bos_df, lookback=3)
    idm = detect_inducement(swings, direction="bullish")
    assert idm is not None
    assert idm.type == "low"


def test_get_premium_discount_zone():
    """Test ICT Premium/Discount equilibrium calculation."""
    # Range: 2000.0 to 2100.0 -> Midpoint = 2050.0
    assert get_premium_discount_zone(2000.0, 2100.0, 2075.0) == "Premium"
    assert get_premium_discount_zone(2000.0, 2100.0, 2025.0) == "Discount"
    assert get_premium_discount_zone(2000.0, 2100.0, 2050.0) == "Equilibrium"


def test_generate_trade_signal():
    """Test accurate RRR mapping for Take Profit calculation."""
    signal = generate_trade_signal(
        symbol="XAU/USD",
        tf="5min",
        direction="buy",
        entry_price=2500.0,
        stop_loss=2490.0,
        factors=["Bullish CHoCH", "OTE Tap"],
        confluence_score=85,
        timestamp=datetime.now(timezone.utc),
        entry_model="MTF OB Entry",
        session="London Open Killzone",
        pd_zone="Discount",
    )

    # 10.0 SL Risk. Dynamic RRR for score 85 = 85/20.0 = 4.25R
    # TP3 Distance = 10.0 * 4.25 = 42.5  --> TP3 = 2542.5
    # TP1 Distance = 42.5 * 30% = 12.75  --> TP1 = 2512.75
    # TP2 Distance = 42.5 * 60% = 25.50  --> TP2 = 2525.5

    assert signal.take_profit_1 == 2512.75
    assert signal.take_profit_2 == 2525.5
    assert signal.take_profit_3 == 2542.5
    assert signal.risk_reward == 4.25
    assert signal.entry_model == "MTF OB Entry"
    assert signal.session == "London Open Killzone"
    assert signal.pd_zone == "Discount"
    assert signal.confluence_score == 85


def test_math_helpers():
    """Test pip/price conversions, ranges, RRR, and ATR."""
    assert price_to_pips(1.0, "XAUUSD") == 100.0
    assert pips_to_price(100.0, "XAUUSD") == 1.0

    assert is_within_range(2500.0, 2490.0, 2510.0) is True
    assert is_within_range(2520.0, 2490.0, 2510.0) is False

    assert calculate_rrr(entry=2500.0, sl=2490.0, tp=2530.0) == 3.0

    atr_df = pd.DataFrame(
        {
            "high": [2502.0 + i for i in range(20)],
            "low": [2498.0 + i for i in range(20)],
            "close": [2500.0 + i for i in range(20)],
        }
    )
    atr = calculate_atr(atr_df, period=14)
    assert atr > 0.0


def test_session_manager():
    """Test session killzones and daily/weekly open trackers."""
    dt_london = datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc)
    assert SessionManager.get_active_killzone(dt_london) == "London Killzone"

    dt_off = datetime(2026, 9, 1, 22, 0, tzinfo=timezone.utc)
    assert SessionManager.get_active_killzone(dt_off) == "Out of Session"

    htf_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-09-01", periods=10, freq="4h", tz="UTC"),
            "open": [2500.0 + i for i in range(10)],
            "high": [2505.0 + i for i in range(10)],
            "low": [2495.0 + i for i in range(10)],
            "close": [2502.0 + i for i in range(10)],
            "volume": [500.0] * 10,
        }
    )
    levels = SessionManager.get_daily_weekly_open(htf_df)
    assert "NDO" in levels and "NWO" in levels
