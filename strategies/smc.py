"""
SMC Entry Model Detection Engine.
Detects Order Blocks, Optimal Trade Entries (OTE), and Liquidity Pools/Sweeps.
"""

import pandas as pd
import uuid
from datetime import datetime
from typing import List, Optional, Literal

from strategies.models import SwingPoint, OrderBlock, OTEZone, LiquidityPool, TradeSignal
from utils.math_helpers import fibonacci_levels, price_to_pips


def find_order_blocks(
    df: pd.DataFrame, swings: List[SwingPoint], direction: Literal["bullish", "bearish"], symbol: str, tf: str
) -> List[OrderBlock]:
    """
    Identifies Order Blocks created before a major displacement.
    Bullish OB: The last down candle before a strong up move.
    Bearish OB: The last up candle before a strong down move.
    """
    obs = []

    # We need at least a few candles to identify an OB
    if len(df) < 5 or len(swings) < 2:
        return obs

    # For a simplified programmatic OB: we look at the origin of the last structural swing
    last_swing = swings[-1]

    # Search backwards from the swing to find the OB candle
    search_idx = last_swing.candle_index

    if direction == "bullish" and last_swing.type == "low":
        # Find the last bearish candle (close < open) near the swing low
        for i in range(search_idx, max(0, search_idx - 5), -1):
            if df["close"].iloc[i] < df["open"].iloc[i]:
                ob_high = df["high"].iloc[i]
                ob_low = df["low"].iloc[i]
                obs.append(
                    OrderBlock(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        timeframe=tf,
                        direction=direction,
                        ob_high=ob_high,
                        ob_low=ob_low,
                        ob_50=(ob_high + ob_low) / 2.0,
                        origin_timestamp=df["timestamp"].iloc[i],
                        is_mitigated=False,
                        mitigation_timestamp=None,
                        strength_score=0.8,
                    )
                )
                break

    elif direction == "bearish" and last_swing.type == "high":
        # Find the last bullish candle (close > open) near the swing high
        for i in range(search_idx, max(0, search_idx - 5), -1):
            if df["close"].iloc[i] > df["open"].iloc[i]:
                ob_high = df["high"].iloc[i]
                ob_low = df["low"].iloc[i]
                obs.append(
                    OrderBlock(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        timeframe=tf,
                        direction=direction,
                        ob_high=ob_high,
                        ob_low=ob_low,
                        ob_50=(ob_high + ob_low) / 2.0,
                        origin_timestamp=df["timestamp"].iloc[i],
                        is_mitigated=False,
                        mitigation_timestamp=None,
                        strength_score=0.8,
                    )
                )
                break

    return obs


def is_ob_mitigated(df: pd.DataFrame, ob: OrderBlock) -> bool:
    """
    An Order Block is mitigated when price trades and CLOSES past its 50% median line.
    """
    # Filter candles that occurred AFTER the OB was formed
    future_df = df[df["timestamp"] > ob.origin_timestamp]
    if future_df.empty:
        return False

    if ob.direction == "bullish":
        # Bullish OB mitigated if a candle closes below the 50% line
        mitigating_candles = future_df[future_df["close"] < ob.ob_50]
        return not mitigating_candles.empty
    else:
        # Bearish OB mitigated if a candle closes above the 50% line
        mitigating_candles = future_df[future_df["close"] > ob.ob_50]
        return not mitigating_candles.empty


def calculate_ote_zone(
    swing_low: float, swing_high: float, direction: Literal["bullish", "bearish"], symbol: str, tf: str
) -> OTEZone:
    """Calculates the ICT Optimal Trade Entry (61.8% to 78.6%) for a given displacement leg."""
    fibs = fibonacci_levels(swing_low, swing_high)

    if direction == "bullish":
        return OTEZone(
            symbol=symbol,
            timeframe=tf,
            direction=direction,
            fib_0=swing_low,
            fib_1=swing_high,
            ote_entry=fibs["0.618"],
            ote_mid=fibs["0.705"],
            ote_top=fibs["0.786"],
        )
    else:
        # For bearish, retracement pulls up from the bottom
        # Fib 0 is high, Fib 1 is low in standard charting, but using absolute math:
        diff = swing_high - swing_low
        return OTEZone(
            symbol=symbol,
            timeframe=tf,
            direction=direction,
            fib_0=swing_high,
            fib_1=swing_low,
            ote_entry=swing_high - (diff * 0.618),
            ote_mid=swing_high - (diff * 0.705),
            ote_top=swing_high - (diff * 0.786),
        )


def detect_liquidity_pools(
    swings: List[SwingPoint], symbol: str, tf: str, pip_tolerance: float = 40.0
) -> List[LiquidityPool]:
    """Scans historical swings to find Equal Highs (EQH) and Equal Lows (EQL)."""
    pools = []
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    # Detect EQH
    if len(highs) >= 2:
        for i in range(len(highs) - 1):
            # Ensure swings are distinct structural points (separated by at least 3 candles)
            if highs[-1].candle_index - highs[i].candle_index > 3:
                if price_to_pips(abs(highs[i].price - highs[-1].price), symbol) <= pip_tolerance:
                    pools.append(
                        LiquidityPool(
                            symbol=symbol,
                            timeframe=tf,
                            pool_type="EQH",
                            price_level=max(highs[i].price, highs[-1].price),
                            price_tolerance=pip_tolerance,
                            touch_count=2,
                            is_swept=False,
                            sweep_timestamp=None,
                        )
                    )
                    break

    # Detect EQL
    if len(lows) >= 2:
        for i in range(len(lows) - 1):
            # Ensure swings are distinct structural points (separated by at least 3 candles)
            if lows[-1].candle_index - lows[i].candle_index > 3:
                if price_to_pips(abs(lows[i].price - lows[-1].price), symbol) <= pip_tolerance:
                    pools.append(
                        LiquidityPool(
                            symbol=symbol,
                            timeframe=tf,
                            pool_type="EQL",
                            price_level=min(lows[i].price, lows[-1].price),
                            price_tolerance=pip_tolerance,
                            touch_count=2,
                            is_swept=False,
                            sweep_timestamp=None,
                        )
                    )
                    break

    return pools


def detect_liquidity_sweep(df: pd.DataFrame, pools: List[LiquidityPool]) -> Optional[LiquidityPool]:
    """Checks if the most recent candle swept a liquidity pool and rejected."""
    if df.empty or not pools:
        return None

    last_candle = df.iloc[-1]

    for pool in pools:
        if pool.is_swept:
            continue

        if pool.pool_type == "EQH":
            # Wick above the EQH, but close below it
            if last_candle["high"] > pool.price_level and last_candle["close"] < pool.price_level:
                pool.is_swept = True
                pool.sweep_timestamp = last_candle["timestamp"]
                return pool

        elif pool.pool_type == "EQL":
            # Wick below the EQL, but close above it
            if last_candle["low"] < pool.price_level and last_candle["close"] > pool.price_level:
                pool.is_swept = True
                pool.sweep_timestamp = last_candle["timestamp"]
                return pool

    return None


def generate_trade_signal(
    symbol: str,
    tf: str,
    direction: Literal["buy", "sell"],
    entry_price: float,
    stop_loss: float,
    factors: List[str],
    timestamp: datetime,
) -> TradeSignal:
    """Assembles a valid TradeSignal, automatically calculating Take Profits based on RRR."""
    sl_pips_diff = abs(entry_price - stop_loss)

    if direction == "buy":
        tp1 = entry_price + (sl_pips_diff * 1.5)
        tp2 = entry_price + (sl_pips_diff * 3.0)
    else:
        tp1 = entry_price - (sl_pips_diff * 1.5)
        tp2 = entry_price - (sl_pips_diff * 3.0)

    return TradeSignal(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        direction=direction,
        entry_price=round(entry_price, 3),
        stop_loss=round(stop_loss, 3),
        take_profit_1=round(tp1, 3),
        take_profit_2=round(tp2, 3),
        risk_reward=3.0,
        signal_type="SMC_Confluence",
        confluence_score=85,
        confluence_factors=factors,
        timestamp=timestamp,
        timeframe=tf,
    )
