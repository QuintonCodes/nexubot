"""
SMC Entry Model Detection Engine.
Detects Order Blocks, Optimal Trade Entries (OTE), FVGs, Inducements, and Liquidity Pools/Sweeps.
"""

import pandas as pd
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from strategies.models import FVG, LiquidityPool, OrderBlock, OTEZone, SwingPoint, TradeSignal
from utils.math_helpers import calculate_atr, calculate_rrr, fibonacci_levels, price_to_pips


def detect_fair_value_gaps(df: pd.DataFrame, symbol: str, tf: str) -> List[FVG]:
    """Identifies institutional 3-candle imbalances/FVGs."""
    fvgs = []
    if len(df) < 3:
        return fvgs

    for i in range(1, len(df) - 1):
        prev_c = df.iloc[i - 1]
        next_c = df.iloc[i + 1]

        # Bullish FVG
        if next_c["low"] > prev_c["high"]:
            fvgs.append(
                FVG(
                    symbol=symbol,
                    timeframe=tf,
                    direction="bullish",
                    top=next_c["low"],
                    bottom=prev_c["high"],
                    timestamp=df.iloc[i]["timestamp"],
                    mitigated=False,
                )
            )
        # Bearish FVG
        elif next_c["high"] < prev_c["low"]:
            fvgs.append(
                FVG(
                    symbol=symbol,
                    timeframe=tf,
                    direction="bearish",
                    top=prev_c["low"],
                    bottom=next_c["high"],
                    timestamp=df.iloc[i]["timestamp"],
                    mitigated=False,
                )
            )
    return fvgs


def detect_inducement(swings: List[SwingPoint], direction: Literal["bullish", "bearish"]) -> Optional[SwingPoint]:
    """Finds the minor bait swing before the structural extreme."""
    if len(swings) < 3:
        return None

    if direction == "bullish":
        lows = [s for s in swings if s.type == "low"]
        if len(lows) >= 2:
            return lows[-2]
    else:
        highs = [s for s in swings if s.type == "high"]
        if len(highs) >= 2:
            return highs[-2]
    return None


def get_premium_discount_zone(swing_low: float, swing_high: float, current_price: float) -> str:
    """Calculates ICT Premium/Discount equilibrium arrays."""
    midpoint = (swing_high + swing_low) / 2.0
    if current_price > midpoint:
        return "Premium"
    elif current_price < midpoint:
        return "Discount"
    return "Equilibrium"


def find_order_blocks(
    df: pd.DataFrame, swings: List[SwingPoint], direction: Literal["bullish", "bearish"], symbol: str, tf: str
) -> List[OrderBlock]:
    """Identifies institutional Order Blocks factoring dynamic scoring and a 20-candle lookback."""
    obs = []

    if len(df) < 5 or len(swings) < 2:
        return obs

    # Scan the last 3 relevant swing extremes instead of just the absolute last one
    relevant_swings = [s for s in reversed(swings) if s.type == ("low" if direction == "bullish" else "high")][:3]

    for last_swing in relevant_swings:
        search_idx = last_swing.candle_index
        lookback_limit = max(0, search_idx - 20)
        atr_val = calculate_atr(df.iloc[: search_idx + 1]) if search_idx > 14 else 2.0

        if direction == "bullish":
            for i in range(search_idx, lookback_limit, -1):
                body = abs(df["close"].iloc[i] - df["open"].iloc[i])
                total_range = df["high"].iloc[i] - df["low"].iloc[i]

                # Filter weak indecision candles
                if body < (atr_val * 0.10) or df["close"].iloc[i] >= df["open"].iloc[i]:
                    continue

                # Displacement Validation
                displacement_valid = False
                for j in range(i + 1, min(i + 6, len(df))):
                    if abs(df["close"].iloc[j] - df["open"].iloc[j]) > (atr_val * 0.8):
                        displacement_valid = True
                        break
                if not displacement_valid:
                    continue

                # Check FVG overlap for boost
                fvg_boost = 0.0
                for j in range(i + 1, min(i + 4, len(df) - 1)):
                    if df["low"].iloc[j + 1] > df["high"].iloc[j - 1]:
                        fvg_boost = 0.2
                        break

                # Dynamic Score Calculation
                bwr = body / total_range if total_range > 0 else 0
                displacement_pips = abs(df["close"].iloc[search_idx] - df["close"].iloc[i])

                score = 0.4 + fvg_boost
                if bwr > 0.6:
                    score += 0.2  # High body-to-wick ratio
                if displacement_pips > (atr_val * 1.2):
                    score += 0.2  # Strong displacement impulse

                ob_high, ob_low = df["high"].iloc[i], df["low"].iloc[i]
                obs.append(
                    OrderBlock(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        timeframe=tf,
                        direction=direction,
                        ob_high=ob_high,
                        ob_low=ob_low,
                        ob_50=float(ob_high + ob_low) / 2.0,
                        origin_timestamp=df["timestamp"].iloc[i],
                        mitigated=False,
                        mitigation_timestamp=None,
                        strength_score=min(1.0, score),
                    )
                )
                break

        elif direction == "bearish":
            for i in range(search_idx, lookback_limit, -1):
                body = abs(df["close"].iloc[i] - df["open"].iloc[i])
                total_range = df["high"].iloc[i] - df["low"].iloc[i]

                if body < (atr_val * 0.10) or df["close"].iloc[i] <= df["open"].iloc[i]:
                    continue

                # Displacement Validation
                displacement_valid = False
                for j in range(i + 1, min(i + 6, len(df))):
                    if abs(df["close"].iloc[j] - df["open"].iloc[j]) > (atr_val * 0.8):
                        displacement_valid = True
                        break
                if not displacement_valid:
                    continue

                # Check FVG overlap for boost
                fvg_boost = 0.0
                for j in range(i + 1, min(i + 4, len(df) - 1)):
                    if df["high"].iloc[j + 1] < df["low"].iloc[j - 1]:
                        fvg_boost = 0.2
                        break

                bwr = body / total_range if total_range > 0 else 0
                displacement_pips = abs(df["close"].iloc[search_idx] - df["close"].iloc[i])

                score = 0.4 + fvg_boost
                if bwr > 0.6:
                    score += 0.2
                if displacement_pips > (atr_val * 1.2):
                    score += 0.2

                ob_high, ob_low = df["high"].iloc[i], df["low"].iloc[i]
                obs.append(
                    OrderBlock(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        timeframe=tf,
                        direction=direction,
                        ob_high=ob_high,
                        ob_low=ob_low,
                        ob_50=float(ob_high + ob_low) / 2.0,
                        origin_timestamp=df["timestamp"].iloc[i],
                        mitigated=False,
                        mitigation_timestamp=None,
                        strength_score=min(1.0, score),
                    )
                )
                break

        if obs:  # Stop scanning older swings if we found a valid structural OB
            break

    return obs


def is_ob_mitigated(df: pd.DataFrame, ob: Dict[str, Any]) -> bool:
    """
    An Order Block is mitigated when price trades and CLOSES past its 50% median line.
    """
    # Filter candles that occurred AFTER the OB was formed
    future_df = df[df["timestamp"] > ob["origin_timestamp"]]
    if future_df.empty:
        return False

    # Cast to float to handle DB Decimal precision returns
    ob_high = float(max(ob["ob_high"], ob["ob_low"]))
    ob_low = float(min(ob["ob_high"], ob["ob_low"]))
    ob_50 = (ob_high + ob_low) / 2.0

    if ob["direction"] == "bullish":
        mitigating_candles = future_df[future_df["close"] < ob_50]
        return not mitigating_candles.empty
    else:
        mitigating_candles = future_df[future_df["close"] > ob_50]
        return not mitigating_candles.empty


def calculate_ote_zone(
    swing_low: float, swing_high: float, direction: Literal["bullish", "bearish"], symbol: str, tf: str
) -> OTEZone:
    """Calculates the ICT Optimal Trade Entry (61.8% to 78.6%) for a given displacement leg."""
    fibs = fibonacci_levels(swing_low, swing_high, direction)

    return OTEZone(
        symbol=symbol,
        timeframe=tf,
        direction=direction,
        fib_0=fibs["0.0"],
        fib_1=fibs["1.0"],
        ote_entry=fibs["0.618"],
        ote_mid=fibs["0.705"],
        ote_top=fibs["0.786"],
    )


def detect_liquidity_pools(
    swings: List[SwingPoint], symbol: str, tf: str, pip_tolerance: float = 40.0
) -> List[LiquidityPool]:
    """Scans historical swings to find Equal Highs (EQH) and Equal Lows (EQL)."""
    pools = []
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    # Detect EQH across the last 4 major highs
    if len(highs) >= 2:
        recent_highs = highs[-4:] if len(highs) > 4 else highs
        for rh in reversed(recent_highs):
            for ph in reversed(highs):
                if rh.candle_index > ph.candle_index and (rh.candle_index - ph.candle_index > 3):
                    if price_to_pips(abs(rh.price - ph.price), symbol) <= pip_tolerance:
                        pools.append(
                            LiquidityPool(
                                symbol=symbol,
                                timeframe=tf,
                                pool_type="EQH",
                                price_level=max(rh.price, ph.price),
                                price_tolerance=pip_tolerance,
                                touch_count=2,
                                swept=False,
                                sweep_timestamp=None,
                            )
                        )
                        break

    # Detect EQL
    if len(lows) >= 2:
        recent_lows = lows[-4:] if len(lows) > 4 else lows
        for rl in reversed(recent_lows):
            for pl in reversed(lows):
                if rl.candle_index > pl.candle_index and (rl.candle_index - pl.candle_index > 3):
                    if price_to_pips(abs(rl.price - pl.price), symbol) <= pip_tolerance:
                        pools.append(
                            LiquidityPool(
                                symbol=symbol,
                                timeframe=tf,
                                pool_type="EQL",
                                price_level=min(rl.price, pl.price),
                                price_tolerance=pip_tolerance,
                                touch_count=2,
                                swept=False,
                                sweep_timestamp=None,
                            )
                        )
                        break

    return pools


def detect_liquidity_sweep(df: pd.DataFrame, pools: List[LiquidityPool]) -> Optional[LiquidityPool]:
    """Checks if any of the recent 3 candles swept a liquidity pool and rejected."""
    if len(df) < 3 or not pools:
        return None

    recent_candles = df.tail(3)

    for pool in pools:
        if pool.swept:
            continue

        for _, candle in recent_candles.iterrows():
            if pool.pool_type == "EQH":
                if candle["high"] > pool.price_level and candle["close"] < pool.price_level:
                    pool.swept = True
                    pool.sweep_timestamp = candle["timestamp"]
                    return pool

            elif pool.pool_type == "EQL":
                if candle["low"] < pool.price_level and candle["close"] > pool.price_level:
                    pool.swept = True
                    pool.sweep_timestamp = candle["timestamp"]
                    return pool

    return None


def generate_trade_signal(
    symbol: str,
    tf: str,
    direction: Literal["buy", "sell"],
    entry_price: float,
    stop_loss: float,
    factors: List[str],
    confluence_score: int,
    timestamp: datetime,
    entry_model: str,
    session: str,
    pd_zone: str,
) -> TradeSignal:
    """Assembles a valid TradeSignal, smartly calculating dynamic RR Take Profits."""
    sl_pips_diff = abs(entry_price - stop_loss)

    # Dynamic RR scales linearly with setup strength (Score 70 = 3.5R, Score 100 = 5.0R)
    dynamic_rr = max(3.0, confluence_score / 20.0)

    tp3_distance = sl_pips_diff * dynamic_rr
    tp1_distance = tp3_distance * 0.30  # TP1 captures 30% of the total target move
    tp2_distance = tp3_distance * 0.60  # TP2 captures 60% of the total target move

    if direction == "buy":
        tp3 = entry_price + tp3_distance
        tp2 = entry_price + tp2_distance
        tp1 = entry_price + tp1_distance
    else:
        tp3 = entry_price - tp3_distance
        tp2 = entry_price - tp2_distance
        tp1 = entry_price - tp1_distance

    actual_rrr = calculate_rrr(entry_price, stop_loss, tp3)

    return TradeSignal(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        direction=direction,
        entry_price=round(entry_price, 5),
        stop_loss=round(stop_loss, 5),
        take_profit_1=round(tp1, 5),
        take_profit_2=round(tp2, 5),
        take_profit_3=round(tp3, 5),
        risk_reward=actual_rrr,
        signal_type="SMC_Confluence",
        entry_model=entry_model,
        session=session,
        pd_zone=pd_zone,
        confluence_score=confluence_score,
        confluence_factors=factors,
        timestamp=timestamp,
        timeframe=tf,
    )
