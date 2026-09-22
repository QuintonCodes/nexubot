"""
Market Structure Detection Engine.
Calculates Swings, BOS, CHoCH, MSS, and CISD from OHLCV DataFrames.
Strictly relies on close prices for structural breaks to filter liquidity sweeps.
"""

import pandas as pd
from typing import List, Literal, Optional

from strategies.models import StructureEvent, SwingPoint
from utils.math_helpers import price_to_pips


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    """
    Identifies swing highs and lows using a non-repainting lookback window.
    Applies strict left/right differentials to prevent double-counting adjacent identical pivots.
    """
    swings = []
    last_high_price = None
    last_low_price = None

    for i in range(lookback, len(df) - lookback):
        left_highs = df["high"].iloc[i - lookback : i]
        right_highs = df["high"].iloc[i + 1 : i + lookback + 1]
        current_high = df["high"].iloc[i]

        left_lows = df["low"].iloc[i - lookback : i]
        right_lows = df["low"].iloc[i + 1 : i + lookback + 1]
        current_low = df["low"].iloc[i]

        if current_high > left_highs.max() and current_high >= right_highs.max():
            classification = "UNCONFIRMED"
            if last_high_price is not None:
                classification = "HH" if current_high > last_high_price else "LH"

            swings.append(
                SwingPoint(
                    timestamp=df["timestamp"].iloc[i],
                    price=current_high,
                    type="high",
                    classification=classification,
                    candle_index=i,
                )
            )
            last_high_price = current_high

        elif current_low < left_lows.min() and current_low <= right_lows.min():
            classification = "UNCONFIRMED"
            if last_low_price is not None:
                classification = "HL" if current_low > last_low_price else "LL"

            swings.append(
                SwingPoint(
                    timestamp=df["timestamp"].iloc[i],
                    price=current_low,
                    type="low",
                    classification=classification,
                    candle_index=i,
                )
            )
            last_low_price = current_low

    return swings


def classify_structure(df: pd.DataFrame, swings: List[SwingPoint]) -> Literal["bullish", "bearish", "ranging"]:
    """Replays historical price action to determine the true current bias."""
    if len(swings) < 2 or len(df) < 20:
        return "ranging"

    bias = "ranging"
    current_high = None
    current_low = None
    swing_idx = 0

    for i in range(swings[0].candle_index + 1, len(df)):
        # Incrementally update active structures anchoring the bias
        while swing_idx < len(swings) and swings[swing_idx].candle_index < i:
            if swings[swing_idx].type == "high":
                current_high = swings[swing_idx].price
            else:
                current_low = swings[swing_idx].price
            swing_idx += 1

        candle_close = df.iloc[i]["close"]
        if current_high is not None and current_low is not None:
            if candle_close > current_high:
                bias = "bullish"
            elif candle_close < current_low:
                bias = "bearish"

    return bias


def detect_bos(df: pd.DataFrame, swings: List[SwingPoint], symbol: str, tf: str) -> Optional[StructureEvent]:
    """
    Detects a Break of Structure (BOS) occurring on the most recently closed candle.
    A BOS is confirmed ONLY if the candle closes beyond the last major swing point.
    """
    if not swings or len(df) < 2:
        return None

    last_candle = df.iloc[-1]
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    if not highs or not lows:
        return None

    last_high = highs[-1]
    last_low = lows[-1]

    # Bullish BOS: Close above last swing high
    if last_candle["close"] > last_high.price and df.iloc[-2]["close"] <= last_high.price:
        return StructureEvent(
            event_type="BOS",
            direction="bullish",
            price_level=last_high.price,
            timestamp=last_candle["timestamp"],
            symbol=symbol,
            timeframe=tf,
            confirmed=True,
        )

    # Bearish BOS: Close below last swing low
    if last_candle["close"] < last_low.price and df.iloc[-2]["close"] >= last_low.price:
        return StructureEvent(
            event_type="BOS",
            direction="bearish",
            price_level=last_low.price,
            timestamp=last_candle["timestamp"],
            symbol=symbol,
            timeframe=tf,
            confirmed=True,
        )

    return None


def detect_choch(
    df: pd.DataFrame, swings: List[SwingPoint], current_structure: str, symbol: str, tf: str
) -> Optional[StructureEvent]:
    """
    Detects a Change of Character (CHoCH).
    This is an opposing structure break (e.g., a bearish BOS while the structure was bullish).
    """
    bos_event = detect_bos(df, swings, symbol, tf)
    if not bos_event:
        return None

    if current_structure == "bullish" and bos_event.direction == "bearish":
        bos_event.event_type = "CHoCH"
        return bos_event

    if current_structure == "bearish" and bos_event.direction == "bullish":
        bos_event.event_type = "CHoCH"
        return bos_event

    return None


def detect_mss(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    current_structure: str,
    symbol: str,
    tf: str,
    displacement_pips: float = 30.0,
) -> Optional[StructureEvent]:
    """Detects Market Structure Shift (MSS): CHoCH accompanied by strong momentum."""
    choch = detect_choch(df, swings, current_structure, symbol, tf)
    if not choch:
        return None

    last_candle = df.iloc[-1]
    body_size = price_to_pips(abs(last_candle["close"] - last_candle["open"]), symbol)

    if body_size >= displacement_pips:
        choch.event_type = "MSS"
        return choch

    return None


def detect_cisd(df: pd.DataFrame, ob: dict, symbol: str, tf: str) -> Optional[StructureEvent]:
    """Detects Change in State of Delivery (CISD) via 50% re-encroachment."""
    if len(df) < 2:
        return None

    last_candle = df.iloc[-1]
    ob_50 = (ob["ob_high"] + ob["ob_low"]) / 2.0

    # Verify if delivery shifted by rejecting off the OB's equilibrium
    if ob["direction"] == "bullish" and last_candle["low"] <= ob_50 <= last_candle["high"]:
        return StructureEvent(
            event_type="CISD",
            direction="bullish",
            price_level=ob_50,
            timestamp=last_candle["timestamp"],
            symbol=symbol,
            timeframe=tf,
            confirmed=True,
        )

    elif ob["direction"] == "bearish" and last_candle["low"] <= ob_50 <= last_candle["high"]:
        return StructureEvent(
            event_type="CISD",
            direction="bearish",
            price_level=ob_50,
            timestamp=last_candle["timestamp"],
            symbol=symbol,
            timeframe=tf,
            confirmed=True,
        )

    return None
