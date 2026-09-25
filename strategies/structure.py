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


def detect_bos(
    df: pd.DataFrame, swings: List[SwingPoint], current_bias: str, symbol: str, tf: str
) -> Optional[StructureEvent]:
    """
    Detects a Break of Structure (BOS) occurring on the most recently closed candle.
    A BOS is confirmed ONLY if the candle closes beyond the last major swing point.
    """
    if not swings or len(df) < 2 or current_bias == "ranging":
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    if not highs or not lows:
        return None

    # Bullish BOS: Close above last swing high
    if current_bias == "bullish":
        last_high = highs[-1]
        if last_candle["close"] > last_high.price and prev_candle["close"] <= last_high.price:
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
    elif current_bias == "bearish":
        last_low = lows[-1]
        if last_candle["close"] < last_low.price and prev_candle["close"] >= last_low.price:
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
    df: pd.DataFrame, swings: List[SwingPoint], current_bias: str, symbol: str, tf: str
) -> Optional[StructureEvent]:
    """
    Detects a Change of Character (CHoCH) by monitoring for a break of the opposing pivot point.
    """
    if not swings or len(df) < 2 or current_bias == "ranging":
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    if not highs or not lows:
        return None

    if current_bias == "bullish":
        last_low = lows[-1]  # The last Higher Low (HL)
        if last_candle["close"] < last_low.price and prev_candle["close"] >= last_low.price:
            return StructureEvent(
                event_type="CHoCH",
                direction="bearish",
                price_level=last_low.price,
                timestamp=last_candle["timestamp"],
                symbol=symbol,
                timeframe=tf,
                confirmed=True,
            )

    elif current_bias == "bearish":
        last_high = highs[-1]  # The last Lower High (LH)
        if last_candle["close"] > last_high.price and prev_candle["close"] <= last_high.price:
            return StructureEvent(
                event_type="CHoCH",
                direction="bullish",
                price_level=last_high.price,
                timestamp=last_candle["timestamp"],
                symbol=symbol,
                timeframe=tf,
                confirmed=True,
            )

    return None


def detect_mss(
    df: pd.DataFrame, swings: List[SwingPoint], current_bias: str, symbol: str, tf: str, displacement_pips: float = 20.0
) -> Optional[StructureEvent]:
    """Detects Market Structure Shift (MSS): CHoCH accompanied by strong momentum."""
    choch = detect_choch(df, swings, current_bias, symbol, tf)
    if not choch:
        return None

    last_candle = df.iloc[-1]
    body_size = price_to_pips(abs(last_candle["close"] - last_candle["open"]), symbol)

    if body_size >= displacement_pips:
        choch.event_type = "MSS"
        return choch

    return None


def detect_cisd(df: pd.DataFrame, ob: dict, symbol: str, tf: str) -> Optional[StructureEvent]:
    """
    Detects Change in State of Delivery (CISD) via displacement candle close.
    Validates if delivery shifts by closing through the opening price of the origin candle.
    """
    if len(df) < 2:
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    # Find the origin candle that formed the OB
    origin_matches = df[df["timestamp"] == ob["origin_timestamp"]]
    if origin_matches.empty:
        return None

    ob_open = origin_matches.iloc[0]["open"]

    # If bullish OB (down candle), delivery shifts bearish if price closes BELOW the open of that down candle
    if ob["direction"] == "bullish" and last_candle["close"] < ob_open and prev_candle["close"] >= ob_open:
        return StructureEvent(
            event_type="CISD",
            direction="bearish",
            price_level=ob_open,
            timestamp=last_candle["timestamp"],
            symbol=symbol,
            timeframe=tf,
            confirmed=True,
        )

    # If bearish OB (up candle), delivery shifts bullish if price closes ABOVE the open of that up candle
    elif ob["direction"] == "bearish" and last_candle["close"] > ob_open and prev_candle["close"] <= ob_open:
        return StructureEvent(
            event_type="CISD",
            direction="bullish",
            price_level=ob_open,
            timestamp=last_candle["timestamp"],
            symbol=symbol,
            timeframe=tf,
            confirmed=True,
        )

    return None
