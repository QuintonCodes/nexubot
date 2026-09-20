"""
Market Structure Detection Engine.
Calculates Swings, BOS, CHoCH, MSS, and CISD from OHLCV DataFrames.
Strictly relies on close prices for structural breaks to filter liquidity sweeps.
"""

import pandas as pd
from typing import List, Literal, Optional

from strategies.models import SwingPoint, StructureEvent


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    """
    Identifies swing highs and lows using a non-repainting lookback window.
    Applies strict left/right differentials to prevent double-counting adjacent identical pivots.
    """
    swings = []
    last_high_price = None
    last_low_price = None

    # We stop evaluating at `len(df) - lookback` to prevent repainting.
    # The right side of the formation must be fully closed.
    for i in range(lookback, len(df) - lookback):
        left_highs = df["high"].iloc[i - lookback : i]
        right_highs = df["high"].iloc[i + 1 : i + lookback + 1]
        current_high = df["high"].iloc[i]

        left_lows = df["low"].iloc[i - lookback : i]
        right_lows = df["low"].iloc[i + 1 : i + lookback + 1]
        current_low = df["low"].iloc[i]

        # Swing High: Strictly higher than left window, >= right window
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

        # Swing Low: Strictly lower than left window, <= right window
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


def classify_structure(swings: List[SwingPoint]) -> Literal["bullish", "bearish", "ranging"]:
    """
    Determines the current market bias based on the sequence of the last few swings.
    """
    if len(swings) < 4:
        return "ranging"

    recent_swings = swings[-4:]
    classes = [s.classification for s in recent_swings]

    if "HH" in classes and "HL" in classes and "LL" not in classes:
        return "bullish"
    elif "LL" in classes and "LH" in classes and "HH" not in classes:
        return "bearish"

    return "ranging"


def detect_bos(df: pd.DataFrame, swings: List[SwingPoint], symbol: str, tf: str) -> Optional[StructureEvent]:
    """
    Detects a Break of Structure (BOS) occurring on the most recently closed candle.
    A BOS is confirmed ONLY if the candle closes beyond the last major swing point.
    """
    if not swings or len(df) < 2:
        return None

    last_candle = df.iloc[-1]

    # Get the most recent swing high and low
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

    # CHoCH is just a BOS that goes against the established bias
    if current_structure == "bullish" and bos_event.direction == "bearish":
        bos_event.event_type = "CHoCH"
        return bos_event

    if current_structure == "bearish" and bos_event.direction == "bullish":
        bos_event.event_type = "CHoCH"
        return bos_event

    return None
