import pandas as pd
from typing import Dict


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates the Average True Range (ATR) to measure dynamic market volatility."""
    if len(df) < period + 1:
        return 2.0  # Safe fallback for XAUUSD if insufficient history

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def calculate_rrr(entry: float, sl: float, tp: float) -> float:
    """Calculates Risk-to-Reward Ratio (RRR)."""
    risk = abs(entry - sl)
    if risk == 0:
        return 0.0
    reward = abs(tp - entry)
    return round(reward / risk, 2)


def fibonacci_levels(swing_low: float, swing_high: float, direction: str = "bullish") -> Dict[str, float]:
    """Standard ICT Optimal Trade Entry (OTE) retracement coordinates."""
    diff = swing_high - swing_low

    if direction == "bullish":
        # Bullish retracements pull back DOWN from the swing high
        return {
            "0.0": swing_high,
            "0.618": swing_high - (diff * 0.618),
            "0.705": swing_high - (diff * 0.705),
            "0.786": swing_high - (diff * 0.786),
            "1.0": swing_low,
        }
    else:
        # Bearish retracements pull back UP from the swing low
        return {
            "0.0": swing_low,
            "0.618": swing_low + (diff * 0.618),
            "0.705": swing_low + (diff * 0.705),
            "0.786": swing_low + (diff * 0.786),
            "1.0": swing_high,
        }


def is_within_range(price: float, bound_a: float, bound_b: float) -> bool:
    """Checks if a given price falls within a specific high/low zone."""
    return min(bound_a, bound_b) <= price <= max(bound_a, bound_b)


def price_to_pips(price_diff: float, symbol: str) -> float:
    """Normalizes raw price differentials into standard pips based on asset class."""
    if symbol in ["XAUUSD", "XAU/USD"]:
        # Standard MT5/Prop Firm quote formatting where 0.01 = 1 pip
        return abs(price_diff) * 100
    if "JPY" in symbol:
        return abs(price_diff) * 100
    return abs(price_diff) * 10000


def pips_to_price(pips: float, symbol: str) -> float:
    """Converts pip values back to raw price differentials."""
    if symbol in ["XAUUSD", "XAU/USD"]:
        return pips / 100.0
    if "JPY" in symbol:
        return pips / 100.0
    return pips / 10000.0
