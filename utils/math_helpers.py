"""
Pure mathematical utility functions for calculating Pips, RRR, and Fibonacci levels.
Optimized for XAU/USD (Gold).
"""

from typing import Dict


def get_pip_multiplier(symbol: str) -> float:
    """Returns the multiplier to convert raw price differences to pips."""
    if "XAU" in symbol or "GOLD" in symbol:
        return 10.0  # $1.00 move = 10 pips
    if "JPY" in symbol:
        return 100.0
    return 10000.0  # Standard forex (e.g., EUR/USD)


def price_to_pips(price_diff: float, symbol: str) -> float:
    """Converts a raw price difference into a pip value."""
    return abs(price_diff) * get_pip_multiplier(symbol)


def pips_to_price(pips: float, symbol: str) -> float:
    """Converts a pip value into a raw price difference."""
    return pips / get_pip_multiplier(symbol)


def calculate_rrr(entry: float, sl: float, tp: float) -> float:
    """Calculates Risk-to-Reward Ratio (RRR)."""
    risk = abs(entry - sl)
    if risk == 0:
        return 0.0
    reward = abs(tp - entry)
    return round(reward / risk, 2)


def fibonacci_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
    """
    Calculates standard SMC Fibonacci levels between two price points.
    Returns absolute price levels for each key ratio.
    """
    diff = swing_high - swing_low
    return {
        "0.0": swing_low,
        "0.236": swing_low + (diff * 0.236),
        "0.382": swing_low + (diff * 0.382),
        "0.5": swing_low + (diff * 0.5),
        "0.618": swing_low + (diff * 0.618),
        "0.705": swing_low + (diff * 0.705),
        "0.786": swing_low + (diff * 0.786),
        "1.0": swing_high,
    }


def is_within_range(price: float, zone_low: float, zone_high: float) -> bool:
    """Checks if a price falls within a specific high/low zone (inclusive)."""
    return min(zone_low, zone_high) <= price <= max(zone_low, zone_high)
