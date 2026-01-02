import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List


class PatternRecognizer:
    """
    High-Fidelity Pattern Recognition Engine.
    Filters out weak patterns to ensure high win-rate.
    """

    def analyze_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scans for chart patterns: Double Top/Bottom, Head & Shoulders, Flags.
        """
        patterns = []
        if len(df) < 60:
            return patterns

        # 1. Swing Detection (Pivots)
        window = 2
        df["is_pivot_high"] = (
            (df["high"] > df["high"].shift(1))
            & (df["high"] > df["high"].shift(2))
            & (df["high"] > df["high"].shift(-1))
            & (df["high"] > df["high"].shift(-2))
        )
        df["is_pivot_low"] = (
            (df["low"] < df["low"].shift(1))
            & (df["low"] < df["low"].shift(2))
            & (df["low"] < df["low"].shift(-1))
            & (df["low"] < df["low"].shift(-2))
        )
        df["max_id"] = df.iloc[argrelextrema(df["close"].values, np.greater_equal, order=5)[0]]["close"]

        highs = df[df["is_pivot_high"]].copy()
        lows = df[df["is_pivot_low"]].copy()
        curr_price = df.iloc[-1]["close"]

        # --- DOUBLE TOP (Bearish) ---
        if len(highs) >= 2:
            h1 = highs.iloc[-2]
            h2 = highs.iloc[-1]

            # Check recency: The second peak must be recent (within last 10 candles)
            if (df.index.get_loc(df.index[-1]) - df.index.get_loc(h2.name)) < 15:
                # Tolerance 0.15% (Slightly looser for faster detection)
                if abs(h1["high"] - h2["high"]) / h1["high"] < 0.0015:
                    mask = (df.index > h1.name) & (df.index < h2.name)
                    if mask.any():
                        neckline = df.loc[mask, "low"].min()
                        # Aggressive Entry: Signal slightly BEFORE neckline break if volume matches
                        dist_to_break = (curr_price - neckline) / curr_price

                        if curr_price < neckline or (
                            dist_to_break < 0.001 and df.iloc[-1]["volume"] > df.iloc[-1]["vol_sma"]
                        ):
                            patterns.append(
                                {
                                    "pattern": "Double Top",
                                    "signal": "SELL",
                                    "direction": "SHORT",
                                    "confidence": 88.0,
                                    "order_type": "STOP",
                                    "price": neckline,
                                }
                            )

        if len(lows) >= 2:
            l1 = lows.iloc[-2]
            l2 = lows.iloc[-1]

            if (df.index.get_loc(df.index[-1]) - df.index.get_loc(l2.name)) < 15:
                if abs(l1["low"] - l2["low"]) / l1["low"] < 0.0015:
                    mask = (df.index > l1.name) & (df.index < l2.name)
                    if mask.any():
                        neckline = df.loc[mask, "high"].max()
                        dist_to_break = (neckline - curr_price) / neckline

                        if curr_price > neckline or (
                            dist_to_break < 0.001 and df.iloc[-1]["volume"] > df.iloc[-1]["vol_sma"]
                        ):
                            patterns.append(
                                {
                                    "pattern": "Double Bottom",
                                    "signal": "BUY",
                                    "direction": "LONG",
                                    "confidence": 88.0,
                                    "order_type": "STOP",
                                    "price": neckline,
                                }
                            )

        # --- HEAD AND SHOULDERS (Bearish) ---
        # Left Shoulder, Head (Higher), Right Shoulder (Lower than head, ~Left)
        if len(highs) >= 3:
            l_sh = highs.iloc[-3]
            head = highs.iloc[-2]
            r_sh = highs.iloc[-1]

            # Head must be higher than both shoulders
            if head["high"] > l_sh["high"] and head["high"] > r_sh["high"]:
                # Shoulders roughly equal (2% tolerance)
                if abs(l_sh["high"] - r_sh["high"]) / l_sh["high"] < 0.02:
                    # Neckline break check
                    neckline = min(
                        df.loc[l_sh.name : head.name, "low"].min(), df.loc[head.name : r_sh.name, "low"].min()
                    )
                    if curr_price < neckline:
                        patterns.append(
                            {"pattern": "Head & Shoulders", "signal": "SELL", "direction": "SHORT", "confidence": 90.0}
                        )

        # --- BULL/BEAR FLAGS (Continuation) ---
        flags = self.find_flags(df)
        if flags:
            patterns.extend(flags)

        return patterns

    def check_market_structure(self, df: pd.DataFrame) -> str:
        """
        Returns 'BULL', 'BEAR', or 'UNCLEAR' based on recent swing points.
        Prevents buying into a Lower Low structure.
        """
        # Get last 20 candles
        recent = df.iloc[-20:].copy()

        # Find simple local max/min
        recent["is_high"] = recent["high"] == recent["high"].rolling(5, center=True).max()
        recent["is_low"] = recent["low"] == recent["low"].rolling(5, center=True).min()

        last_highs = recent[recent["is_high"]]["high"].values
        last_lows = recent[recent["is_low"]]["low"].values

        if len(last_highs) < 2 or len(last_lows) < 2:
            return "UNCLEAR"

        # Bullish Structure: Higher Highs AND Higher Lows
        if last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2]:
            return "BULL"

        # Bearish Structure: Lower Lows AND Lower Highs
        if last_lows[-1] < last_lows[-2] and last_highs[-1] < last_highs[-2]:
            return "BEAR"

        return "UNCLEAR"

    def find_flags(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identifies Flags: Sharp Pole -> Low Vol Consolidation -> Breakout
        """
        patterns = []
        curr = df.iloc[-1]
        lookback = 20
        if len(df) < lookback:
            return patterns

        recent = df.iloc[-lookback:]
        pole_start = df.iloc[-lookback - 5]
        atr = curr["atr"]
        if atr == 0:
            return patterns  # Safety check

        pole_end = recent.iloc[0]

        # BULL FLAG
        # 1. Pole: Strong move UP
        move_up = pole_end["close"] - pole_start["close"]
        if move_up > (3 * atr):
            # 2. Flag Body: Consolidation (Volume drops)
            flag_body = recent.iloc[:-1]
            vol_drop = flag_body["volume"].mean() < flag_body["vol_sma"].mean()

            # 3. Structure: Price stayed in upper 50% of the pole
            held_gains = flag_body["low"].min() > (pole_start["low"] + (0.5 * move_up))

            if vol_drop and held_gains:
                # 4. Breakout: Current candle breaks flag high + Volume
                flag_high = flag_body["high"].max()
                if curr["close"] > flag_high and curr["volume"] > curr["vol_sma"]:
                    patterns.append({"pattern": "Bull Flag", "signal": "BUY", "direction": "LONG", "confidence": 85.0})

        # BEAR FLAG
        # 1. Pole: Strong move DOWN
        move_down = pole_start["close"] - pole_end["close"]
        if move_down > (3 * atr):
            flag_body = recent.iloc[:-1]
            vol_drop = flag_body["volume"].mean() < flag_body["vol_sma"].mean()

            # Structure: Price stayed in lower 50% of the pole
            held_lows = flag_body["high"].max() < (pole_start["high"] - (0.5 * move_down))

            if vol_drop and held_lows:
                flag_low = flag_body["low"].min()
                if curr["close"] < flag_low and curr["volume"] > curr["vol_sma"]:
                    patterns.append(
                        {"pattern": "Bear Flag", "signal": "SELL", "direction": "SHORT", "confidence": 85.0}
                    )

        return patterns


class FakeBreakoutDetector:
    """
    Risk Management Layer.
    Analyzes 'Traps' to invalidate weak signals.
    """

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Scans for manipulation patterns
        """
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        risk_score = 0
        reasons = []

        # 1. Low Volume Breakout (Fakeout)
        # If we are making a new high but volume is dropping
        if curr["high"] > prev["high"] and curr["volume"] < (curr["vol_sma"] * 0.8):
            risk_score += 40
            reasons.append("Low Vol Breakout")

        # 2. RSI Divergence (Exhaustion)
        # Price High, RSI Lower
        if curr["high"] > prev["high"] and curr["rsi"] < df.iloc[-5]["rsi"]:
            risk_score += 30
            reasons.append("RSI Divergence")

        # 3. Climax Wick (Rejection)
        body = abs(curr["close"] - curr["open"])
        wick_len = curr["high"] - max(curr["close"], curr["open"])
        if wick_len > (body * 2):
            risk_score += 30
            reasons.append("Wick Rejection")

        return {"is_fake": risk_score >= 50, "risk_score": risk_score, "reasons": reasons}
