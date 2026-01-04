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

        # Work on a short local copy to avoid mutating caller's df
        local = df.copy()

        # --- Robust Swing Detection (Pivots) ---
        order = 3  # slightly larger order reduces noise
        try:
            highs_idx = argrelextrema(local["high"].values, np.greater_equal, order=order)[0]
            lows_idx = argrelextrema(local["low"].values, np.less_equal, order=order)[0]
        except Exception:
            return patterns

        highs = local.iloc[highs_idx].dropna()
        lows = local.iloc[lows_idx].dropna()
        curr_price = local.iloc[-1]["close"]
        last_pos = len(local) - 1

        # helper: safe recency check (positions, not index labels)
        def is_recent(row_pos, max_age=15):
            return (last_pos - row_pos) <= max_age

        # --- DOUBLE TOP (Bearish) ---
        if len(highs) >= 2:
            h1 = highs.iloc[-2]
            h2 = highs.iloc[-1]
            pos_h2 = highs.index.get_loc(h2.name)
            # ensure recency and similar peak heights
            if is_recent(pos_h2, max_age=15) and abs(h1["high"] - h2["high"]) / h1["high"] < 0.002:
                # Tolerance 0.15% (Slightly looser for faster detection)
                mask = (local.index > h1.name) & (local.index < h2.name)
                if mask.any():
                    neckline = local.loc[mask, "low"].min()
                    # require reasonable dip between peaks and volume confirmation
                    dip_size = max(h1["high"], h2["high"]) - neckline
                    if dip_size > (local.iloc[-1]["atr"] * 0.5):
                        vol_ok = local.iloc[-1]["volume"] > local.iloc[-1]["vol_sma"]
                        dist_to_break = (curr_price - neckline) / curr_price
                        if curr_price < neckline or (dist_to_break < 0.0015 and vol_ok):
                            patterns.append(
                                {
                                    "pattern": "Double Top",
                                    "signal": "SELL",
                                    "direction": "SHORT",
                                    "confidence": 83.0 + (5.0 if vol_ok else 0.0),
                                    "order_type": "STOP",
                                    "price": neckline,
                                }
                            )

        # --- DOUBLE BOTTOM (Bullish) ---
        if len(lows) >= 2:
            l1 = lows.iloc[-2]
            l2 = lows.iloc[-1]
            pos_l2 = lows.index.get_loc(l2.name)
            if is_recent(pos_l2, max_age=15) and abs(l1["low"] - l2["low"]) / l1["low"] < 0.002:
                mask = (local.index > l1.name) & (local.index < l2.name)
                if mask.any():
                    neckline = local.loc[mask, "high"].max()
                    dip_size = neckline - min(l1["low"], l2["low"])
                    if dip_size > (local.iloc[-1]["atr"] * 0.5):
                        vol_ok = local.iloc[-1]["volume"] > local.iloc[-1]["vol_sma"]
                        dist_to_break = (neckline - curr_price) / neckline
                        if curr_price > neckline or (dist_to_break < 0.0015 and vol_ok):
                            patterns.append(
                                {
                                    "pattern": "Double Bottom",
                                    "signal": "BUY",
                                    "direction": "LONG",
                                    "confidence": 83.0 + (5.0 if vol_ok else 0.0),
                                    "order_type": "STOP",
                                    "price": neckline,
                                }
                            )

        # --- HEAD AND SHOULDERS (Bearish) ---
        if len(highs) >= 3:
            l_sh = highs.iloc[-3]
            head = highs.iloc[-2]
            r_sh = highs.iloc[-1]

            # head must be clearly higher than shoulders and shoulders roughly equal
            shoulder_tol = 0.02
            if head["high"] > l_sh["high"] and head["high"] > r_sh["high"]:
                if abs(l_sh["high"] - r_sh["high"]) / l_sh["high"] < shoulder_tol:
                    # neckline: lower of the troughs between L->H and H->R
                    neckline_1 = local.loc[l_sh.name : head.name, "low"].min()
                    neckline_2 = local.loc[head.name : r_sh.name, "low"].min()
                    neckline = min(neckline_1, neckline_2)
                    # require meaningful head-to-neck gap and recent structure
                    if (head["high"] - neckline) > (local.iloc[-1]["atr"] * 0.8):
                        if curr_price < neckline:
                            vol_ok = local.iloc[-1]["volume"] > local.iloc[-1]["vol_sma"]
                            patterns.append(
                                {
                                    "pattern": "Head & Shoulders",
                                    "signal": "SELL",
                                    "direction": "SHORT",
                                    "confidence": 85.0 + (5.0 if vol_ok else 0.0),
                                    "order_type": "STOP",
                                    "price": neckline,
                                }
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
        pole_end = recent.iloc[0]
        atr = curr["atr"]
        if atr == 0 or np.isnan(atr):
            return patterns  # Safety check

        # BULL FLAG
        move_up = pole_end["close"] - pole_start["close"]
        if move_up > (3 * atr):
            flag_body = recent.iloc[:-1]
            vol_drop = flag_body["volume"].mean() < flag_body["vol_sma"].mean()
            held_gains = flag_body["low"].min() > (pole_start["low"] + (0.5 * move_up))

            if vol_drop and held_gains:
                flag_high = flag_body["high"].max()
                if curr["close"] > flag_high and curr["volume"] > curr["vol_sma"]:
                    patterns.append({"pattern": "Bull Flag", "signal": "BUY", "direction": "LONG", "confidence": 85.0})

        # BEAR FLAG
        move_down = pole_start["close"] - pole_end["close"]
        if move_down > (3 * atr):
            flag_body = recent.iloc[:-1]
            vol_drop = flag_body["volume"].mean() < flag_body["vol_sma"].mean()
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
        if len(df) < 5:
            return {"is_fake": False, "risk_score": 0, "reasons": []}

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        risk_score = 0
        reasons = []

        # 1. Low Volume Breakout (Fakeout)
        if curr["high"] > prev["high"] and curr["volume"] < (curr["vol_sma"] * 0.8):
            risk_score += 40
            reasons.append("Low Vol Breakout")

        # 2. RSI Divergence (Exhaustion)
        rsi_5 = df.iloc[-5]["rsi"]
        if curr["high"] > prev["high"] and curr["rsi"] < rsi_5:
            risk_score += 25
            reasons.append("RSI Divergence")

        # 3. Climax Wick (Rejection)
        body = abs(curr["close"] - curr["open"])
        wick_len = curr["high"] - max(curr["close"], curr["open"])
        if wick_len > (body * 2):
            risk_score += 30
            reasons.append("Wick Rejection")

        # 4. Range/Volume mismatch: large range with low volume
        range_ratio = (curr["high"] - curr["low"]) / (curr["atr"] + 1e-9)
        if range_ratio > 1.5 and curr["volume"] < curr["vol_sma"]:
            risk_score += 20
            reasons.append("Range with Low Volume")

        # 5. Quick retracement after breakout (sign of trap)
        # If a breakout is followed by a close back within previous range
        prev_range_high = df.iloc[-3]["high"] if len(df) >= 3 else prev["high"]
        prev_range_low = df.iloc[-3]["low"] if len(df) >= 3 else prev["low"]
        if curr["high"] > prev_range_high and curr["close"] < prev_range_high:
            risk_score += 20
            reasons.append("Retrace After Breakout")

        return {"is_fake": risk_score >= 50, "risk_score": risk_score, "reasons": reasons}
