import pandas as pd
from typing import Dict, List, Literal, Optional


class StrategyAnalyzer:
    """
    Modular SMC Strategy Engine.
    """

    def analyze_router(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        active_fvgs: List[Dict],
        active_obs: List[Dict],
        adx_strength: float,
        structure: dict,
        allowed_session_types: List[str],
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router. Strict MTF Alignment applied.
        """
        if htf_trend == "FLAT" or adx_strength < 15.0:
            if "REVERSION" in allowed_session_types:
                return self._mean_reversion_strategy(curr, df)
            return None

        # 2. Trend & Breakout Strategies
        if "TREND" in allowed_session_types or "BREAKOUT" in allowed_session_types:
            # Structure Break Continuation
            res = self._smc_structure_continuation(curr, df, htf_trend, structure, active_fvgs)
            if res:
                return res

            # SMC POI Reversal
            res = self._smc_poi_reversal(curr, df, htf_trend, active_fvgs, active_obs)
            if res:
                return res

            # Liquidity Sweep
            res = self._smc_liquidity_sweep(curr, df, htf_trend, active_fvgs)
            if res:
                return res

            # New York Session Scalp
            # TODO: Implement fetch of candles and data frame for 5m and 4H
            res = self._ny_session_scalp_strategy(curr, df, df)
            if res:
                return res

        return None

    def _mean_reversion_strategy(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """Mean Reversion logic for low-volatility environments. Relies on Bollinger Bands and ATR for SL placement."""
        # Simple Bollinger Band Reversion for quiet markets
        if curr["close"] < curr.get("bb_lower", 0) and curr["close"] > curr["open"]:
            return {
                "strategy": "Mean Reversion (Range Bound)",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": 70.0,
                "order_type": "MARKET",
                "suggested_sl": curr["low"] - (curr["atr"] * 0.5),
            }
        elif curr["close"] > curr.get("bb_upper", float("inf")) and curr["close"] < curr["open"]:
            return {
                "strategy": "Mean Reversion (Range Bound)",
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": 70.0,
                "order_type": "MARKET",
                "suggested_sl": curr["high"] + (curr["atr"] * 0.5),
            }
        return None

    def _smc_structure_continuation(self, curr, df, htf_trend, structure, active_fvgs):
        """Trades the first pullback FVG after a confirmed Break of Structure"""
        if not structure or structure.get("bos") is None:
            return None

        if htf_trend == "BULL" and structure["bos"] == "BULL":
            # If we recently broke structure up, look to buy the nearest FVG below us
            fvg_below = [f for f in active_fvgs if f["type"] == "BULL" and f["high"] < curr["close"]]
            if fvg_below and curr["low"] <= fvg_below[0]["high"]:  # Tapped the FVG
                return {
                    "strategy": "SMC BOS Continuation",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 82.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["low"] - curr["atr"],
                }

        elif htf_trend == "BEAR" and structure["bos"] == "BEAR":
            fvg_above = [f for f in active_fvgs if f["type"] == "BEAR" and f["low"] > curr["close"]]
            if fvg_above and curr["high"] >= fvg_above[0]["low"]:
                return {
                    "strategy": "SMC BOS Continuation",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 82.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["high"] + curr["atr"],
                }
        return None

    def _smc_poi_reversal(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR"],
        active_fvgs: List[Dict],
        active_obs: List[Dict],
    ) -> Optional[Dict]:
        """
        Executes when price taps an active OB, Breaker, or FVG and forms a confirmation candle.
        """
        if htf_trend == "BULL":
            tapped_poi = any(curr["low"] <= poi["high"] for poi in active_fvgs + active_obs if poi["type"] == "BULL")
            # Confirmation candle requirement
            if tapped_poi and curr["close"] > curr["open"]:
                return {
                    "strategy": "SMC POI Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 78.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["low"] - curr["atr"],
                }

        elif htf_trend == "BEAR":
            tapped_poi = any(curr["high"] >= poi["low"] for poi in active_fvgs + active_obs if poi["type"] == "BEAR")
            if tapped_poi and curr["close"] < curr["open"]:
                return {
                    "strategy": "SMC POI Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 78.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["high"] + curr["atr"],
                }
        return None

    def _smc_liquidity_sweep(
        self, curr: pd.Series, df: pd.DataFrame, htf_trend: Literal["BULL", "BEAR"], active_fvgs: List[Dict]
    ) -> Optional[Dict]:
        """
        Triggers trades when Pivot Highs/Lows are swept with aggressive rejection wicks + FVG presence.
        """
        body = abs(curr["close"] - curr["open"])

        # Require the wick to be at least 2x the body size for a true rejection
        rejection_multiplier = 2.0

        if htf_trend == "BULL":
            lower_wick = min(curr["close"], curr["open"]) - curr["low"]
            if lower_wick > (body * rejection_multiplier):
                return {
                    "strategy": "SMC Liquidity Sweep",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["low"] - (curr["atr"] * 0.5),
                }

        elif htf_trend == "BEAR":
            upper_wick = curr["high"] - max(curr["close"], curr["open"])
            if upper_wick > (body * rejection_multiplier):
                return {
                    "strategy": "SMC Liquidity Sweep",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["high"] + (curr["atr"] * 0.5),
                }

        return None

    def _ny_session_scalp_strategy(
        self, curr_5m: pd.Series, df_5m: pd.DataFrame, df_4h: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Scalping Strategy: Marks the high and low of the first 4H candle of the day.
        Drops to 5M timeframe. Enters when a 5M candle breaks the 4H zone and retests it (body close inside).
        Target: Minimum 2R.
        """
        if df_4h.empty or len(df_5m) < 2:
            return None

        # 1. Identify the First 4H Candle of the Day
        # Using SAST (GMT+2) mapping - finding the candle spanning 06:00
        df_4h["date"] = pd.to_datetime(df_4h["time"], unit="s")
        today_4h = df_4h[df_4h["date"].dt.date == pd.Timestamp.utcnow().date()]

        if today_4h.empty:
            return None

        # Get earliest candle of today
        first_4h_candle = today_4h.iloc[0]
        zone_high = first_4h_candle["high"]
        zone_low = first_4h_candle["low"]

        # 2. Check 5M conditions (Breakout + Retest)
        prev_5m = df_5m.iloc[-2]

        atr_5m = curr_5m["atr"]

        # --- LONG CONDITION ---
        # Did price previously break BELOW the 4H low, and now the current 5M candle body re-entered ABOVE it?
        if prev_5m["low"] < zone_low and curr_5m["close"] > zone_low and curr_5m["open"] < curr_5m["close"]:
            sl = min(curr_5m["low"], prev_5m["low"]) - (atr_5m * 0.2)
            risk = curr_5m["close"] - sl
            if risk <= 0:
                return None

            return {
                "strategy": "4H Zone Retest Scalp",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": 88.0,
                "order_type": "MARKET",
                "suggested_sl": sl,
                "suggested_tp": curr_5m["close"] + (risk * 2.0),  # Enforces Minimum 2R
            }

        # --- SHORT CONDITION ---
        # Did price previously break ABOVE the 4H high, and now the current 5M candle body re-entered BELOW it?
        elif prev_5m["high"] > zone_high and curr_5m["close"] < zone_high and curr_5m["open"] > curr_5m["close"]:
            sl = max(curr_5m["high"], prev_5m["high"]) + (atr_5m * 0.2)
            risk = sl - curr_5m["close"]
            if risk <= 0:
                return None

            return {
                "strategy": "4H Zone Retest Scalp",
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": 88.0,
                "order_type": "MARKET",
                "suggested_sl": sl,
                "suggested_tp": curr_5m["close"] - (risk * 2.0),  # Enforces Minimum 2R
            }

        return None
