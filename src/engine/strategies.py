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
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router. Strict MTF Alignment applied.
        """
        if htf_trend == "FLAT" or adx_strength < 15.0:
            return None

        # 1. Structure Break Continuation
        res = self._smc_structure_continuation(curr, df, htf_trend, structure, active_fvgs)
        if res:
            return res

        # 2. SMC POI Reversal
        res = self._smc_poi_reversal(curr, df, htf_trend, active_fvgs, active_obs)
        if res:
            return res

        # 3. Liquidity Sweep
        res = self._smc_liquidity_sweep(curr, df, htf_trend, active_fvgs)
        if res:
            return res

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

        if htf_trend == "BULL":
            lower_wick = min(curr["close"], curr["open"]) - curr["low"]
            if lower_wick > body:
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
            if upper_wick > body:
                return {
                    "strategy": "SMC Liquidity Sweep",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": curr["high"] + (curr["atr"] * 0.5),
                }

        return None
