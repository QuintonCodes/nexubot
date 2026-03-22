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
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router. Strict MTF Alignment applied.
        """
        # 1. Strict MTF Alignment & ADX Trend Filter
        if htf_trend == "FLAT" or adx_strength < 20.0:
            return None

        # 2. SMC POI Reversal
        res = self._smc_poi_reversal(curr, df, htf_trend, active_fvgs, active_obs)
        if res:
            return res

        # 3. Liquidity Sweep
        res = self._smc_liquidity_sweep(curr, df, htf_trend, active_fvgs)
        if res:
            return res

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
            # Confirmation candle requirement
            if curr.get("bull_engulfing") or curr.get("bull_pin") or curr.get("doji"):
                # Simplified check for tapping POI
                tapped_poi = any(
                    curr["low"] <= poi["high"] for poi in active_fvgs + active_obs if poi["type"] == "BULL"
                )
                if tapped_poi:
                    return {
                        "strategy": "SMC POI Reversal",
                        "signal": "BUY",
                        "direction": "LONG",
                        "confidence": 80.0,
                        "order_type": "MARKET",
                        "suggested_sl": curr["low"] - curr["atr"],
                    }

        elif htf_trend == "BEAR":
            if curr.get("bear_engulfing") or curr.get("bear_pin") or curr.get("doji"):
                tapped_poi = any(
                    curr["high"] >= poi["low"] for poi in active_fvgs + active_obs if poi["type"] == "BEAR"
                )
                if tapped_poi:
                    return {
                        "strategy": "SMC POI Reversal",
                        "signal": "SELL",
                        "direction": "SHORT",
                        "confidence": 80.0,
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
            # Aggressive rejection sweep condition
            if lower_wick > (body * 1.5):
                # Check if FVG exists above the sweep (validating the sweep's structural integrity)
                fvg_above = any(fvg["low"] > curr["low"] for fvg in active_fvgs if fvg["type"] == "BULL")
                if fvg_above:
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
            if upper_wick > (body * 1.5):
                fvg_below = any(fvg["high"] < curr["high"] for fvg in active_fvgs if fvg["type"] == "BEAR")
                if fvg_below:
                    return {
                        "strategy": "SMC Liquidity Sweep",
                        "signal": "SELL",
                        "direction": "SHORT",
                        "confidence": 85.0,
                        "order_type": "MARKET",
                        "suggested_sl": curr["high"] + (curr["atr"] * 0.5),
                    }

        return None
