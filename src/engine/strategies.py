import pandas as pd
from typing import Dict, List, Literal, Optional, Union


class StrategyAnalyzer:
    """
    Pure SMC Strategy Engine strictly aligned with HTF institutional flow.
    """

    def analyze_router(
        self,
        curr: Union[pd.Series, dict],
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        active_fvgs: List[Dict],
        active_ifvgs: List[Dict],
        active_obs: List[Dict],
        structure: dict,
        is_liquidity_swept: int,
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router.
        Enforces HTF alignment and executes strictly on structural confirmations.
        """
        vwap_val = curr.get("vwap", 0) if isinstance(curr, dict) else curr.get("vwap", 0)
        close_price = curr["close"]
        vwap_trend = "BULL" if close_price > vwap_val else "BEAR"

        signal = None

        # Liquidity Sweeps (3-Tiers)
        if not signal:
            signal = self._smc_liquidity_sweep(curr, df, htf_trend, structure, is_liquidity_swept)

        # IFVG Continuation
        if not signal:
            signal = self._ifvg_mitigation(curr, df, htf_trend, structure, active_ifvgs, vwap_trend)

        # POI Reversals (Order Blocks & FVGs)
        if not signal:
            signal = self._smc_poi_reversal(curr, df, htf_trend, structure, active_fvgs, active_obs, vwap_trend)

        # VWAP Bounce
        if not signal:
            signal = self._vwap_bounce(curr, df, htf_trend, structure, vwap_trend)

        if signal:
            signal["is_liquidity_swept"] = is_liquidity_swept
            return signal

        return None

    def _smc_liquidity_sweep(
        self, curr: dict, df: pd.DataFrame, htf_trend: str, structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Executes when liquidity is swept across 3 tiers (Daily, Major 50p, Internal),
        followed immediately by a CHoCH/BOS. Requires HTF alignment since it's a structural reversal.
        """
        bos, choch = structure.get("bos"), structure.get("choch")
        if not (bos or choch) or is_liquidity_swept == 0:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_5", df["low"].tail(5).min())
        recent_high = curr.get("recent_high_5", df["high"].tail(5).max())

        # Determine Strategy Name and Confidence based on Sweep Tier
        if is_liquidity_swept == 3:
            strat_name = "Daily Liquidity Sweep"
            conf = 94.0
        elif is_liquidity_swept == 2:
            strat_name = "Major Swing Sweep (50p)"
            conf = 90.0
        else:
            strat_name = "Internal Sweep Trap"
            conf = 86.0

        #   --- BULLISH SWEEPS   ---
        if (bos == "BULL" or choch == "BULL") and htf_trend == "BULL":
            return {
                "strategy": strat_name,
                "signal": "BUY",
                "direction": "LONG",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        #   --- BEARISH SWEEPS   ---
        if (bos == "BEAR" or choch == "BEAR") and htf_trend == "BEAR":
            return {
                "strategy": strat_name,
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_high + atr_buffer,
            }

        return None

    def _ifvg_mitigation(
        self, curr: dict, df: pd.DataFrame, htf_trend: str, structure: dict, active_ifvgs: List[Dict], vwap_trend: str
    ) -> Optional[Dict]:
        """
        Mitigates IFVG risks by identifying potential re-tests.
        """
        bos, choch = structure.get("bos"), structure.get("choch")
        if not (bos or choch):
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_4", df["low"].tail(4).min())
        recent_high = curr.get("recent_high_4", df["high"].tail(4).max())

        if (bos == "BULL" or choch == "BULL") and htf_trend == "BULL" and vwap_trend == "BULL":
            tapped_ifvg = any(recent_low <= i_fvg["high"] for i_fvg in active_ifvgs if i_fvg["type"] == "BULL")
            if tapped_ifvg:
                return {
                    "strategy": "IFVG Re-Test",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 89.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        if (bos == "BEAR" or choch == "BEAR") and htf_trend == "BEAR" and vwap_trend == "BEAR":
            tapped_ifvg = any(recent_high >= i_fvg["low"] for i_fvg in active_ifvgs if i_fvg["type"] == "BEAR")
            if tapped_ifvg:
                return {
                    "strategy": "IFVG Re-Test",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 89.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None

    def _smc_poi_reversal(
        self,
        curr: Union[pd.Series, dict],
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        structure: dict,
        active_fvgs: List[Dict],
        active_obs: List[Dict],
        vwap_trend: str,
    ) -> Optional[Dict]:
        """
        Detects price bounces off the VWAP with structural confirmation.
        """
        bos, choch = structure.get("bos"), structure.get("choch")
        if not (bos or choch):
            return None

        recent_low = curr.get("recent_low_4", df["low"].tail(4).min())
        recent_high = curr.get("recent_high_4", df["high"].tail(4).max())
        atr_buffer = curr["atr"] * 0.2

        #   --- BULLISH CONFIRMATIONS   ---
        if (bos == "BULL" or choch == "BULL") and htf_trend == "BULL":
            major_ob = any(
                recent_low <= ob["high"] for ob in active_obs if ob["type"] == "BULL" and ob.get("tier") == "MAJOR"
            )
            internal_ob = any(
                recent_low <= ob["high"] for ob in active_obs if ob["type"] == "BULL" and ob.get("tier") == "INTERNAL"
            )
            tapped_fvg = any(recent_low <= fvg["high"] for fvg in active_fvgs if fvg["type"] == "BULL")

            if major_ob:
                return {
                    "strategy": "Major OB Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 88.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }
            if internal_ob:
                return {
                    "strategy": "Internal OB Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 84.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

            if vwap_trend == "BULL" and tapped_fvg:
                return {
                    "strategy": "FVG Bounce + Confirmation",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        #   --- BEARISH CONFIRMATIONS   ---
        if (bos == "BEAR" or choch == "BEAR") and htf_trend == "BEAR":
            major_ob = any(
                recent_high >= ob["low"] for ob in active_obs if ob["type"] == "BEAR" and ob.get("tier") == "MAJOR"
            )
            internal_ob = any(
                recent_high >= ob["low"] for ob in active_obs if ob["type"] == "BEAR" and ob.get("tier") == "INTERNAL"
            )
            tapped_fvg = any(recent_high >= fvg["low"] for fvg in active_fvgs if fvg["type"] == "BEAR")

            if major_ob:
                return {
                    "strategy": "Major OB Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 88.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }
            if internal_ob:
                return {
                    "strategy": "Internal OB Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 84.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

            if vwap_trend == "BEAR" and tapped_fvg:
                return {
                    "strategy": "FVG Bounce + Confirmation",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None

    def _vwap_bounce(
        self,
        curr: Union[pd.Series, dict],
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        structure: dict,
        vwap_trend: str,
    ) -> Optional[Dict]:
        """
        Detects price bounces off the VWAP with structural confirmation.
        """
        bos, choch = structure.get("bos"), structure.get("choch")
        if not (bos or choch):
            return None

        atr_buffer, vwap_val = curr["atr"] * 0.2, curr.get("vwap", 0)

        if (bos == "BULL" or choch == "BULL") and htf_trend == "BULL" and vwap_trend == "BULL":
            recent_low = curr.get("recent_low_4", df["low"].tail(4).min())
            if recent_low <= vwap_val and curr["close"] > vwap_val:
                return {
                    "strategy": "VWAP Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 83.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        if (bos == "BEAR" or choch == "BEAR") and htf_trend == "BEAR" and vwap_trend == "BEAR":
            recent_high = curr.get("recent_high_4", df["high"].tail(4).max())
            if recent_high >= vwap_val and curr["close"] < vwap_val:
                return {
                    "strategy": "VWAP Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 83.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None
