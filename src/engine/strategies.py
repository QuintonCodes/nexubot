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
        signal = None

        # 1. Liquidity Sweeps (Requires CHoCH to confirm reversal)
        if not signal:
            signal = self._smc_liquidity_sweep(curr, df, htf_trend, structure, is_liquidity_swept)

        # 2. Counter-Trend Mean Reversion
        if not signal:
            signal = self._smc_counter_reversion(curr, df, htf_trend, structure, is_liquidity_swept)

        # 3. IFVG Continuation (Requires established structure memory)
        if not signal:
            signal = self._ifvg_mitigation(curr, df, htf_trend, structure, active_ifvgs)

        # 4. POI Reversals (Pullbacks to Order Blocks & FVGs)
        if not signal:
            signal = self._smc_poi_reversal(curr, df, htf_trend, structure, active_fvgs, active_obs)

        # 5. VWAP Bounce (Wick below, close above)
        if not signal:
            signal = self._vwap_bounce(curr, df, htf_trend, structure, vwap_val)

        if signal:
            signal["is_liquidity_swept"] = is_liquidity_swept
            return signal

        return None

    def _smc_liquidity_sweep(
        self, curr: dict, df: pd.DataFrame, htf_trend: str, structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Detects powerful liquidity sweeps that trigger a structural break (CHoCH or BOS) and align with HTF trend.
        """
        choch = structure.get("choch")
        bos = structure.get("bos")

        # Valid break is either a Reversal (CHoCH) or Continuation (BOS)
        struct_break = choch if choch else bos

        if not struct_break or is_liquidity_swept == 0:
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
        if struct_break == "BULL" and htf_trend == "BULL":
            return {
                "strategy": strat_name,
                "signal": "BUY",
                "direction": "LONG",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        #   --- BEARISH SWEEPS   ---
        if struct_break == "BEAR" and htf_trend == "BEAR":
            return {
                "strategy": strat_name,
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_high + atr_buffer,
            }

        return None

    def _smc_counter_reversion(
        self, curr: dict, df: pd.DataFrame, htf_trend: str, structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Counter-Trend Mean Reversion.
        Triggers strictly on Major (Tier 2) or Daily (Tier 3) Liquidity Sweeps where local structure reverses.
        """
        choch = structure.get("choch")
        bos = structure.get("bos")

        struct_break = choch if choch else bos

        # Require a structural break and a major liquidity sweep to validate trading against the HTF
        if not struct_break or is_liquidity_swept < 2:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_5", df["low"].tail(5).min())
        recent_high = curr.get("recent_high_5", df["high"].tail(5).max())

        strat_name = "Daily Reversion (Counter)" if is_liquidity_swept == 3 else "Major Sweep (Counter)"
        conf = 82.0 if is_liquidity_swept == 3 else 76.0

        # BEARISH Reversion: Market pumps, sweeps a major high, prints a bearish break, but HTF is BULL/FLAT
        if struct_break == "BEAR" and htf_trend in ["BULL", "FLAT"]:
            return {
                "strategy": strat_name,
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_high + atr_buffer,
            }

        # BULLISH Reversion: Market dumps, sweeps a major low, prints a bullish CHoCH, but HTF is BEAR/FLAT
        if struct_break == "BULL" and htf_trend in ["BEAR", "FLAT"]:
            return {
                "strategy": strat_name,
                "signal": "BUY",
                "direction": "LONG",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        return None

    def _ifvg_mitigation(
        self, curr: dict, df: pd.DataFrame, htf_trend: str, structure: dict, active_ifvgs: List[Dict]
    ) -> Optional[Dict]:
        """
        Detects bounces off active FVGs with structural and HTF trend confirmation.
        """
        current_struct = structure.get("structure")
        if current_struct == "FLAT":
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_4", df["low"].tail(4).min())
        recent_high = curr.get("recent_high_4", df["high"].tail(4).max())

        # Ensure we are actually bouncing (candle is green for longs, red for shorts)
        is_bouncing_up = curr["close"] > curr["open"]
        is_bouncing_down = curr["close"] < curr["open"]

        if current_struct == "BULL" and htf_trend == "BULL" and is_bouncing_up:
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

        if current_struct == "BEAR" and htf_trend == "BEAR" and is_bouncing_down:
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
    ) -> Optional[Dict]:
        """
        Detects bounces off active FVGs and Order Blocks with structural confirmation.
        """
        current_struct = structure.get("structure")
        if current_struct == "FLAT":
            return None

        recent_low = curr.get("recent_low_4", df["low"].tail(4).min())
        recent_high = curr.get("recent_high_4", df["high"].tail(4).max())
        atr_buffer = curr["atr"] * 0.2

        is_bouncing_up = curr["close"] > curr["open"]
        is_bouncing_down = curr["close"] < curr["open"]

        #   --- BULLISH CONFIRMATIONS   ---
        if current_struct == "BULL" and htf_trend == "BULL" and is_bouncing_up:
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

            if tapped_fvg:
                return {
                    "strategy": "FVG Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        #   --- BEARISH CONFIRMATIONS   ---
        if current_struct == "BEAR" and htf_trend == "BEAR" and is_bouncing_down:
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

            if tapped_fvg:
                return {
                    "strategy": "FVG Bounce",
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
        vwap_val: float,
    ) -> Optional[Dict]:
        """
        VWAP Bounces require a wick below VWAP but a close above it (for longs) or vice versa for shorts.
        """
        current_struct = structure.get("structure")
        if current_struct == "FLAT" or vwap_val == 0:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_4", df["low"].tail(4).min())
        recent_high = curr.get("recent_high_4", df["high"].tail(4).max())

        # VWAP Logic: Price is allowed to wick below VWAP, but current candle must close above it (bounce)
        if current_struct == "BULL" and htf_trend == "BULL":
            if recent_low <= vwap_val and curr["close"] > vwap_val and curr["close"] > curr["open"]:
                return {
                    "strategy": "VWAP Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 83.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        if current_struct == "BEAR" and htf_trend == "BEAR":
            if recent_high >= vwap_val and curr["close"] < vwap_val and curr["close"] < curr["open"]:
                return {
                    "strategy": "VWAP Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 83.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None
