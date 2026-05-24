import pandas as pd
from typing import Dict, List, Optional, Union


class StrategyAnalyzer:
    """
    Pure SMC Strategy Engine.
    Filters entries based on Volume Profiles and Institutional Order Flow.
    """

    def analyze_router(
        self,
        curr: Union[pd.Series, dict],
        active_fvgs: List[Dict],
        active_ifvgs: List[Dict],
        active_obs: List[Dict],
        structure: dict,
        is_liquidity_swept: int,
        htf_trend: float,
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router.
        """
        vwap_val = curr.get("vwap", 0) if isinstance(curr, dict) else curr.get("vwap", 0)
        signal = None

        # 1. Liquidity Sweeps
        if not signal:
            signal = self._smc_liquidity_sweep(curr, structure, is_liquidity_swept)

        # 2. Counter-Trend Mean Reversion
        if not signal:
            signal = self._smc_counter_reversion(curr, structure, is_liquidity_swept)

        # 3. IFVG Continuation (Direction dictated by the IFVG itself)
        if not signal:
            signal = self._ifvg_mitigation(curr, active_ifvgs)

        # 4. POI Reversals (Pullbacks to Order Blocks & FVGs)
        if not signal:
            signal = self._smc_poi_reversal(curr, active_fvgs, active_obs)

        # 5. VWAP Bounce
        if not signal:
            signal = self._vwap_bounce(curr, vwap_val)

        if signal:
            signal["is_liquidity_swept"] = is_liquidity_swept
            return signal

        return None

    def _smc_liquidity_sweep(
        self, curr: Union[pd.Series, dict], structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Detects liquidity sweeps that trigger a structural break.
        """
        struct_break = structure.get("choch") if structure.get("choch") else structure.get("bos")
        if not struct_break or is_liquidity_swept == 0:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_5", 0)
        recent_high = curr.get("recent_high_5", 0)

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
        if struct_break == "BULL":
            return {
                "strategy": strat_name,
                "signal": "BUY",
                "direction": "LONG",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        #   --- BEARISH SWEEPS   ---
        if struct_break == "BEAR":
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
        self, curr: Union[pd.Series, dict], structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Counter-Trend Mean Reversion on Major sweeps.
        """
        struct_break = structure.get("choch") if structure.get("choch") else structure.get("bos")

        if not struct_break or is_liquidity_swept < 2:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_5", 0)
        recent_high = curr.get("recent_high_5", 0)

        strat_name = "Daily Reversion (Counter)" if is_liquidity_swept == 3 else "Major Sweep (Counter)"
        conf = 82.0 if is_liquidity_swept == 3 else 76.0

        # Swept a high, breaking bearish structurally (ML handles HTF alignment penalty)
        if struct_break == "BEAR":
            return {
                "strategy": strat_name,
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_high + atr_buffer,
            }

        if struct_break == "BULL":
            return {
                "strategy": strat_name,
                "signal": "BUY",
                "direction": "LONG",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        return None

    def _ifvg_mitigation(self, curr: Union[pd.Series, dict], active_ifvgs: List[Dict]) -> Optional[Dict]:
        """
        Detects bounces off active IFVGs. Direction is HARD OVERRIDDEN by the IFVG nature.
        """
        atr_buffer = curr["atr"] * 0.5
        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)

        # Ensure we are actually bouncing (candle is green for longs, red for shorts)
        is_bouncing_up = curr["close"] > curr["open"]
        is_bouncing_down = curr["close"] < curr["open"]

        if is_bouncing_up:
            tapped_bullish_ifvg = any(recent_low <= i_fvg["high"] for i_fvg in active_ifvgs if i_fvg["type"] == "BULL")
            if tapped_bullish_ifvg:
                return {
                    "strategy": "IFVG Re-Test",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 89.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        if is_bouncing_down:
            tapped_bearish_ifvg = any(recent_high >= i_fvg["low"] for i_fvg in active_ifvgs if i_fvg["type"] == "BEAR")
            if tapped_bearish_ifvg:
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
        active_fvgs: List[Dict],
        active_obs: List[Dict],
    ) -> Optional[Dict]:
        """
        Detects bounces off active FVGs and Order Blocks.
        """
        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)
        atr_buffer = curr["atr"] * 0.5

        is_bouncing_up = curr["close"] > curr["open"]
        is_bouncing_down = curr["close"] < curr["open"]

        # Filter OBs by minimum displacement volume and sort by strength
        valid_bull_obs = [ob for ob in active_obs if ob["type"] == "BULL" and ob.get("vol_strength", 1.0) >= 1.1]
        valid_bull_obs.sort(key=lambda x: x.get("vol_strength", 0), reverse=True)

        valid_bear_obs = [ob for ob in active_obs if ob["type"] == "BEAR" and ob.get("vol_strength", 1.0) >= 1.1]
        valid_bear_obs.sort(key=lambda x: x.get("vol_strength", 0), reverse=True)

        #   --- BULLISH CONFIRMATIONS   ---
        if is_bouncing_up:
            major_ob = next(
                (ob for ob in valid_bull_obs if ob.get("tier") == "MAJOR" and recent_low <= ob["high"]), None
            )
            internal_ob = next(
                (ob for ob in valid_bull_obs if ob.get("tier") == "INTERNAL" and recent_low <= ob["high"]), None
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
        if is_bouncing_down:
            major_ob = next(
                (ob for ob in valid_bear_obs if ob.get("tier") == "MAJOR" and recent_high >= ob["low"]), None
            )
            internal_ob = next(
                (ob for ob in valid_bear_obs if ob.get("tier") == "INTERNAL" and recent_high >= ob["low"]), None
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
        vwap_val: float,
    ) -> Optional[Dict]:
        """
        VWAP Bounces.
        """
        if vwap_val == 0:
            return None

        atr_buffer = curr["atr"] * 0.3
        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)

        # Demand institutional volume backing the bounce (20% above average volume)
        vol_sma = curr.get("vol_sma_20", 1)
        if vol_sma == 0:
            vol_sma = 1
        vol_strength = curr.get("volume", 0) / vol_sma

        if vol_strength < 1.2:
            return None  # Ignore weak retail bounces, wait for institutional displacement

        if recent_low <= vwap_val and curr["close"] > vwap_val and curr["close"] > curr["open"]:
            return {
                "strategy": "VWAP Bounce",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": 83.0,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

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
