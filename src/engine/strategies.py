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
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router. Returns the first qualifying signal.
        """
        # Zones touched 3+ times are exhausted — skip regardless of strategy
        mitigation_count = (
            curr.get("mitigation_count", 0) if isinstance(curr, dict) else curr.get("mitigation_count", 0)
        )

        if mitigation_count >= 3:
            return None

        signal = None

        # 1. Liquidity Sweeps
        if not signal:
            signal = self._smc_liquidity_sweep(curr, structure, is_liquidity_swept)

        # 2. IFVG Continuation
        if not signal:
            signal = self._ifvg_mitigation(curr, active_ifvgs, active_obs)

        # 3. POI Reversals — only when a sweep has already occurred
        if not signal:
            signal = self._smc_poi_reversal(curr, active_fvgs, active_obs, is_liquidity_swept)

        # 4. VWAP Bounce — only when structural context supports it
        if not signal:
            vwap_val = curr.get("vwap", 0) if isinstance(curr, dict) else curr.get("vwap", 0)
            structural_break = structure.get("structural_break", 0.0)
            signal = self._vwap_bounce(curr, vwap_val, is_liquidity_swept, structural_break)

        if signal:
            signal["is_liquidity_swept"] = is_liquidity_swept
            return signal

        return None

    def _smc_liquidity_sweep(
        self, curr: Union[pd.Series, dict], structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Detects liquidity sweeps that trigger a structural break.
        Only BOS signals are taken.
        """
        if is_liquidity_swept == 0:
            return None

        # Only accept BOS confirmation (momentum continuation), not CHoCH (reversal)
        # CHoCH entries (counter-trend) are structurally unsound at M1 granularity
        bos = structure.get("bos")
        if not bos:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_5", 0)
        recent_high = curr.get("recent_high_5", 0)

        # Determine Strategy Name and Confidence based on Sweep Tier
        if is_liquidity_swept == 3:
            strat_name = "Daily Liquidity Sweep"
            conf = 84.0
        elif is_liquidity_swept == 2:
            strat_name = "Major Swing Sweep (50p)"
            conf = 80.0
        else:
            strat_name = "Internal Sweep Trap"
            conf = 76.0

        #   --- BULLISH SWEEPS   ---
        if bos == "BULL":
            return {
                "strategy": strat_name,
                "signal": "BUY",
                "direction": "LONG",
                "confidence": conf,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        #   --- BEARISH SWEEPS   ---
        if bos == "BEAR":
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
        self, curr: Union[pd.Series, dict], active_ifvgs: List[Dict], active_obs: List[Dict] = None
    ) -> Optional[Dict]:
        """
        Detects bounces off active IFVGs with CE Validation and Overlap Protection.
        """
        active_obs = active_obs or []
        atr_buffer = curr["atr"] * 0.5
        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)
        close_price = curr["close"]

        # Ensure we are actually bouncing (candle is green for longs, red for shorts)
        is_bouncing_up = close_price > curr["open"]
        is_bouncing_down = close_price < curr["open"]

        # Opposing Blockade Check (Don't buy into a Major Bearish OB, don't sell into a Major Bullish OB)
        blocked_by_bear = any(
            ob["low"] <= close_price <= ob["high"]
            for ob in active_obs
            if ob["type"] == "BEAR" and ob.get("tier") == "MAJOR"
        )
        blocked_by_bull = any(
            ob["low"] <= close_price <= ob["high"]
            for ob in active_obs
            if ob["type"] == "BULL" and ob.get("tier") == "MAJOR"
        )

        if is_bouncing_up and not blocked_by_bear:
            for i_fvg in active_ifvgs:
                # Skip exhausted zones
                if i_fvg.get("mitigations", 0) > 2:
                    continue
                if i_fvg["type"] == "BULL" and recent_low <= i_fvg["high"]:
                    ce_midpoint = (i_fvg["high"] + i_fvg["low"]) / 2
                    if close_price > ce_midpoint:  # Must close above midpoint to prove rejection
                        return {
                            "strategy": "IFVG Re-Test",
                            "signal": "BUY",
                            "direction": "LONG",
                            "confidence": 78.0,
                            "order_type": "MARKET",
                            "suggested_sl": recent_low - atr_buffer,
                        }

        if is_bouncing_down and not blocked_by_bull:
            for i_fvg in active_ifvgs:
                if i_fvg.get("mitigations", 0) > 2:
                    continue
                if i_fvg["type"] == "BEAR" and recent_high >= i_fvg["low"]:
                    ce_midpoint = (i_fvg["high"] + i_fvg["low"]) / 2
                    if close_price < ce_midpoint:  # Must close below midpoint to prove rejection
                        return {
                            "strategy": "IFVG Re-Test",
                            "signal": "SELL",
                            "direction": "SHORT",
                            "confidence": 78.0,
                            "order_type": "MARKET",
                            "suggested_sl": recent_high + atr_buffer,
                        }

        return None

    def _smc_poi_reversal(
        self,
        curr: Union[pd.Series, dict],
        active_fvgs: List[Dict],
        active_obs: List[Dict],
        is_liquidity_swept: int = 0,
    ) -> Optional[Dict]:
        """
        Detects bounces off active FVGs and Order Blocks with CE Validation.
        Requires is_liquidity_swept > 0.
        """
        # Require at least a Tier-1 sweep before entering a POI reversal
        if is_liquidity_swept == 0:
            return None

        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)
        close_price = curr["close"]
        atr_buffer = curr["atr"] * 0.5

        is_bouncing_up = close_price > curr["open"]
        is_bouncing_down = close_price < curr["open"]

        # Filter OBs by minimum displacement volume and sort by strength
        valid_bull_obs = [
            ob
            for ob in active_obs
            if ob["type"] == "BULL" and ob.get("vol_strength", 1.0) >= 1.1 and ob.get("mitigations", 0) <= 2
        ]
        valid_bull_obs.sort(key=lambda x: x.get("vol_strength", 0), reverse=True)

        valid_bear_obs = [
            ob
            for ob in active_obs
            if ob["type"] == "BEAR" and ob.get("vol_strength", 1.0) >= 1.1 and ob.get("mitigations", 0) <= 2
        ]
        valid_bear_obs.sort(key=lambda x: x.get("vol_strength", 0), reverse=True)

        # Blockades
        blocked_by_bear = any(
            ob["low"] <= close_price <= ob["high"] for ob in valid_bear_obs if ob.get("tier") == "MAJOR"
        )
        blocked_by_bull = any(
            ob["low"] <= close_price <= ob["high"] for ob in valid_bull_obs if ob.get("tier") == "MAJOR"
        )

        #   --- BULLISH CONFIRMATIONS   ---
        if is_bouncing_up and not blocked_by_bear:
            major_ob = next(
                (ob for ob in valid_bull_obs if ob.get("tier") == "MAJOR" and recent_low <= ob["high"]), None
            )
            internal_ob = next(
                (ob for ob in valid_bull_obs if ob.get("tier") == "INTERNAL" and recent_low <= ob["high"]), None
            )

            tapped_fvg = any(
                fvg["type"] == "BULL"
                and fvg.get("mitigations", 0) <= 2
                and recent_low <= fvg["high"]
                and close_price > ((fvg["high"] + fvg["low"]) / 2)
                for fvg in active_fvgs
            )

            if major_ob:
                return {
                    "strategy": "Major OB Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 78.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }
            if internal_ob:
                return {
                    "strategy": "Internal OB Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 74.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

            if tapped_fvg:
                return {
                    "strategy": "FVG Bounce",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 75.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        #   --- BEARISH CONFIRMATIONS   ---
        if is_bouncing_down and not blocked_by_bull:
            major_ob = next(
                (ob for ob in valid_bear_obs if ob.get("tier") == "MAJOR" and recent_high >= ob["low"]), None
            )
            internal_ob = next(
                (ob for ob in valid_bear_obs if ob.get("tier") == "INTERNAL" and recent_high >= ob["low"]), None
            )

            tapped_fvg = any(
                fvg["type"] == "BEAR"
                and fvg.get("mitigations", 0) <= 2
                and recent_high >= fvg["low"]
                and close_price < ((fvg["high"] + fvg["low"]) / 2)
                for fvg in active_fvgs
            )

            if major_ob:
                return {
                    "strategy": "Major OB Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 78.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }
            if internal_ob:
                return {
                    "strategy": "Internal OB Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 74.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

            if tapped_fvg:
                return {
                    "strategy": "FVG Bounce",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 75.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None

    def _vwap_bounce(
        self,
        curr: Union[pd.Series, dict],
        vwap_val: float,
        is_liquidity_swept: int = 0,
        structural_break: float = 0.0,
    ) -> Optional[Dict]:
        """
        VWAP Bounces — only when structural context exists.

        Requires either a liquidity sweep or a structural break to be present.
        """
        if vwap_val == 0:
            return None

        # Require structural context: either a sweep or a structure break
        if is_liquidity_swept == 0 and structural_break == 0.0:
            return None

        # Demand institutional volume backing the bounce (20% above average volume)
        vol_sma = curr.get("vol_sma_20", 1)
        if vol_sma == 0:
            vol_sma = 1
        vol_strength = curr.get("volume", 0) / vol_sma

        if vol_strength < 1.2:
            return None  # Ignore weak retail bounces, wait for institutional displacement

        atr_buffer = curr["atr"] * 0.3
        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)

        if recent_low <= vwap_val and curr["close"] > vwap_val and curr["close"] > curr["open"]:
            return {
                "strategy": "VWAP Bounce",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": 73.0,
                "order_type": "MARKET",
                "suggested_sl": recent_low - atr_buffer,
            }

        if recent_high >= vwap_val and curr["close"] < vwap_val and curr["close"] < curr["open"]:
            return {
                "strategy": "VWAP Bounce",
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": 73.0,
                "order_type": "MARKET",
                "suggested_sl": recent_high + atr_buffer,
            }

        return None
