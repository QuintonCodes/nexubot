import pandas as pd
from typing import Dict, List, Optional, Union


class StrategyAnalyzer:
    """
    Pure SMC Strategy Engine.
    Filters entries based on Volume Profiles and Institutional Order Flow, and Liquidity Sweeps.
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

        pd_status = structure.get("pd_array", 0.5)
        allow_long = pd_status <= 0.60
        allow_short = pd_status >= 0.40

        if not allow_long and not allow_short:
            return None

        signal = None

        # 1. Liquidity Sweeps
        if not signal:
            signal = self._smc_liquidity_sweep(curr, structure, is_liquidity_swept)

        # 2. ICT Optimal Entry
        if not signal:
            signal = self._ict_optimal_trade_entry(curr, structure)

        # 3. IFVG Continuation
        if not signal:
            signal = self._ifvg_mitigation(curr, active_ifvgs, active_obs, is_liquidity_swept)

        # 4. POI Reversals — only when a sweep has already occurred
        if not signal:
            signal = self._smc_poi_reversal(curr, active_fvgs, active_obs, is_liquidity_swept)

        # 5. VWAP Bounce — only when structural context supports it
        vwap_val = curr.get("vwap", 0) if isinstance(curr, dict) else curr.get("vwap", 0)
        if not signal:
            signal = self._vwap_bounce(curr, vwap_val, is_liquidity_swept)

        if signal:
            # Rejection implementation using pre-calculated flags
            if signal["direction"] == "LONG" and not allow_long:
                return None
            if signal["direction"] == "SHORT" and not allow_short:
                return None

            if signal["strategy"] == "FVG Bounce" and signal.get("confidence", 0) < 82.0:
                return None

            signal["is_liquidity_swept"] = is_liquidity_swept
            return signal

        return None

    def _smc_liquidity_sweep(
        self, curr: Union[pd.Series, dict], structure: dict, is_liquidity_swept: int
    ) -> Optional[Dict]:
        """
        Detects aggressive liquidity sweeps with BOS confirmation.
        """
        if is_liquidity_swept < 2:
            return None

        # Only accept BOS confirmation (momentum continuation), not CHoCH (reversal)
        bos = structure.get("bos")
        if not bos:
            return None

        atr_buffer = curr["atr"] * 0.2
        recent_low = curr.get("recent_low_5", 0)
        recent_high = curr.get("recent_high_5", 0)

        if is_liquidity_swept == 3:
            strat_name = "Daily/Asian Sweep"
            conf = 84.0
        else:
            strat_name = "Major Swing Sweep"
            conf = 80.0

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

    def _ict_optimal_trade_entry(self, curr: Union[pd.Series, dict], structure: dict) -> Optional[Dict]:
        """ICT OTE (62-79% Fibonacci Retracement) entry post-BOS."""
        last_low, last_high = structure.get("last_low"), structure.get("last_high")
        if not last_low or not last_high:
            return None

        close_price = curr["close"]
        atr_buffer = curr["atr"] * 0.2
        range_size = last_high - last_low

        if structure["structure"] == "BULL":
            fib_62 = last_high - (range_size * 0.618)
            fib_79 = last_high - (range_size * 0.786)
            if fib_79 <= close_price <= fib_62 and curr["close"] > curr["open"]:
                return {
                    "strategy": "ICT OTE (Bullish)",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": last_low - atr_buffer,
                }
        elif structure["structure"] == "BEAR":
            fib_62 = last_low + (range_size * 0.618)
            fib_79 = last_low + (range_size * 0.786)
            if fib_62 <= close_price <= fib_79 and curr["close"] < curr["open"]:
                return {
                    "strategy": "ICT OTE (Bearish)",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 85.0,
                    "order_type": "MARKET",
                    "suggested_sl": last_high + atr_buffer,
                }
        return None

    def _ifvg_mitigation(
        self,
        curr: Union[pd.Series, dict],
        active_ifvgs: List[Dict],
        active_obs: List[Dict] = None,
        is_liquidity_swept: int = 0,
    ) -> Optional[Dict]:
        """
        Detects bounces off active IFVGs with CE Validation and Overlap Protection.
        """
        if is_liquidity_swept < 2:
            return None

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
                    if close_price < ce_midpoint:
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
        """Detects reversals at Points of Interest (OBs, FVGs) but only after a liquidity sweep has occurred."""
        if is_liquidity_swept < 2:
            return None

        recent_low = curr.get("recent_low_4", 0)
        recent_high = curr.get("recent_high_4", 0)
        close_price = curr["close"]
        atr_buffer = curr["atr"] * 0.5

        is_bouncing_up = close_price > curr["open"]
        is_bouncing_down = close_price < curr["open"]

        # Extract OBs and newly added Breaker Blocks
        valid_bull_obs = [ob for ob in active_obs if ob["type"] == "BULL" and ob.get("mitigations", 0) <= 2]
        valid_bear_obs = [ob for ob in active_obs if ob["type"] == "BEAR" and ob.get("mitigations", 0) <= 2]
        blocked_by_bear = any(
            ob["low"] <= close_price <= ob["high"] for ob in valid_bear_obs if ob.get("tier") in ["MAJOR", "BREAKER"]
        )
        blocked_by_bull = any(
            ob["low"] <= close_price <= ob["high"] for ob in valid_bull_obs if ob.get("tier") in ["MAJOR", "BREAKER"]
        )

        #   --- BULLISH CONFIRMATIONS   ---
        if is_bouncing_up and not blocked_by_bear:
            # Breaker and OB require Consequent Encroachment (CE) crossing
            for ob in valid_bull_obs:
                if ob.get("tier") == "MAJOR":
                    if ob.get("vol_strength", 0) < 2.0 or is_liquidity_swept < 2:
                        continue
                if ob.get("tier") in ["MAJOR", "BREAKER"] and recent_low <= ob["high"]:
                    ce = (ob["high"] + ob["low"]) / 2
                    # Must close past 50% CE to prove momentum
                    if close_price > ce:
                        return {
                            "strategy": f"{ob['tier']} OB CE Bounce",
                            "signal": "BUY",
                            "direction": "LONG",
                            "confidence": 78.0,
                            "order_type": "MARKET",
                            "suggested_sl": recent_low - atr_buffer,
                        }

            if any(
                fvg["type"] == "BULL"
                and fvg.get("mitigations", 0) <= 2
                and recent_low <= fvg["high"]
                and close_price > ((fvg["high"] + fvg["low"]) / 2)
                for fvg in active_fvgs
            ):
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
            for ob in valid_bear_obs:
                if ob.get("tier") == "MAJOR":
                    if ob.get("vol_strength", 0) < 2.0 or is_liquidity_swept < 2:
                        continue
                if ob.get("tier") in ["MAJOR", "BREAKER"] and recent_high >= ob["low"]:
                    ce = (ob["high"] + ob["low"]) / 2
                    if close_price < ce:
                        return {
                            "strategy": f"{ob['tier']} OB CE Bounce",
                            "signal": "SELL",
                            "direction": "SHORT",
                            "confidence": 78.0,
                            "order_type": "MARKET",
                            "suggested_sl": recent_high + atr_buffer,
                        }

            if any(
                fvg["type"] == "BEAR"
                and fvg.get("mitigations", 0) <= 2
                and recent_high >= fvg["low"]
                and close_price < ((fvg["high"] + fvg["low"]) / 2)
                for fvg in active_fvgs
            ):
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
    ) -> Optional[Dict]:
        """
        Detects bounces off the VWAP with volume and structural context validation.
        """
        if vwap_val == 0 or is_liquidity_swept < 2:
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
