import pandas as pd
from typing import Dict, List, Literal, Optional


class StrategyAnalyzer:
    """
    Pure SMC Strategy Engine strictly aligned with HTF institutional flow.
    """

    def analyze_router(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        active_fvgs: List[Dict],
        active_obs: List[Dict],
        structure: Literal["BULL", "BEAR", "FLAT"],
    ) -> Optional[Dict]:
        """
        Unified SMC Strategy Router.
        Enforces HTF alignment and executes strictly on structural confirmations.
        """

        # 1. Calculate Daily Liquidity (PDH / PDL) for Setup 6
        pdh, pdl = None, None
        if "time" in df.columns:
            df_temp = df.copy()
            df_temp["date"] = pd.to_datetime(df_temp["time"], unit="s").dt.date
            today = df_temp["date"].iloc[-1]
            prev_days = df_temp[df_temp["date"] < today]
            if not prev_days.empty:
                last_day = prev_days["date"].iloc[-1]
                yesterday_df = prev_days[prev_days["date"] == last_day]
                pdh = yesterday_df["high"].max()
                pdl = yesterday_df["low"].min()

        daily_levels = {"pdh": pdh, "pdl": pdl}

        # 2. Establish VWAP Trend State
        vwap_trend = "BULL" if curr["close"] > curr.get("vwap", 0) else "BEAR"

        #   -----------------------------------------------------------------
        # Evaluate SMC Setups
        #   -----------------------------------------------------------------

        # Setups 2 & 6: Liquidity Sweeps (Local Pivots & Daily S/R)
        res = self._smc_liquidity_sweep(curr, df, htf_trend, structure, daily_levels)
        if res:
            return res

        # Setups 1 & 4: POI Reversals (Order Blocks & FVGs + Structural Confirmation)
        res = self._smc_poi_reversal(curr, df, htf_trend, structure, active_fvgs, active_obs, vwap_trend)
        if res:
            return res

        # Setup 3: IFVG Continuation
        res = self._ifvg_continuation(curr, df, htf_trend, structure, active_fvgs, vwap_trend)
        if res:
            return res

        # Setup 7: VWAP Bounce
        res = self._vwap_bounce(curr, df, htf_trend, structure, vwap_trend)
        if res:
            return res

        return None

    def _smc_liquidity_sweep(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        structure: Literal["BULL", "BEAR", "FLAT"],
        daily_levels: dict,
    ) -> Optional[Dict]:
        """
        Setups 2 & 6: Executes when significant liquidity is swept, followed immediately by a CHoCH/BOS.
        """
        bos = structure.get("bos")
        choch = structure.get("choch")

        if not (bos or choch):
            return None

        last_low = structure.get("last_low")
        last_high = structure.get("last_high")
        pdl = daily_levels.get("pdl")
        pdh = daily_levels.get("pdh")
        atr_buffer = curr["atr"] * 0.2

        # Lookback window to verify the sweep happened right before the BOS
        recent_low = df["low"].tail(5).min()
        recent_high = df["high"].tail(5).max()

        #   --- BULLISH SWEEPS   ---
        if bos == "BULL" or choch == "BULL":
            # Setup 6: Daily Sweep + Structural Shift
            if pdl and recent_low < pdl and curr["close"] > pdl:
                return {
                    "strategy": "Daily Liquidity Sweep",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 92.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

            # Setup 2: Immediate Trap (Local Pivot Sweep + CHoCH/BOS)
            if last_low and recent_low < last_low and curr["close"] > last_low:
                return {
                    "strategy": "Local Sweep Trap",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 88.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        #   --- BEARISH SWEEPS   ---
        if bos == "BEAR" or choch == "BEAR":
            # Setup 6: Daily Sweep + Structural Shift
            if pdh and recent_high > pdh and curr["close"] < pdh:
                return {
                    "strategy": "Daily Liquidity Sweep",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 92.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

            # Setup 2: Immediate Trap (Local Pivot Sweep + CHoCH/BOS)
            if last_high and recent_high > last_high and curr["close"] < last_high:
                return {
                    "strategy": "Local Sweep Trap",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 88.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None

    def _smc_poi_reversal(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        structure: Literal["BULL", "BEAR", "FLAT"],
        active_fvgs: List[Dict],
        active_obs: List[Dict],
        vwap_trend: str,
    ) -> Optional[Dict]:
        """
        Setups 1 & 4: Price must tap an OB/FVG and confirm with a BOS/CHoCH.
        """
        bos = structure.get("bos")
        choch = structure.get("choch")

        if not (bos or choch):
            return None

        recent_low = df["low"].tail(4).min()
        recent_high = df["high"].tail(4).max()
        atr_buffer = curr["atr"] * 0.2

        #   --- BULLISH CONFIRMATIONS   ---
        if (bos == "BULL" or choch == "BULL") and htf_trend == "BULL":
            # Setup 1: OB Sweep/Tap + BOS
            tapped_ob = any(recent_low <= ob["high"] for ob in active_obs if ob["type"] == "BULL")
            if tapped_ob:
                return {
                    "strategy": "OB Bounce + Confirmation",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 87.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

            # Setup 4: FVG Bounce (Must align with VWAP for continuations)
            if vwap_trend == "BULL":
                tapped_fvg = any(recent_low <= fvg["high"] for fvg in active_fvgs if fvg["type"] == "BULL")
                if tapped_fvg:
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
            # Setup 1: OB Sweep/Tap + BOS
            tapped_ob = any(recent_high >= ob["low"] for ob in active_obs if ob["type"] == "BEAR")
            if tapped_ob:
                return {
                    "strategy": "OB Bounce + Confirmation",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 87.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

            # Setup 4: FVG Bounce (Must align with VWAP for continuations)
            if vwap_trend == "BEAR":
                tapped_fvg = any(recent_high >= fvg["low"] for fvg in active_fvgs if fvg["type"] == "BEAR")
                if tapped_fvg:
                    return {
                        "strategy": "FVG Bounce + Confirmation",
                        "signal": "SELL",
                        "direction": "SHORT",
                        "confidence": 85.0,
                        "order_type": "MARKET",
                        "suggested_sl": recent_high + atr_buffer,
                    }

        return None

    def _ifvg_continuation(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        structure: Literal["BULL", "BEAR", "FLAT"],
        active_fvgs: List[Dict],
        vwap_trend: str,
    ) -> Optional[Dict]:
        """
        Setup 3: Trades failed FVGs (Inversions) aligning with VWAP trend continuation.
        """
        bos = structure.get("bos")
        atr_buffer = curr["atr"] * 0.2

        if bos == "BULL" and vwap_trend == "BULL" and htf_trend == "BULL":
            recent_low = df["low"].tail(4).min()
            # If a BEAR FVG is below price and price bounces off its upper boundary -> IFVG
            tapped_ifvg = any(
                recent_low <= fvg["high"] and curr["close"] > fvg["high"]
                for fvg in active_fvgs
                if fvg["type"] == "BEAR"
            )
            if tapped_ifvg:
                return {
                    "strategy": "IFVG Continuation",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 84.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        if bos == "BEAR" and vwap_trend == "BEAR" and htf_trend == "BEAR":
            recent_high = df["high"].tail(4).max()
            # If a BULL FVG is above price and price rejects off its lower boundary -> IFVG
            tapped_ifvg = any(
                recent_high >= fvg["low"] and curr["close"] < fvg["low"] for fvg in active_fvgs if fvg["type"] == "BULL"
            )
            if tapped_ifvg:
                return {
                    "strategy": "IFVG Continuation",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 84.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None

    def _vwap_bounce(
        self,
        curr: pd.Series,
        df: pd.DataFrame,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        structure: Literal["BULL", "BEAR", "FLAT"],
        vwap_trend: str,
    ) -> Optional[Dict]:
        """
        Setup 7: Price taps daily VWAP, holds it, and confirms with a structural shift.
        """
        bos = structure.get("bos")
        choch = structure.get("choch")

        if not (bos or choch):
            return None

        atr_buffer = curr["atr"] * 0.2

        if (bos == "BULL" or choch == "BULL") and htf_trend == "BULL":
            vwap_val = curr.get("vwap", 0)
            recent_low = df["low"].tail(4).min()
            if recent_low <= vwap_val and curr["close"] > vwap_val:
                return {
                    "strategy": "VWAP Bounce + Confirmation",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 83.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_low - atr_buffer,
                }

        if (bos == "BEAR" or choch == "BEAR") and htf_trend == "BEAR":
            vwap_val = curr.get("vwap", float("inf"))
            recent_high = df["high"].tail(4).max()
            if recent_high >= vwap_val and curr["close"] < vwap_val:
                return {
                    "strategy": "VWAP Bounce + Confirmation",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 83.0,
                    "order_type": "MARKET",
                    "suggested_sl": recent_high + atr_buffer,
                }

        return None
