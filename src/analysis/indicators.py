import pandas as pd
from typing import Dict, Tuple, Union


class TechnicalAnalyzer:
    """
    Advanced SMC data processing engine.
    Calculates essential structural and volatility metrics without lagging indicators.
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies technical indicators to the provided DataFrame.
        """
        # Ensure DateTime column exists for Time-based indicators
        if "datetime" not in df.columns:
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            else:
                df["datetime"] = pd.Timestamp.now()

        # 1. ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # 2. Daily VWAP
        df["pv"] = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
        df["date_group"] = df["datetime"].dt.date
        df["cum_pv"] = df.groupby("date_group")["pv"].cumsum()
        df["cum_vol"] = df.groupby("date_group")["volume"].cumsum()
        df["vwap"] = df["cum_pv"] / df["cum_vol"]

        # 3. Daily Open (SMC Premium/Discount Baseline)
        df["daily_open"] = df.groupby("date_group")["open"].transform("first")

        # 4. Internal Pivot Tracking
        df["pivot_high"] = df["high"] == df["high"].rolling(6, center=True).max()
        df["pivot_low"] = df["low"] == df["low"].rolling(6, center=True).min()

        # 5. Volume Profile for Institutional Displacement
        df["vol_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()

        # 6. Lookback Windows
        df["recent_low_5"] = df["low"].rolling(5).min()
        df["recent_high_5"] = df["high"].rolling(5).max()
        df["recent_low_4"] = df["low"].rolling(4).min()
        df["recent_high_4"] = df["high"].rolling(4).max()
        df["major_low_50"] = df["low"].rolling(50).min().shift(5)
        df["major_high_50"] = df["high"].rolling(50).max().shift(5)

        # 7. HTF Trend (Institutional EMAs)
        # Cast to float to ensure mathematical calculation avoids object-type bugs
        close_series = df["close"].astype(float)

        df["ema_50"] = close_series.ewm(span=50, adjust=False).mean()
        df["ema_200"] = close_series.ewm(span=200, adjust=False).mean()

        df["htf_trend"] = 0.0
        df.loc[df["ema_50"] > df["ema_200"], "htf_trend"] = 1.0
        df.loc[df["ema_50"] < df["ema_200"], "htf_trend"] = -1.0

        return df.fillna(0)

    @staticmethod
    def detect_liquidity_sweeps(curr: Union[pd.Series, dict], structure: dict, daily_levels: dict) -> Tuple[int, float]:
        """
        Unified Liquidity Sweep Logic.
        Returns: (Sweep Tier [0-3], Sweep Depth in ATR)
        """
        pdl = daily_levels.get("pdl")
        pdh = daily_levels.get("pdh")
        last_low = structure.get("last_low")
        last_high = structure.get("last_high")

        close_price = curr["close"]
        recent_low_5 = curr.get("recent_low_5", 0)
        recent_high_5 = curr.get("recent_high_5", 0)
        major_low_50 = curr.get("major_low_50", None)
        major_high_50 = curr.get("major_high_50", None)

        atr = curr.get("atr", 1.0)
        if atr == 0:
            atr = 1.0

        is_swept = 0
        sweep_depth = 0.0

        # Tier 3: Daily Sweeps (Most Significant)
        if pdl and recent_low_5 < pdl and close_price > pdl:
            is_swept = 3
            sweep_depth = (pdl - recent_low_5) / atr
        elif pdh and recent_high_5 > pdh and close_price < pdh:
            is_swept = 3
            sweep_depth = (recent_high_5 - pdh) / atr

        # Tier 2: Major 50-Period Sweeps
        elif major_low_50 and recent_low_5 < major_low_50 and close_price > major_low_50:
            is_swept = 2
            sweep_depth = (major_low_50 - recent_low_5) / atr
        elif major_high_50 and recent_high_5 > major_high_50 and close_price < major_high_50:
            is_swept = 2
            sweep_depth = (recent_high_5 - major_high_50) / atr

        # Tier 1: Internal Sweeps (Local Structural Pivots)
        elif last_low and recent_low_5 < last_low and close_price > last_low:
            is_swept = 1
            sweep_depth = (last_low - recent_low_5) / atr
        elif last_high and recent_high_5 > last_high and close_price < last_high:
            is_swept = 1
            sweep_depth = (recent_high_5 - last_high) / atr

        return is_swept, sweep_depth

    @staticmethod
    def detect_structure(df: pd.DataFrame) -> Dict:
        """
        Detects BOS and CHoCH strictly via candle closes.
        Calculates Premium/Discount Array Status.
        """
        if len(df) < 20:
            return {"bos": None, "choch": None, "structure": "FLAT", "structural_break": 0.0, "pd_array": 0.5}

        recent_df = df.tail(200)
        confirmed_df = recent_df.iloc[:-3]

        # Extract actual pivot prices
        highs = confirmed_df[confirmed_df["pivot_high"]]["high"].values
        lows = confirmed_df[confirmed_df["pivot_low"]]["low"].values

        if len(highs) < 2 or len(lows) < 2:
            return {"bos": None, "choch": None, "structure": "FLAT", "structural_break": 0.0, "pd_array": 0.5}

        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]
        current_close = df.iloc[-1]["close"]

        # 1. Determine local structure/trend based on previous swings
        is_uptrend = (last_high > prev_high) and (last_low > prev_low)
        is_downtrend = (last_high < prev_high) and (last_low < prev_low)
        structure = "BULL" if is_uptrend else ("BEAR" if is_downtrend else "FLAT")

        bos, choch = None, None
        structural_break = 0.0

        # 2. Detect BOS and CHoCH on the live edge (Requires body close)
        if structure == "BULL":
            if current_close > last_high:
                bos = "BULL"
                structural_break = 1.0
            elif current_close < last_low:
                choch = "BEAR"
                structural_break = -2.0
        elif structure == "BEAR":
            if current_close < last_low:
                bos = "BEAR"
                structural_break = -1.0
            elif current_close > last_high:
                choch = "BULL"
                structural_break = 2.0

        # Premium / Discount Calculation (0.0 to 1.0)
        # 0.0 = At recent low (Discount), 1.0 = At recent high (Premium)
        pd_range = last_high - last_low
        pd_array_status = 0.5
        if pd_range > 0:
            pd_array_status = (current_close - last_low) / pd_range
            pd_array_status = max(0.0, min(1.0, pd_array_status))

        return {
            "bos": bos,
            "choch": choch,
            "structure": structure,
            "last_high": last_high,
            "last_low": last_low,
            "structural_break": structural_break,
            "pd_array": pd_array_status,
        }

    @staticmethod
    def extract_active_pois(data: Union[pd.DataFrame, list]) -> Tuple[list, list, list]:
        """
        Extracts POIs, handles conversions, and tracks their Mitigation Count.
        """

        active_fvgs, active_ifvgs, active_obs = [], [], []

        records = data.to_dict("records") if isinstance(data, pd.DataFrame) else data
        if len(records) < 3:
            return [], [], []

        for i in range(2, len(records)):
            c1, c2, curr = records[i - 2], records[i - 1], records[i]
            curr_low, curr_high = curr["low"], curr["high"]

            # 1. Manage FVGs -> Mitigated FVGs become IFVGs (Inverse Fair Value Gaps)
            for f in active_fvgs[:]:
                if f["type"] == "BULL" and curr_low < f["low"]:
                    # Broken Bullish FVG -> Bearish IFVG
                    active_ifvgs.append({"type": "BEAR", "high": f["high"], "low": f["low"], "mitigations": 0})
                    active_fvgs.remove(f)
                elif f["type"] == "BEAR" and curr_high > f["high"]:
                    # Broken Bearish FVG -> Bullish IFVG
                    active_ifvgs.append({"type": "BULL", "high": f["high"], "low": f["low"], "mitigations": 0})
                    active_fvgs.remove(f)

            # 2. Track Mitigations & Invalidate fully broken zones
            for pool in [active_fvgs, active_ifvgs, active_obs]:
                for zone in pool[:]:
                    # Check for mitigation touches (Wick crosses into the POI)
                    if zone["type"] == "BULL" and curr_low <= zone["high"]:
                        zone["mitigations"] += 1
                    elif zone["type"] == "BEAR" and curr_high >= zone["low"]:
                        zone["mitigations"] += 1

                    # Remove heavily mitigated/broken zones (Over 3 touches = void)
                    if (
                        zone["mitigations"] > 3
                        or (zone["type"] == "BULL" and curr["close"] < zone["low"])
                        or (zone["type"] == "BEAR" and curr["close"] > zone["high"])
                    ):
                        pool.remove(zone)

            # 3. Detect new FVG (Now strictly filtered by Institutional Volume)
            vol_sma = c2.get("vol_sma_20", 1)
            vol_strength = round((c2["volume"] / vol_sma), 2) if vol_sma > 0 else 1.0

            # Only register FVGs with volume strength >= 1.0 (average or above average displacement)
            if vol_strength >= 1.0:
                if c1["high"] < curr["low"] and c2["close"] > c2["open"]:
                    active_fvgs.append({"type": "BULL", "high": curr["low"], "low": c1["high"], "mitigations": 0})
                elif c1["low"] > curr["high"] and c2["close"] < c2["open"]:
                    active_fvgs.append({"type": "BEAR", "high": c1["low"], "low": curr["high"], "mitigations": 0})

            # 4. Detect New Order Blocks (Major vs Internal) with Volume Strength calculation
            is_pivot = c1.get("pivot_high", False) or c1.get("pivot_low", False)
            ob_tier = "MAJOR" if is_pivot else "INTERNAL"

            vol_sma = c2.get("vol_sma_20", 1)
            vol_strength = round((c2["volume"] / vol_sma), 2) if vol_sma > 0 else 1.0

            # Require actual institutional displacement
            required_vol = 1.2 if ob_tier == "MAJOR" else 1.0

            if vol_strength >= required_vol:
                if c2["close"] > c2["open"] and c1["close"] < c1["open"] and c2["close"] > c1["high"]:
                    active_obs.append(
                        {
                            "type": "BULL",
                            "high": c1["high"],
                            "low": c1["low"],
                            "tier": ob_tier,
                            "vol_strength": vol_strength,
                            "mitigations": 0,
                        }
                    )
                elif c2["close"] < c2["open"] and c1["close"] > c1["open"] and c2["close"] < c1["low"]:
                    active_obs.append(
                        {
                            "type": "BEAR",
                            "high": c1["high"],
                            "low": c1["low"],
                            "tier": ob_tier,
                            "vol_strength": vol_strength,
                            "mitigations": 0,
                        }
                    )

        return active_fvgs, active_ifvgs, active_obs

    @staticmethod
    def get_htf_trend(df: pd.DataFrame) -> float:
        """
        Determines HTF trend dynamically using the 50/200 EMA Cross.
        """
        if df.empty:
            return 0.0

        # Read the preserved historical state mapped to this exact candle
        if "htf_trend" in df.columns:
            return float(df.iloc[-1]["htf_trend"])

        # Live Fallback
        ema_50 = df["close"].ewm(span=50, adjust=False).mean()
        ema_200 = df["close"].ewm(span=200, adjust=False).mean()

        if ema_50.iloc[-1] > ema_200.iloc[-1]:
            return 1.0
        elif ema_50.iloc[-1] < ema_200.iloc[-1]:
            return -1.0

        return 0.0
