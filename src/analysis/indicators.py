import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

from src.config import CANDLE_LIMIT


class TechnicalAnalyzer:
    """Centralized class for all technical indicator calculations and structural analysis."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Applies technical indicators to the provided DataFrame."""
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
        tr = np.maximum.reduce([high_low, high_close, low_close])
        df["atr"] = pd.Series(tr).rolling(window=14).mean()
        df["atr_ema_48"] = df["atr"].ewm(span=48, adjust=False).mean()
        df["atr_expansion_ratio"] = df["atr"] / df["atr_ema_48"].replace(0, np.nan)
        df["atr_expansion_ratio"] = df["atr_expansion_ratio"].fillna(1.0)
        df.loc[df.index[:48], "atr_expansion_ratio"] = 1.0

        # 2. Daily VWAP
        df["pv"] = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
        df["date_group"] = df["datetime"].dt.date
        df["cum_pv"] = df.groupby("date_group")["pv"].cumsum()
        df["cum_vol"] = df.groupby("date_group")["volume"].cumsum()
        df["vwap"] = df["cum_pv"] / df["cum_vol"]

        # 3. Daily Open (SMC Premium/Discount Baseline)
        df["daily_open"] = df.groupby("date_group")["open"].transform("first")

        # 4. Internal Pivot Tracking
        df["pivot_high"] = (df["high"] == df["high"].rolling(window=6, min_periods=6).max().shift(-3)).fillna(False)
        df["pivot_low"] = (df["low"] == df["low"].rolling(window=6, min_periods=6).min().shift(-3)).fillna(False)

        # 5. Volume Profile for Institutional Displacement
        df["vol_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
        df["vol_ratio"] = df["vol_ratio"].fillna(1.0)

        # Vol Trend calculated over a stronger 5-bar delta threshold instead of 2-bars
        df["volume_trend_3"] = df["vol_ratio"] - df["vol_ratio"].shift(5).fillna(df["vol_ratio"])

        # 6. Lookback Windows
        df["recent_low_5"] = df["low"].rolling(5).min()
        df["recent_high_5"] = df["high"].rolling(5).max()
        df["recent_low_4"] = df["low"].rolling(4).min()
        df["recent_high_4"] = df["high"].rolling(4).max()
        df["major_low_50"] = df["low"].rolling(50).min().shift(5)
        df["major_high_50"] = df["high"].rolling(50).max().shift(5)

        # 7. HTF Trend (Institutional EMAs)
        close_series = df["close"].astype(float)
        df["ema_50"] = close_series.ewm(span=50, adjust=False).mean()
        df["ema_200"] = close_series.ewm(span=200, adjust=False).mean()

        df["htf_trend"] = 0.0
        df.loc[df["ema_50"] > df["ema_200"], "htf_trend"] = 1.0
        df.loc[df["ema_50"] < df["ema_200"], "htf_trend"] = -1.0

        # Asian Range Tracking (Power of 3)
        df["hour"] = df["datetime"].dt.hour
        asian_mask = (df["hour"] >= 2) & (df["hour"] < 10)
        df["is_asian"] = asian_mask
        asian_highs = df[asian_mask].groupby("date_group")["high"].max()
        asian_lows = df[asian_mask].groupby("date_group")["low"].min()
        df["asian_high"] = df["date_group"].map(asian_highs)
        df["asian_low"] = df["date_group"].map(asian_lows)

        df["color"] = np.where(df["close"] > df["open"], 1, -1)

        # Safely fill price-based columns before global zeroing to prevent SL Cap / VWAP corruption
        df["atr"] = df["atr"].ffill().fillna(df["close"] * 0.0005)
        df["vwap"] = df["vwap"].ffill().fillna(df["close"])
        df["vol_sma_20"] = df["vol_sma_20"].ffill().fillna(1.0)
        df["asian_high"] = df["asian_high"].ffill().fillna(df["high"])
        df["asian_low"] = df["asian_low"].ffill().fillna(df["low"])

        return df

    @staticmethod
    def compile_features(
        curr: Union[pd.Series, dict],
        signal_direction: str,
        active_fvgs: list,
        active_ifvgs: list,
        active_obs: list,
        structure_info: dict,
        is_liquidity_swept: int,
        sweep_depth_atr: float,
        atr: float,
        rr_at_entry: float = 0.0,
    ) -> Dict:
        """Compiles the strict SMC feature set for neural network predictions and training."""
        close_price = curr["close"]
        safe_atr = max(float(atr), 1e-8)

        # Killzone Map
        dt = (
            curr["datetime"]
            if isinstance(curr.get("datetime"), pd.Timestamp)
            else pd.to_datetime(curr.get("time"), unit="s")
        )
        h, m = dt.hour, dt.minute
        decimal_hour = h + (m / 60.0)

        # Distance & Mitigation Tracking
        all_zones = active_fvgs + active_ifvgs + active_obs
        zone_age_bars = 0.0

        if all_zones:
            nearest_poi = min(all_zones, key=lambda x: min(abs(x["high"] - close_price), abs(x["low"] - close_price)))
            zone_age_bars = float(nearest_poi.get("age", 0))

        body = abs(close_price - curr["open"])
        rng = curr["high"] - curr["low"] + 1e-10
        body_ratio = body / rng
        upper_wick_pct = (curr["high"] - max(curr["open"], close_price)) / safe_atr
        lower_wick_pct = (min(curr["open"], close_price) - curr["low"]) / safe_atr

        favor_wick_pct = lower_wick_pct if signal_direction == "LONG" else upper_wick_pct
        adverse_wick_pct = upper_wick_pct if signal_direction == "LONG" else lower_wick_pct

        vol_ratio = curr.get("vol_ratio", 1.0)
        volume_trend_3 = curr.get("volume_trend_3", 0.0)

        pdh, pdl = curr.get("pdh"), curr.get("pdl")
        dist_to_pdh_atr = abs(close_price - pdh) / safe_atr if pdh else 0.0
        dist_to_pdl_atr = abs(close_price - pdl) / safe_atr if pdl else 0.0

        ah, al = curr.get("asian_high"), curr.get("asian_low")
        dist_to_asia_extremes_atr = min(abs(close_price - ah), abs(close_price - al)) / safe_atr if ah and al else 0.0

        vwap_distance_atr = (close_price - curr.get("vwap", close_price)) / safe_atr

        return {
            "pd_array_status": structure_info.get("pd_array", 0.5),
            "dist_to_pdh_atr": dist_to_pdh_atr,
            "dist_to_pdl_atr": dist_to_pdl_atr,
            "dist_to_asia_extremes_atr": dist_to_asia_extremes_atr,
            "vwap_distance_atr": vwap_distance_atr,
            "is_liquidity_swept_tier": float(is_liquidity_swept),
            "sweep_depth_atr": sweep_depth_atr,
            "body_ratio": body_ratio,
            "favor_wick_pct": favor_wick_pct,
            "adverse_wick_pct": adverse_wick_pct,
            "vol_ratio": vol_ratio,
            "volume_trend_3": volume_trend_3,
            "atr_expansion_ratio": float(curr.get("atr_expansion_ratio", 1.0)),
            "rr_at_entry": rr_at_entry,
            "hour_sin": math.sin(2 * math.pi * decimal_hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * decimal_hour / 24.0),
            "structure_age_bars": float(structure_info.get("bars_since_break", 0)),
            "zone_age_bars": zone_age_bars,
        }

    @staticmethod
    def detect_liquidity_sweeps(curr: Union[pd.Series, dict], structure: dict, daily_levels: dict) -> Tuple[int, float]:
        """
        Unified Liquidity Sweep Logic.
        Returns: (Sweep Tier [0-3], Sweep Depth in ATR)
        """
        pdl, pdh = daily_levels.get("pdl"), daily_levels.get("pdh")
        asian_high, asian_low = curr.get("asian_high"), curr.get("asian_low")
        last_low, last_high = structure.get("last_low"), structure.get("last_high")

        close_price = curr["close"]
        recent_low_5, recent_high_5 = curr.get("recent_low_5", 0), curr.get("recent_high_5", 0)
        major_low_50, major_high_50 = curr.get("major_low_50", None), curr.get("major_high_50", None)

        fallback_atr = close_price * 0.0005
        atr = curr.get("atr", fallback_atr)
        if atr == 0 or pd.isna(atr):
            atr = fallback_atr

        is_swept, sweep_depth = 0, 0.0

        # Round Number Pool proximity check
        def is_round_number_sweep(price, high, low):
            if price <= 0:
                return False
            magnitude = 10 ** math.floor(math.log10(price))
            step = magnitude / 10 if magnitude > 10 else 1.0
            closest_round = round(price / step) * step
            return min(abs(high - closest_round), abs(low - closest_round)) < (atr * 0.5)

        # Tier 3: Daily Sweeps & Asian Range Manipulations
        is_london = False
        time_val = curr.get("datetime") or curr.get("time")
        if time_val:
            dt = time_val if isinstance(time_val, pd.Timestamp) else pd.to_datetime(time_val, unit="s")
            is_london = 9 <= dt.hour <= 11

        # Tier 3: Daily Sweeps (Most Significant)
        if pdl and recent_low_5 < pdl and close_price > pdl:
            is_swept, sweep_depth = 3, (pdl - recent_low_5) / atr
        elif pdh and recent_high_5 > pdh and close_price < pdh:
            is_swept, sweep_depth = 3, (recent_high_5 - pdh) / atr
        elif is_london and asian_low and recent_low_5 < asian_low and close_price > asian_low:
            is_swept, sweep_depth = 3, (asian_low - recent_low_5) / atr
        elif is_london and asian_high and recent_high_5 > asian_high and close_price < asian_high:
            is_swept, sweep_depth = 3, (recent_high_5 - asian_high) / atr

        # Tier 2: Major 50-Period Sweeps & Round Number Sweeps
        elif major_low_50 and recent_low_5 < major_low_50 and close_price > major_low_50:
            is_swept, sweep_depth = 2, (major_low_50 - recent_low_5) / atr
        elif major_high_50 and recent_high_5 > major_high_50 and close_price < major_high_50:
            is_swept, sweep_depth = 2, (recent_high_5 - major_high_50) / atr
        elif is_round_number_sweep(close_price, recent_high_5, recent_low_5):
            is_swept, sweep_depth = 2, 0.5

        # Tier 1: Internal Sweeps (Local Structural Pivots)
        elif last_low and recent_low_5 < last_low and close_price > last_low:
            is_swept, sweep_depth = 1, (last_low - recent_low_5) / atr
        elif last_high and recent_high_5 > last_high and close_price < last_high:
            is_swept, sweep_depth = 1, (recent_high_5 - last_high) / atr

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
        confirmed_df = recent_df.iloc[:-1]

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

        bos, choch, structural_break = None, None, 0.0

        # 2. Detect BOS and CHoCH on the live edge (Requires body close)
        if structure == "BULL":
            if current_close > last_high:
                bos, structural_break = "BULL", 1.0
            elif current_close < last_low:
                choch, structural_break = "BEAR", -2.0
        elif structure == "BEAR":
            if current_close < last_low:
                bos, structural_break = "BEAR", -1.0
            elif current_close > last_high:
                choch, structural_break = "BULL", 2.0

        # Premium / Discount Calculation (0.0 to 1.0)
        pd_range = last_high - last_low
        pd_array_status = max(0.0, min(1.0, (current_close - last_low) / pd_range)) if pd_range > 0 else 0.5

        pivots = confirmed_df[confirmed_df["pivot_high"] | confirmed_df["pivot_low"]]
        if not pivots.empty:
            last_pivot_idx_pos = df.index.get_loc(pivots.index[-1])
            bars_since = len(df) - last_pivot_idx_pos - 1
        else:
            bars_since = len(df)

        return {
            "bos": bos,
            "choch": choch,
            "structure": structure,
            "last_high": last_high,
            "last_low": last_low,
            "structural_break": structural_break,
            "pd_array": pd_array_status,
            "bars_since_break": bars_since,
        }

    @staticmethod
    def extract_active_pois(
        data: Union[pd.DataFrame, list], lookback_limit: int = CANDLE_LIMIT
    ) -> Tuple[list, list, list]:
        """Extracts POIs, handles conversions, and tracks their Mitigation Count."""
        active_fvgs, active_ifvgs, active_obs = [], [], []
        records = data.to_dict("records") if isinstance(data, pd.DataFrame) else data

        if len(records) > lookback_limit:
            records = records[-lookback_limit:]

        if len(records) < 3:
            return [], [], []

        for i in range(2, len(records)):
            c1, c2, curr = records[i - 2], records[i - 1], records[i]
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.update_pois(
                c1, c2, curr, active_fvgs, active_ifvgs, active_obs
            )

        return active_fvgs, active_ifvgs, active_obs

    @staticmethod
    def get_htf_trend(df: pd.DataFrame) -> float:
        """Determines HTF trend dynamically using the 50/200 EMA Cross."""
        if df.empty:
            return 0.0

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

    @staticmethod
    def update_pois(
        c1: dict, c2: dict, curr: dict, active_fvgs: list, active_ifvgs: list, active_obs: list
    ) -> Tuple[List, List, List]:
        """Runs an incremental O(1) state update step for arrays mapping."""
        curr_low, curr_high, curr_close = curr["low"], curr["high"], curr["close"]
        surviving_fvgs = []

        for pool in [active_fvgs, active_ifvgs, active_obs]:
            for z in pool:
                z["age"] = z.get("age", 0) + 1

        for f in active_fvgs[:]:
            if f["type"] == "BULL" and curr_low < f["low"]:
                active_ifvgs.append(
                    {
                        "type": "BEAR",
                        "high": f["high"],
                        "low": f["low"],
                        "mitigations": 0,
                        "is_touching": False,
                        "age": 0,
                    }
                )
            elif f["type"] == "BEAR" and curr_high > f["high"]:
                active_ifvgs.append(
                    {
                        "type": "BULL",
                        "high": f["high"],
                        "low": f["low"],
                        "mitigations": 0,
                        "is_touching": False,
                        "age": 0,
                    }
                )
            else:
                surviving_fvgs.append(f)
        active_fvgs = surviving_fvgs

        for pool, pool_name in [(active_fvgs, "fvg"), (active_ifvgs, "ifvg"), (active_obs, "ob")]:
            surviving_zones = []
            for zone in pool[:]:
                is_inside = (zone["type"] == "BULL" and curr_low <= zone["high"]) or (
                    zone["type"] == "BEAR" and curr_high >= zone["low"]
                )
                if is_inside:
                    if not zone.get("is_touching", False):
                        zone["mitigations"] += 1
                        zone["is_touching"] = True
                else:
                    zone["is_touching"] = False

                if zone["mitigations"] > 3:
                    continue

                if pool_name == "ob":
                    if zone["type"] == "BULL" and curr_close < zone["low"]:
                        zone["type"], zone["tier"], zone["mitigations"], zone["age"] = "BEAR", "BREAKER", 0, 0
                        surviving_zones.append(zone)
                        continue
                    if zone["type"] == "BEAR" and curr_close > zone["high"]:
                        zone["type"], zone["tier"], zone["mitigations"], zone["age"] = "BULL", "BREAKER", 0, 0
                        surviving_zones.append(zone)
                        continue
                else:
                    if (zone["type"] == "BULL" and curr_close < zone["low"]) or (
                        zone["type"] == "BEAR" and curr_close > zone["high"]
                    ):
                        continue

                surviving_zones.append(zone)
            pool[:] = surviving_zones

        vol_sma = c2.get("vol_sma_20", 1)
        vol_strength = round((c2["volume"] / vol_sma), 2) if vol_sma > 0 else 1.0
        c2_range = c2["high"] - c2["low"]
        c2_body = abs(c2["close"] - c2["open"])

        if c2_range > 0 and vol_strength >= 1.5 and (c2_body / c2_range) >= 0.70:
            if c1["high"] < curr_low and c2["close"] > c2["open"]:
                active_fvgs.append(
                    {
                        "type": "BULL",
                        "high": curr_low,
                        "low": c1["high"],
                        "mitigations": 0,
                        "is_touching": False,
                        "age": 0,
                    }
                )
            elif c1["low"] > curr_high and c2["close"] < c2["open"]:
                active_fvgs.append(
                    {
                        "type": "BEAR",
                        "high": c1["low"],
                        "low": curr_high,
                        "mitigations": 0,
                        "is_touching": False,
                        "age": 0,
                    }
                )

        is_pivot = c1.get("pivot_high", False) or c1.get("pivot_low", False)
        ob_tier = "MAJOR" if is_pivot else "INTERNAL"
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
                        "is_touching": False,
                        "age": 0,
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
                        "is_touching": False,
                        "age": 0,
                    }
                )

        return active_fvgs, active_ifvgs, active_obs
