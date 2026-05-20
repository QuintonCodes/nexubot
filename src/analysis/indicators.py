import pandas as pd
from typing import Dict, Literal, Tuple


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

        # 4. Internal Pivot Tracking (For local structure and liquidity pools)
        df["pivot_high"] = df["high"] == df["high"].rolling(10, center=True).max()
        df["pivot_low"] = df["low"] == df["low"].rolling(10, center=True).min()

        return df.fillna(0)

    @staticmethod
    def detect_liquidity_sweeps(curr: dict, df: pd.DataFrame, structure: dict, daily_levels: dict) -> int:
        """
        Unified Liquidity Sweep Logic.
        Returns: 3 (Daily Sweep), 2 (Major 50p Sweep), 1 (Internal Sweep), 0 (None)
        """
        pdl = daily_levels.get("pdl")
        pdh = daily_levels.get("pdh")
        last_low = structure.get("last_low")
        last_high = structure.get("last_high")
        close_price = curr["close"]

        recent_low_5 = curr.get("recent_low_5", df["low"].tail(5).min())
        recent_high_5 = curr.get("recent_high_5", df["high"].tail(5).max())

        # Major 50-period swing highs/lows (offsetting the last 5 to ensure we are grabbing established swings)
        major_low_50 = df["low"].iloc[-55:-5].min() if len(df) >= 55 else None
        major_high_50 = df["high"].iloc[-55:-5].max() if len(df) >= 55 else None

        is_swept = 0

        # Tier 3: Daily Sweeps (Most Significant)
        if pdl and recent_low_5 < pdl and close_price > pdl:
            is_swept = 2
        elif pdh and recent_high_5 > pdh and close_price < pdh:
            is_swept = 2

        # Tier 2: Major 50-Period Sweeps
        elif major_low_50 and recent_low_5 < major_low_50 and close_price > major_low_50:
            is_swept = 2
        elif major_high_50 and recent_high_5 > major_high_50 and close_price < major_high_50:
            is_swept = 2

        # Internal Sweeps (Local Structural Pivots)
        elif last_low and recent_low_5 < last_low and close_price > last_low:
            is_swept = 1
        elif last_high and recent_high_5 > last_high and close_price < last_high:
            is_swept = 1

        return is_swept

    @staticmethod
    def detect_structure(df: pd.DataFrame) -> Dict:
        """
        Detects Break of Structure (BOS) and Change of Character (CHoCH)
        using recent confirmed pivot highs and lows.
        """
        if len(df) < 20:
            return {"bos": None, "choch": None, "structure": "FLAT"}

        # Exclude the last 5 candles.
        confirmed_df = df.iloc[:-5]

        # Extract actual pivot prices
        highs = confirmed_df[confirmed_df["pivot_high"]]["high"].values
        lows = confirmed_df[confirmed_df["pivot_low"]]["low"].values

        if len(highs) < 2 or len(lows) < 2:
            return {"bos": None, "choch": None, "structure": "FLAT"}

        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]
        current_close = df.iloc[-1]["close"]

        # 1. Determine local structure/trend based on previous swings
        is_uptrend = (last_high > prev_high) and (last_low > prev_low)
        is_downtrend = (last_high < prev_high) and (last_low < prev_low)
        structure = "BULL" if is_uptrend else ("BEAR" if is_downtrend else "FLAT")

        bos, choch = None, None

        # 2. Detect BOS and CHoCH on the live edge
        if structure == "BULL":
            if current_close > last_high:
                bos = "BULL"  # Price broke the Higher High -> Continuation
            elif current_close < last_low:
                choch = "BEAR"  # Price broke the Higher Low -> Reversal Character
        elif structure == "BEAR":
            if current_close < last_low:
                bos = "BEAR"  # Price broke the Lower Low -> Continuation
            elif current_close > last_high:
                choch = "BULL"  # Price broke the Lower High -> Reversal Character

        return {"bos": bos, "choch": choch, "structure": structure, "last_high": last_high, "last_low": last_low}

    @staticmethod
    def extract_active_pois(df: pd.DataFrame) -> Tuple[list, list, list]:
        """
        Extracts FVGs, converts mitigated FVGs into IFVGs, and detects Major/Internal OBs.
        """
        active_fvgs, active_ifvgs, active_obs = [], [], []

        if len(df) < 3:
            return active_fvgs, active_ifvgs, active_obs

        records = df.to_dict("records")

        for i in range(2, len(records)):
            c1, c2, curr = records[i - 2], records[i - 1], records[i]
            curr_low, curr_high = curr["low"], curr["high"]

            # 1. Manage FVGs -> Mitigated FVGs become IFVGs (Inverse Fair Value Gaps)
            for f in active_fvgs[:]:
                if f["type"] == "BULL" and curr_low < f["low"]:
                    # Broken Bullish FVG -> Bearish IFVG
                    active_ifvgs.append({"type": "BEAR", "high": f["high"], "low": f["low"]})
                    active_fvgs.remove(f)
                elif f["type"] == "BEAR" and curr_high > f["high"]:
                    # Broken Bearish FVG -> Bullish IFVG
                    active_ifvgs.append({"type": "BULL", "high": f["high"], "low": f["low"]})
                    active_fvgs.remove(f)

            # Invalidate IFVGs once they are fully broken in the opposite direction
            active_ifvgs = [
                i_f
                for i_f in active_ifvgs
                if not (i_f["type"] == "BULL" and curr_low < i_f["low"])
                and not (i_f["type"] == "BEAR" and curr_high > i_f["high"])
            ]

            # Invalidate Order Blocks
            active_obs = [
                o
                for o in active_obs
                if not (o["type"] == "BULL" and curr_low < o["low"])
                and not (o["type"] == "BEAR" and curr_high > o["high"])
            ]

            # 2. Detect new FVG
            if c1["high"] < curr["low"] and c2["close"] > c2["open"]:
                active_fvgs.append({"type": "BULL", "high": curr["low"], "low": c1["high"]})
            elif c1["low"] > curr["high"] and c2["close"] < c2["open"]:
                active_fvgs.append({"type": "BEAR", "high": c1["low"], "low": curr["high"]})

            # 3. Detect New Order Blocks (Major vs Internal)
            # A Major OB aligns with a Pivot, an Internal OB forms during flow.
            is_pivot = c1.get("pivot_high", False) or c1.get("pivot_low", False)
            ob_tier = "MAJOR" if is_pivot else "INTERNAL"

            if c2["close"] > c2["open"] and c1["close"] < c1["open"] and c2["close"] > c1["high"]:
                active_obs.append({"type": "BULL", "high": c1["high"], "low": c1["low"], "tier": ob_tier})
            elif c2["close"] < c2["open"] and c1["close"] > c1["open"] and c2["close"] < c1["low"]:
                active_obs.append({"type": "BEAR", "high": c1["high"], "low": c1["low"], "tier": ob_tier})

        return active_fvgs, active_ifvgs, active_obs

    @staticmethod
    def get_htf_trend(df: pd.DataFrame) -> Literal["BULL", "BEAR", "FLAT"]:
        """
        Determines the Higher Timeframe trend purely using Market Structure.
        Evaluates the sequence of recent pivot highs and lows to determine order flow.
        """
        if df.empty or len(df) < 20:
            return "FLAT"

        # Calculate basic pivots for the HTF context
        ph = df["high"] == df["high"].rolling(5, center=True).max()
        pl = df["low"] == df["low"].rolling(5, center=True).min()

        highs = df[ph]["high"].values
        lows = df[pl]["low"].values

        if len(highs) < 2 or len(lows) < 2:
            return "FLAT"

        # Get the last two confirmed swing points
        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]

        # Structure Logic: HH + HL = Bullish | LH + LL = Bearish
        if last_high > prev_high and last_low > prev_low:
            return "BULL"
        elif last_high < prev_high and last_low < prev_low:
            return "BEAR"

        return "FLAT"  # Consolidation / Choppy Market
