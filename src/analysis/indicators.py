import pandas as pd
from typing import Dict, Literal


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

        # 1. ATR (Used purely for Stop Loss buffers, not for entry logic)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # 2. Daily VWAP (Institutional Baseline)
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

        bos = None
        choch = None

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
