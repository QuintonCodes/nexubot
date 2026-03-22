import pandas as pd
from typing import Literal


class TechnicalAnalyzer:
    """
    Advanced technical analysis engine.
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, heavy: bool = True) -> pd.DataFrame:
        """
        Applies technical indicators to the provided DataFrame.
        """
        # Ensure DateTime column exists for Time-based indicators
        if "datetime" not in df.columns:
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            else:
                # Fallback if no time column, though unusual
                df["datetime"] = pd.Timestamp.now()

        # Trend & Volatility (Essential)
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # ADX (HTF Trend Filter)
        plus_dm = df["high"].diff()
        minus_dm = df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14).mean() / df["atr"])
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14).mean() / df["atr"])
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df["adx"] = dx.rolling(window=14).mean()

        # VWAP
        df["pv"] = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
        df["date_group"] = df["datetime"].dt.date
        df["cum_pv"] = df.groupby("date_group")["pv"].cumsum()
        df["cum_vol"] = df.groupby("date_group")["volume"].cumsum()
        df["vwap"] = df["cum_pv"] / df["cum_vol"]

        if not heavy:
            return df.fillna(0)

        # Bollinger Bands & Keltner Channels (Squeeze Logic)
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["std20"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["sma20"] + (df["std20"] * 2)
        df["bb_lower"] = df["sma20"] - (df["std20"] * 2)

        df["keltner_upper"] = df["ema_50"] + (df["atr"] * 1.5)
        df["keltner_lower"] = df["ema_50"] - (df["atr"] * 1.5)

        # Squeeze Detection: BB inside Keltner
        df["squeeze_on"] = (df["bb_upper"] < df["keltner_upper"]) & (df["bb_lower"] > df["keltner_lower"])

        # Pivot Tracking
        df["pivot_high"] = df["high"] == df["high"].rolling(10, center=True).max()
        df["pivot_low"] = df["low"] == df["low"].rolling(10, center=True).min()

        return df.fillna(0)

    @staticmethod
    def get_htf_trend(df: pd.DataFrame) -> Literal["BULL", "BEAR", "FLAT"]:
        """Determines the higher timeframe trend based on the relationship between price and EMA 200."""
        if df.empty or len(df) < 5:
            return "FLAT"

        price = df["close"].iloc[-1]
        ema_200 = df["ema_200"].iloc[-1]

        if price > ema_200:
            return "BULL"
        elif price < ema_200:
            return "BEAR"

        return "FLAT"

    @staticmethod
    def detect_structure(df: pd.DataFrame) -> dict:
        """
        Detects Break of Structure (BOS) and Change of Character (CHoCH)
        using recent confirmed pivot highs and lows.
        """
        # Need enough data to find multiple pivots
        if len(df) < 20:
            return {"bos": None, "choch": None, "structure": "FLAT"}

        # Exclude the last 5 candles. Since rolling(10, center=True) looks ahead 5 bars,
        # the most recent 5 candles cannot have confirmed pivots yet.
        confirmed_df = df.iloc[:-5]

        # Extract actual pivot prices
        highs = confirmed_df[confirmed_df["pivot_high"]]["high"].values
        lows = confirmed_df[confirmed_df["pivot_low"]]["low"].values

        if len(highs) < 2 or len(lows) < 2:
            return {"bos": None, "choch": None, "structure": "FLAT"}

        # Get the last two confirmed swing points
        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]

        # Get the current live price to check for breaks
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
