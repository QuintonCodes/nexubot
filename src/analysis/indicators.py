import pandas as pd


class TechnicalAnalyzer:
    """
    Advanced technical analysis engine.
    Optimized for performance with optional heavy indicator calculation.
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, heavy: bool = True) -> pd.DataFrame:
        """
        Applies technical indicators to the provided DataFrame.
        :param heavy: If False, skips expensive calculations (Ichimoku, BB, VWAP)
        """
        # --- Trend: EMAs (Essential) ---
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        # --- Momentum: RSI (Essential) ---
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # --- Volatility: ATR (Essential) ---
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # --- Volume (Essential) ---
        df["vol_sma"] = df["volume"].rolling(window=20).mean()

        if not heavy:
            return df.fillna(0)

        # --- EXPENSIVE INDICATORS (Optional) ---

        # --- MACD ---
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- ADX (Average Directional Index) ---
        plus_dm = df["high"].diff()
        minus_dm = df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14).mean() / df["atr"])
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14).mean() / df["atr"])
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df["adx"] = dx.rolling(window=14).mean()

        # --- Ichimoku Cloud ---
        high_9 = df["high"].rolling(window=9).max()
        low_9 = df["low"].rolling(window=9).min()
        df["tenkan_sen"] = (high_9 + low_9) / 2

        high_26 = df["high"].rolling(window=26).max()
        low_26 = df["low"].rolling(window=26).min()
        df["kijun_sen"] = (high_26 + low_26) / 2

        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)
        high_52 = df["high"].rolling(window=52).max()
        low_52 = df["low"].rolling(window=52).min()
        df["senkou_span_b"] = ((high_52 + low_52) / 2).shift(26)

        # --- Bollinger Bands ---
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["std20"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["sma20"] + (df["std20"] * 2)
        df["bb_lower"] = df["sma20"] - (df["std20"] * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma20"]

        # --- VWAP & Volume Profile Approximation ---
        if "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], unit="s")

        df["pv"] = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
        df["date_group"] = df["datetime"].dt.date
        df["cum_pv"] = df.groupby("date_group")["pv"].cumsum()
        df["cum_vol"] = df.groupby("date_group")["volume"].cumsum()
        df["vwap"] = df["cum_pv"] / df["cum_vol"]

        return df.fillna(0)
