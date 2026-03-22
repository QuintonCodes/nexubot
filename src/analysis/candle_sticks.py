import pandas as pd


class CandleStickDetector:
    """
    Detects valid confirmation candles for SMC Execution.
    """

    @staticmethod
    def calculate_candles(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates candle features and identifies patterns like Doji, Engulfing, and Pin Bars."""
        if len(df) < 2:
            return df

        df["body"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["range"] = df["high"] - df["low"]

        prev_body = df["body"].shift(1)
        prev_open = df["open"].shift(1)
        prev_close = df["close"].shift(1)

        # Doji
        df["doji"] = df["body"] <= (df["range"] * 0.1)

        # Bullish Engulfing
        df["bull_engulfing"] = (
            (df["close"] > df["open"])
            & (prev_close < prev_open)
            & (df["close"] > prev_open)
            & (df["open"] < prev_close)
        )

        # Bearish Engulfing
        df["bear_engulfing"] = (
            (df["close"] < df["open"])
            & (prev_close > prev_open)
            & (df["close"] < prev_open)
            & (df["open"] > prev_close)
        )

        # Bullish Pin Bar
        df["bull_pin"] = (df["lower_wick"] > (df["body"] * 1.5)) & (df["upper_wick"] < df["body"])

        # Bearish Pin Bar
        df["bear_pin"] = (df["upper_wick"] > (df["body"] * 1.5)) & (df["lower_wick"] < df["body"])

        return df
