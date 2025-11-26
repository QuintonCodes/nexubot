import pandas as pd

class TechnicalAnalyzer:
    """
    Professional technical analysis engine using Pandas.
    Calculates EMA, RSI, MACD, Bollinger Bands, and ATR.
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies technical indicators to the provided DataFrame.

        Args:
            df (pd.DataFrame): Data with columns ['close', 'high', 'low', 'volume']

        Returns:
            pd.DataFrame: The original DF with added indicator columns.
        """
        # Trend EMAs (Exponential Moving Averages)
        # Used to determine market direction (Uptrend vs Downtrend)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # RSI (Relative Strength Index) - 14 periods
        # Measures momentum (Overbought > 70 / Oversold < 30)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        # Trend-following momentum indicator
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']

        # Bollinger Bands (20, 2)
        # Measures volatility and relative price levels
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)

        # Band Width (for Squeeze detection)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']

        # ATR (Average True Range)
        # Critical for Stop Loss calculation based on volatility
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - prev_close)
        tr3 = abs(df['low'] - prev_close)
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Volume SMA
        # Used to detect volume spikes relative to average
        df['vol_sma'] = df['volume'].rolling(window=20).mean()

        return df.fillna(0)