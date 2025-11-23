import pandas as pd

class TechnicalAnalyzer:
    """Professional technical analysis engine using Pandas"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators for Strategy & Risk Management.
        """
        # 1. Trend EMAs (Exponential Moving Averages)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # 2. RSI (Relative Strength Index) - 14 periods
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 3. MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']

        # 4. Bollinger Bands (20, 2)
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        # Band Width (for Squeeze detection)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']

        # 5. ATR (Average True Range) - CRITICAL FOR SL/TP
        # TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - prev_close)
        tr3 = abs(df['low'] - prev_close)
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # 6. Volume Moving Average
        df['vol_sma'] = df['volume'].rolling(window=20).mean()

        return df.fillna(0)