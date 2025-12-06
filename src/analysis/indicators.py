import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    """
    Advanced technical analysis engine.
    Now includes VWAP, ADX, and Volume Profile approximations.
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies technical indicators to the provided DataFrame.
        """
        # --- Trend: EMAs ---
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # --- Momentum: RSI ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # --- Trend Strength: ADX (Average Directional Index) ---
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr_14)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['adx'] = dx.rolling(window=14).mean()

        # --- Volatility: Bollinger Bands ---
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']

        # --- Volume Analysis ---
        # Volume SMA
        df['vol_sma'] = df['volume'].rolling(window=20).mean()

        # Delta
        df['delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        df['cum_delta'] = df['delta'].rolling(window=20).sum()

        # --- Institutional: VWAP ---
        v = df['volume']
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * v).rolling(window=96).sum() / v.rolling(window=96).sum() # ~24h on 15m candles

        # VWAP Bands
        vwap_std = df['close'].rolling(window=96).std()
        df['vwap_upper'] = df['vwap'] + (vwap_std * 2)
        df['vwap_lower'] = df['vwap'] - (vwap_std * 2)

        # Volume Profile VAH
        df['vol_profile_vah'] = df['high'].rolling(window=50).max() # Simplified resistance zone

        # --- Risk: ATR ---
        df['atr'] = atr_14

        return df.fillna(0)