import pandas as pd
from typing import Dict, List, Literal, Optional


class StrategyAnalyzer:
    """
    Modular Strategy Engine.
    """

    def analyze_forex(
        self, curr: pd.Series, df: pd.DataFrame, htf_trend: Literal["BULL", "BEAR", "FLAT"], patterns: List[Dict]
    ) -> Optional[Dict]:
        """
        Forex Strategy Router.
        Prioritizes Structure > Trend > Mean Reversion.
        """
        # Pattern Override (High Priority)
        for p in patterns:
            if p["pattern"] in ["Head & Shoulders", "Double Top", "Double Bottom"]:
                # Filter: Only trade patterns in direction of HTF Trend
                if (htf_trend == "BULL" and p["direction"] == "LONG") or (
                    htf_trend == "BEAR" and p["direction"] == "SHORT"
                ):
                    p["strategy"] = f"FX Pattern {p['pattern']}"
                    return p

        # 2. Strategy: The "Golden Pullback" (Trend Following)
        res = self._fx_golden_pullback(curr, htf_trend)
        if res:
            return res

        # 3. Strategy: Bollinger Mean Reversion (Ranging Markets)
        res = self._fx_bb_reversion(curr)
        if res:
            return res

        # 4. Strategy: MACD Divergence (Reversals)
        res = self._fx_macd_divergence(curr, df)
        if res:
            return res

        # 5. Strategy: Volatility Breakout (London/NY Open Logic)
        res = self._fx_volatility_breakout(curr, df)
        if res:
            return res

        return None

    def analyze_crypto(self, curr: pd.Series, df: pd.DataFrame, patterns: List[Dict]) -> Optional[Dict]:
        """
        Crypto Strategy Router.
        Prioritizes Momentum > Volatility > Trend.
        """
        # 1. Pattern: Flags (High Win-rate in Crypto)
        for p in patterns:
            if p["pattern"] in ["Bull Flag", "Bear Flag"]:
                p["strategy"] = "Crypto Flag Breakout"
                return p

        # 2. Strategy: Ichimoku Cloud Breakout (Pure Momentum)
        res = self._crypto_ichimoku_breakout(curr)
        if res:
            return res

        # 3. Strategy: EMA Trend Flow (The "Wave")
        res = self._crypto_ema_trend_flow(curr)
        if res:
            return res

        # 4. Strategy: Volume Squeeze (Explosive Moves)
        res = self._crypto_vol_squeeze(curr)
        if res:
            return res

        # 5. Strategy: Liquidity Grab (V-Shape Reversal)
        res = self._crypto_liquidity_grab(curr, df)
        if res:
            return res

        return None

    # ==========================================
    # FOREX STRATEGIES (5 Solid Implementations)
    # ==========================================

    def _fx_golden_pullback(self, curr: pd.Series, htf_trend: Literal["BULL", "BEAR", "FLAT"]) -> Optional[Dict]:
        """
        Strategy 1: The Golden Pullback
        Logic: Buy dips in strong trends.
        """
        # Check Trend Strength using ADX
        if curr["adx"] < 25:
            return None

        # Long: HTF Bullish + Price > EMA200 + Pullback to EMA50
        if htf_trend == "BULL" and curr["close"] > curr["ema_200"]:
            # Price is near EMA50 (within 0.5 ATR)
            dist_ema50 = abs(curr["low"] - curr["ema_50"])
            if dist_ema50 <= (curr["atr"] * 0.5):
                # Trigger: RSI is not overbought (room to grow)
                if curr["rsi"] < 60:
                    return {"strategy": "FX Golden Pullback", "signal": "BUY", "direction": "LONG", "confidence": 85.0}

        # Short: HTF Bearish + Price < EMA200 + Pullback to EMA50
        if htf_trend == "BEAR" and curr["close"] < curr["ema_200"]:
            dist_ema50 = abs(curr["high"] - curr["ema_50"])
            if dist_ema50 <= (curr["atr"] * 0.5):
                if curr["rsi"] > 40:
                    return {
                        "strategy": "FX Golden Pullback",
                        "signal": "SELL",
                        "direction": "SHORT",
                        "confidence": 85.0,
                    }
        return None

    def _fx_bb_reversion(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 2: Bollinger Mean Reversion
        Logic: Fade moves at the edges of a RANGING market.
        """
        # Essential: ADX must be LOW (< 25) to confirm a Range
        if curr["adx"] > 25:
            return None

        # Buy: Price touches Lower BB + RSI Oversold (< 30)
        if curr["low"] <= curr["bb_lower"] and curr["rsi"] < 30:
            return {"strategy": "FX BB Reversion", "signal": "BUY", "direction": "LONG", "confidence": 80.0}

        # Sell: Price touches Upper BB + RSI Overbought (> 70)
        if curr["high"] >= curr["bb_upper"] and curr["rsi"] > 70:
            return {"strategy": "FX BB Reversion", "signal": "SELL", "direction": "SHORT", "confidence": 80.0}

        return None

    def _fx_macd_divergence(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """
        Strategy 3: MACD Divergence
        Logic: Price makes New Low, MACD makes Higher Low (Reversal).
        """
        prev = df.iloc[-2]

        # Bullish Divergence
        # Price: Lower Low, MACD: Higher Low
        if curr["low"] < prev["low"] and curr["macd"] > prev["macd"]:
            # Confirm with Histogram ticking up
            if curr["macd_hist"] > prev["macd_hist"] and curr["macd_hist"] > 0:
                return {"strategy": "FX MACD Div", "signal": "BUY", "direction": "LONG", "confidence": 88.0}

        # Bearish Divergence
        # Price: Higher High, MACD: Lower High
        if curr["high"] > prev["high"] and curr["macd"] < prev["macd"]:
            if curr["macd_hist"] < prev["macd_hist"] and curr["macd_hist"] < 0:
                return {"strategy": "FX MACD Div", "signal": "SELL", "direction": "SHORT", "confidence": 88.0}

        return None

    def _fx_volatility_breakout(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """
        Strategy 4: Volatility/Inside Bar Breakout
        Logic: Low volatility (compression) followed by a strong move.
        """
        prev = df.iloc[-2]

        # 1. Compression: Previous candle range was very small (< 0.5 ATR)
        prev_range = prev["high"] - prev["low"]
        is_compressed = prev_range < (prev["atr"] * 0.6)

        if not is_compressed:
            return None

        # 2. Expansion: Current candle breaks prev High/Low
        # Long Breakout
        if curr["close"] > prev["high"]:
            if curr["volume"] > curr["vol_sma"]:  # Volume Confirmation
                return {"strategy": "FX Vol Breakout", "signal": "BUY", "direction": "LONG", "confidence": 82.0}

        # Short Breakout
        if curr["close"] < prev["low"]:
            if curr["volume"] > curr["vol_sma"]:
                return {"strategy": "FX Vol Breakout", "signal": "SELL", "direction": "SHORT", "confidence": 82.0}

        return None

    # ==========================================
    # CRYPTO STRATEGIES (5 Solid Implementations)
    # ==========================================

    def _crypto_ichimoku_breakout(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 1: Ichimoku Kumo Breakout
        Logic: The most reputable trend strategy for Crypto.
        """
        # Price is above the Cloud (Span A and Span B)
        above_cloud = curr["close"] > curr["senkou_span_a"] and curr["close"] > curr["senkou_span_b"]
        below_cloud = curr["close"] < curr["senkou_span_a"] and curr["close"] < curr["senkou_span_b"]

        # TK Cross Confirmation (Tenkan > Kijun for Buy)
        tk_bullish = curr["tenkan_sen"] > curr["kijun_sen"]
        tk_bearish = curr["tenkan_sen"] < curr["kijun_sen"]

        dist_kijun = abs(curr["close"] - curr["kijun_sen"])
        is_extended = dist_kijun > (curr["atr"] * 1.5)

        if above_cloud and tk_bullish and not is_extended:
            # Ensure we just broke out or are trending strongly
            return {"strategy": "Crypto Ichimoku", "signal": "BUY", "direction": "LONG", "confidence": 90.0}

        if below_cloud and tk_bearish and not is_extended:
            return {"strategy": "Crypto Ichimoku", "signal": "SELL", "direction": "SHORT", "confidence": 90.0}

        return None

    def _crypto_ema_trend_flow(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 2: EMA Triple Alignment
        Logic: Captures strong momentum when 9 > 50 > 200.
        """
        # Bullish Alignment
        if curr["ema_9"] > curr["ema_50"] and curr["ema_50"] > curr["ema_200"]:
            # Entry Trigger: Price is slightly pulling back to EMA 9 (Dip buy)
            if curr["close"] > curr["ema_9"] and curr["low"] < curr["ema_9"]:
                return {"strategy": "Crypto EMA Flow", "signal": "BUY", "direction": "LONG", "confidence": 85.0}

        # Bearish Alignment
        if curr["ema_9"] < curr["ema_50"] and curr["ema_50"] < curr["ema_200"]:
            if curr["close"] < curr["ema_9"] and curr["high"] > curr["ema_9"]:
                return {"strategy": "Crypto EMA Flow", "signal": "SELL", "direction": "SHORT", "confidence": 85.0}

        return None

    def _crypto_vol_squeeze(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 3: Bollinger Squeeze
        Logic: Periods of low volatility (squeeze) lead to explosive moves.
        """
        # 1. Identify Squeeze (Band width is tight)
        # Note: 0.10 is a generic threshold, relative checking is better but this is solid for code simplicity
        if curr["bb_width"] > 0.12:
            return None

        # 2. Breakout Logic
        if curr["close"] > curr["bb_upper"]:
            # Volume must support the breakout
            if curr["volume"] > curr["vol_sma"]:
                return {"strategy": "Crypto Squeeze", "signal": "BUY", "direction": "LONG", "confidence": 88.0}

        if curr["close"] < curr["bb_lower"]:
            if curr["volume"] > curr["vol_sma"]:
                return {"strategy": "Crypto Squeeze", "signal": "SELL", "direction": "SHORT", "confidence": 88.0}

        return None

    def _crypto_liquidity_grab(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """
        Strategy 4: Liquidity Grab / Turtle Soup
        Logic: Price breaks a recent low/high but closes back inside (Fakeout).
        """
        prev = df.iloc[-2]

        # Bullish Liquidity Grab (Fake Breakdown)
        # Current Low is lower than previous Low, but Close is HIGHER than previous Low
        if curr["low"] < prev["low"] and curr["close"] > prev["low"]:
            # Rejection Wick Check: Long lower wick
            body = abs(curr["close"] - curr["open"])
            lower_wick = min(curr["close"], curr["open"]) - curr["low"]

            if lower_wick > (body * 1.5) and curr["volume"] > curr["vol_sma"]:
                return {"strategy": "Crypto Liq Grab", "signal": "BUY", "direction": "LONG", "confidence": 87.0}

        # Bearish Liquidity Grab (Fake Breakout)
        if curr["high"] > prev["high"] and curr["close"] < prev["high"]:
            body = abs(curr["close"] - curr["open"])
            upper_wick = curr["high"] - max(curr["close"], curr["open"])

            if upper_wick > (body * 1.5) and curr["volume"] > curr["vol_sma"]:
                return {"strategy": "Crypto Liq Grab", "signal": "SELL", "direction": "SHORT", "confidence": 87.0}

        return None
