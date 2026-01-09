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

        # 4. Strategy: Fair Value Gap (3-Candle + Limit)
        res = self._fx_fvg_entry(curr, df)
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

        # 4. Strategy: VWAP Rejection (Session Reset)
        res = self._crypto_vwap_rejection(curr)
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
        # Explicitly use Standard EMA for Pullback (Lag is desired here)
        ema_50 = curr["ema_50"]
        ema_200 = curr["ema_200"]

        # Long: HTF Bullish + Price > EMA200 + Pullback to EMA50
        if htf_trend == "BULL" and curr["close"] > ema_200:
            dist_ema50 = abs(curr["low"] - ema_50)
            if curr["rsi"] > 80:
                return None

            # Require pullback not to be a tiny bounce nor a deep breakdown
            max_pullback = curr["atr"] * 3.0
            if dist_ema50 > max_pullback:
                return None

            entry_price = ema_50

            if entry_price < curr["close"]:
                sl_price = entry_price - (curr["atr"] * 1.0)

                return {
                    "strategy": "FX Golden Pullback",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": 75.0,
                    "order_type": "LIMIT",
                    "price": entry_price,
                    "suggested_sl": sl_price,
                }

        # Short: HTF Bearish + Price < EMA200 + Pullback to EMA50
        if htf_trend == "BEAR" and curr["close"] < ema_200:
            # distance from the high to EMA50
            dist_ema50 = abs(curr["high"] - ema_50)
            if curr["rsi"] < 20:
                return None

            max_pullback = curr["atr"] * 3.0
            if dist_ema50 > max_pullback:
                return None

            entry_price = ema_50

            if entry_price > curr["close"]:
                sl_price = entry_price + (curr["atr"] * 1.0)
                return {
                    "strategy": "FX Golden Pullback",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 75.0,
                    "order_type": "LIMIT",
                    "price": entry_price,
                    "suggested_sl": sl_price,
                }
        return None

    def _fx_bb_reversion(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 2: Bollinger Mean Reversion
        Logic: Fade moves at the edges of a RANGING market.
        """
        # Bollinger width filter
        bb_width = curr["bb_upper"] - curr["bb_lower"]
        if bb_width > (curr["atr"] * 4.0):
            return None

        # Buy: Price touches Lower BB and closes back inside, with wick/volume confirmation
        if curr["low"] <= curr["bb_lower"] and curr["close"] > curr["bb_lower"]:
            vol_confirm = curr["volume"] > curr["vol_sma"]
            suggested_sl = curr["low"] - (curr["atr"] * 0.5)

            return {
                "strategy": "FX BB Reversion",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": 70.0 + (10.0 if vol_confirm else 0.0),
                "order_type": "LIMIT",
                "price": curr["bb_upper"],
                "suggested_sl": suggested_sl,
            }

        # Sell: Price touches Upper BB and closes back inside
        if curr["high"] >= curr["bb_upper"] and curr["close"] < curr["bb_upper"]:
            vol_confirm = curr["volume"] > curr["vol_sma"]
            suggested_sl = curr["high"] + (curr["atr"] * 0.5)
            return {
                "strategy": "FX BB Reversion",
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": 70.0 + (10.0 if vol_confirm else 0.0),
                "order_type": "LIMIT",
                "price": curr["bb_upper"],
                "suggested_sl": suggested_sl,
            }

        return None

    def _fx_fvg_entry(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """
        Strategy 3: Fair Value Gap (FVG)
        Detects 3-candle imbalance and places LIMIT order at gap start.
        """
        if len(df) < 3:
            return None

        c1 = df.iloc[-3]  # The candle before the move
        c2 = df.iloc[-2]  # Displacement
        c3 = df.iloc[-1]  # Current

        # Displacement Check: Middle candle body > ATR
        c2_body = abs(c2["close"] - c2["open"])
        if c2_body < (curr["atr"] * 0.7):
            return None

        # Displacement should have higher-than-average volume (if volume available)
        low_vol = c2["volume"] < curr["vol_sma"]
        min_gap = curr["atr"] * 0.3

        # Bullish FVG (Gap between C1 High and C3 Low)
        if c2["close"] > c2["open"]:  # Green displacement
            if c1["high"] < c3["low"]:
                gap_size = c3["low"] - c1["high"]
                if gap_size > min_gap:
                    # ensure gap not already filled or price moved beyond the gap start
                    if c3["close"] >= c1["high"]:
                        return None

                    # optional trend alignment: prefer longs above EMA50/EMA200 if present
                    confidence = 85.0 if (curr["close"] > curr["ema_50"] and curr["ema_50"] > curr["ema_200"]) else 70.0
                    if low_vol:
                        confidence -= 10.0

                    entry_price = c1["high"] - (curr["atr"] * 0.02)  # slight buffer inside gap
                    suggested_sl = c3["low"] - (curr["atr"] * 0.5)
                    return {
                        "strategy": "SMC FVG",
                        "signal": "BUY",
                        "direction": "LONG",
                        "confidence": confidence,
                        "order_type": "LIMIT",
                        "price": entry_price,
                        "suggested_sl": suggested_sl,
                    }

        # Bearish FVG (Gap between C1 Low and C3 High)
        if c2["close"] < c2["open"]:  # Red displacement
            if c1["low"] > c3["high"]:
                gap_size = c1["low"] - c3["high"]
                if gap_size > min_gap:
                    if c3["close"] <= c1["low"]:
                        return None

                    confidence = 85.0 if (curr["close"] < curr["ema_50"] and curr["ema_50"] < curr["ema_200"]) else 70.0
                    if low_vol:
                        confidence -= 10.0

                    entry_price = c1["low"] + (curr["atr"] * 0.02)
                    suggested_sl = c3["high"] + (curr["atr"] * 0.5)
                    return {
                        "strategy": "SMC FVG",
                        "signal": "SELL",
                        "direction": "SHORT",
                        "confidence": confidence,
                        "order_type": "LIMIT",
                        "price": entry_price,
                        "suggested_sl": suggested_sl,
                    }
        return None

    def _fx_volatility_breakout(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """
        Strategy 4: Volatility/Inside Bar Breakout
        Logic: Low volatility (compression) followed by a strong move.
        """
        if len(df) < 3:
            return None

        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        # multi-bar compression: average prior range vs ATR
        prev_range_avg = ((prev["high"] - prev["low"]) + (prev2["high"] - prev2["low"])) / 2.0
        if prev_range_avg >= (curr["atr"] * 0.85):
            return None

        # breakout candle must be relatively strong vs prior compression or vs ATR
        breakout_range = curr["high"] - curr["low"]
        strong_breakout = breakout_range > max(prev_range_avg * 1.1, curr["atr"] * 0.8)
        if not strong_breakout:
            return None

        # small buffer to avoid noise
        buffer = curr["atr"] * 0.08

        # volume confirmation (optional)
        vol_confirm = curr["volume"] > curr["vol_sma"]

        # Buy breakout: close above previous high
        if curr["close"] > prev["high"]:
            # suggested SL: below the compression low (safer) or breakout candle low
            compression_low = min(prev2["low"], prev["low"])
            suggested_sl = min(compression_low, curr["low"]) - (curr["atr"] * 0.6)

            # confidence scaled by breakout strength and volume
            confidence = 75.0 if breakout_range > (curr["atr"] * 1.5) else 70.0
            return {
                "strategy": "FX Vol Breakout (Stop)",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": confidence + (5.0 if vol_confirm else 0.0),
                "order_type": "STOP",
                "price": prev["high"] + buffer,
                "suggested_sl": suggested_sl,
            }

        # Sell breakout: close below previous low
        if curr["close"] < prev["low"]:
            compression_high = max(prev2["high"], prev["high"])
            suggested_sl = max(compression_high, curr["high"]) + (curr["atr"] * 0.6)

            confidence = 75.0 if breakout_range > (curr["atr"] * 1.5) else 70.0
            return {
                "strategy": "FX Vol Breakout (Stop)",
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": confidence + (5.0 if vol_confirm else 0.0),
                "order_type": "STOP",
                "price": prev["low"] - buffer,
                "suggested_sl": suggested_sl,
            }

        return None

    # ==========================================
    # CRYPTO STRATEGIES (5 Solid Implementations)
    # ==========================================

    def _crypto_ichimoku_breakout(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 1: Ichimoku Kumo Breakout
        Logic: The most reputable trend strategy for Crypto.
        """
        # basic cloud checks
        span_a = curr.get("senkou_span_a")
        span_b = curr.get("senkou_span_b")
        tenkan = curr.get("tenkan_sen")
        kijun = curr.get("kijun_sen")

        if span_a is None or span_b is None or tenkan is None or kijun is None:
            return None

        # Price is above the Cloud (Span A and Span B)
        above_cloud = curr["close"] > span_a and curr["close"] > span_b
        below_cloud = curr["close"] < span_a and curr["close"] < span_b

        # TK Cross Confirmation (Tenkan > Kijun for Buy)
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun

        # cloud thickness helps gauge strength
        cloud_thickness = abs(span_a - span_b)
        is_extended = abs(curr["close"] - kijun) > (curr["atr"] * 3.0)

        # optional chikou confirmation (if present)
        chikou_ok = True
        if "chikou_span" in curr:
            chikou_ok = curr["chikou_span"] > curr["close"] if above_cloud else curr["chikou_span"] < curr["close"]

        # volume confirmation optional
        vol_bonus = 5.0 if curr["volume"] > curr["vol_sma"] else 0.0

        if above_cloud and tk_bullish and not is_extended and chikou_ok:
            # entry slightly above Tenkan or at Tenkan as limit
            entry_price = min(tenkan, curr["close"] * 0.995)
            suggested_sl = kijun - (curr["atr"] * 0.75)
            confidence = 75.0 + (min(cloud_thickness / (curr["atr"] + 1e-9) * 5.0, 10.0)) + vol_bonus
            return {
                "strategy": "Crypto Ichimoku",
                "signal": "BUY",
                "direction": "LONG",
                "confidence": confidence,
                "order_type": "LIMIT",
                "price": entry_price,
                "suggested_sl": suggested_sl,
            }

        if below_cloud and tk_bearish and not is_extended and chikou_ok:
            entry_price = max(tenkan, curr["close"] * 1.005)
            suggested_sl = kijun + (curr["atr"] * 0.75)
            confidence = 75.0 + (min(cloud_thickness / (curr["atr"] + 1e-9) * 5.0, 10.0)) + vol_bonus
            return {
                "strategy": "Crypto Ichimoku",
                "signal": "SELL",
                "direction": "SHORT",
                "confidence": confidence,
                "order_type": "LIMIT",
                "price": entry_price,
                "suggested_sl": suggested_sl,
            }

        return None

    def _crypto_ema_trend_flow(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 2: EMA Triple Alignment
        Logic: Captures strong momentum when 9 > 50 > 200.
        """
        # Explicitly use ZLEMA for Momentum/Trend Flow
        e9 = curr["zlema_9"]
        e50 = curr["zlema_50"]
        e200 = curr["ema_200"]

        # no trade if price is too extended from e9
        dist = abs(curr["close"] - e9)
        if dist > (curr["atr"] * 3.0):
            return None

        vol_bonus = 5.0 if curr["volume"] > curr["vol_sma"] else 0.0

        # Bullish Alignment (ZLEMA 9 > ZLEMA 50 > EMA 200)
        if e9 > e50 and e50 > e200:
            entry_price = e9
            if entry_price < curr["close"]:
                confidence = 80.0 + vol_bonus
                # slightly penalize if price already far above e9
                if dist > (curr["atr"] * 1.0):
                    confidence -= 8.0
                return {
                    "strategy": "Crypto EMA Flow",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": confidence,
                    "order_type": "LIMIT",
                    "price": entry_price,
                    "suggested_sl": e50 - (curr["atr"] * 0.6),
                }

        # Bearish Alignment
        if e9 < e50 and e50 < e200:
            entry_price = e9
            if entry_price > curr["close"]:
                confidence = 80.0 + vol_bonus
                if dist > (curr["atr"] * 1.0):
                    confidence -= 8.0
                return {
                    "strategy": "Crypto EMA Flow",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": confidence,
                    "order_type": "LIMIT",
                    "price": entry_price,
                    "suggested_sl": e50 + (curr["atr"] * 0.6),
                }

        return None

    def _crypto_vwap_rejection(self, curr: pd.Series) -> Optional[Dict]:
        """
        Strategy 3: Session VWAP Rejection.
        Triggers ONLY when price touches the VWAP and rejects with a large wick.
        """
        if "vwap" not in curr:
            return None

        vwap = curr["vwap"]
        prox_threshold = curr["atr"] * 0.3
        body = abs(curr["close"] - curr["open"])
        vol_bonus = 6.0 if curr["volume"] > curr["vol_sma"] else 0.0

        # Bullish rejection: dipped to/through VWAP and closed above it with a long lower wick
        if curr["close"] > vwap and curr["low"] <= (vwap + prox_threshold):
            lower_wick = min(curr["close"], curr["open"]) - curr["low"]
            if lower_wick > (body * 0.8):
                if lower_wick > body:
                    suggested_sl = min(curr["low"], vwap - (curr["atr"] * 0.3))
                    return {
                        "strategy": "VWAP Rejection",
                        "signal": "BUY",
                        "direction": "LONG",
                        "confidence": 80.0 + vol_bonus,
                        "order_type": "MARKET",
                        "suggested_sl": suggested_sl,
                    }

        # Bearish rejection
        elif curr["close"] < vwap and curr["high"] >= (vwap - prox_threshold):
            upper_wick = curr["high"] - max(curr["close"], curr["open"])
            if upper_wick > (body * 0.8):
                suggested_sl = max(curr["high"], vwap + (curr["atr"] * 0.3))
                return {
                    "strategy": "VWAP Rejection",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": 80.0 + vol_bonus,
                    "order_type": "MARKET",
                    "suggested_sl": suggested_sl,
                }
        return None

    def _crypto_liquidity_grab(self, curr: pd.Series, df: pd.DataFrame) -> Optional[Dict]:
        """
        Strategy 4: Liquidity Grab / Turtle Soup
        This strategy relies on a wick, so we must enter IMMEDIATELY on the close.
        """
        if len(df) < 2:
            return None
        prev = df.iloc[-2]

        # Bullish Liquidity Grab (Fake Breakdown)
        if curr["low"] < prev["low"] and curr["close"] > prev["low"]:
            # Rejection Wick Check: Long lower wick
            body = abs(curr["close"] - curr["open"])
            lower_wick = min(curr["close"], curr["open"]) - curr["low"]
            prev_range = prev["high"] - prev["low"]
            # wick should be a sizeable move beyond prior low and larger than a fraction of prev range
            if lower_wick > (body * 0.7) and lower_wick > (prev_range * 0.25):
                vol_ok = "volume" in curr and "vol_sma" in curr and curr["volume"] > curr["vol_sma"]
                confidence = 75.0 + (7.0 if vol_ok else 0.0)
                return {
                    "strategy": "Crypto Liq Grab",
                    "signal": "BUY",
                    "direction": "LONG",
                    "confidence": confidence,
                    "order_type": "MARKET",
                    "suggested_sl": curr["low"] - (curr["atr"] * 0.6),
                }

        # Bearish Liquidity Grab (Fake Breakout)
        if curr["high"] > prev["high"] and curr["close"] < prev["high"]:
            body = abs(curr["close"] - curr["open"])
            upper_wick = curr["high"] - max(curr["close"], curr["open"])
            prev_range = prev["high"] - prev["low"]
            if upper_wick > (body * 0.7) and upper_wick > (prev_range * 0.25):
                vol_ok = "volume" in curr and "vol_sma" in curr and curr["volume"] > curr["vol_sma"]
                confidence = 75.0 + (7.0 if vol_ok else 0.0)
                return {
                    "strategy": "Crypto Liq Grab",
                    "signal": "SELL",
                    "direction": "SHORT",
                    "confidence": confidence,
                    "order_type": "MARKET",
                    "suggested_sl": curr["high"] + (curr["atr"] * 0.6),
                }

        return None
