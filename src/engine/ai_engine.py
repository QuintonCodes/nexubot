import asyncio
import logging
import math
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Literal, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.candle_sticks import CandleStickDetector
from src.data.collector import DataCollector
from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ml_engine import NeuralPredictor
from src.engine.strategies import StrategyAnalyzer
from src.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MAX_LOT,
    DEFAULT_RISK_PCT,
    HIGH_VOLATILITY_IDENTIFIERS,
    PAIR_SIGNAL_COOLDOWN,
    SESSION_CONFIG,
    get_account_risk_caps,
)

logger = logging.getLogger(__name__)


class AITradingEngine:
    """
    Advanced SMC Intelligence Engine.
    Integrates Strict Multi-Timeframe Analysis, Smart Money Concepts, and Pre-trained Neural Networks.
    """

    def __init__(self):
        self.strategy_analyzer = StrategyAnalyzer()
        self.nn_brain = NeuralPredictor()

        self._log_throttle = {}
        self.active_features = {}
        self.db_manager = None
        self.htf_cache = {}
        self.signal_history = {}
        self.user_balance_account = 0.0
        self.currency = "USD"

        # Dynamic Configuration
        self.risk_pct = DEFAULT_RISK_PCT
        self.max_lot = DEFAULT_MAX_LOT
        self.min_confidence = DEFAULT_MIN_CONFIDENCE
        self.allow_high_volatility = False

    async def _adjust_confidence(
        self,
        symbol: str,
        signal: dict,
        nn_prob: float,
        htf_trend: Literal["BULL", "BEAR", "FLAT"],
        volatility_ratio: float,
    ) -> Dict:
        """
        Calculates realistic confidence score.
        """
        base_conf = min(signal["confidence"], DEFAULT_MIN_CONFIDENCE)

        # 1. MTF Trend Alignment (Strictly enforced by router, but rewarded here)
        trend_bonus = 0
        if htf_trend == "BULL" and signal["direction"] == "LONG":
            trend_bonus = 5
        elif htf_trend == "BEAR" and signal["direction"] == "SHORT":
            trend_bonus = 5

        # 2. Historical Performance
        hist_win_rate = 0.5
        if self.db_manager:
            hist_win_rate = await self.db_manager.get_pair_performance(symbol)

        # Heavy Penalty for losers (< 40% win rate), Small Bonus for winners
        history_factor = -10 if hist_win_rate < 0.4 else (10 if hist_win_rate > 0.6 else 0)

        # 3. Neural Network Weighting
        nn_factor = (nn_prob - 0.5) * 40

        # 4. Volatility Penalty (Squeeze or extreme expansion)
        is_volatile_asset = self._is_high_volatility_symbol(symbol)

        if is_volatile_asset:
            # Gold/Silver/Indices are allowed higher volatility ratios
            vol_penalty = -5 if volatility_ratio > 2.0 else (-10 if volatility_ratio > 3.0 else 0)
        else:
            vol_penalty = -10 if volatility_ratio > 1.5 else (-20 if volatility_ratio > 2.0 else 0)

        # Final Calculation
        final_conf = base_conf + trend_bonus + history_factor + nn_factor + vol_penalty
        final_conf = max(0.0, min(99.0, final_conf))

        signal["confidence"] = final_conf
        return signal

    async def _calculate_risk_metrics(
        self, symbol: str, signal: dict, curr: pd.Series, tick, info: dict, nn_result: dict, provider: DataProvider
    ) -> Optional[Dict]:
        """
        Calculates Lot Size and Risk (USD/ZAR conversion) and validates entry freshness.
        """
        # 1. Fetch Account Currency & Live Rates
        acct_summary = await provider.get_account_summary()
        self.currency = acct_summary.get("currency", "USD")
        self.user_balance_account = acct_summary.get("balance", 0.0)

        if self.user_balance_account <= 0:
            return None

        # Fetch Exchange Rate if needed (Account USD -> Display ZAR)
        usdzar_rate = 1.0
        if self.currency == "USD":
            usdzar_rate = await provider.get_usdzar_rate()

        ask, bid = tick.ask, tick.bid
        point = info["point"]
        tick_value = info.get("trade_tick_value", 0)
        min_vol = info.get("min_vol", 0.01)
        max_vol = info.get("max_vol", 100.0)
        vol_step = info.get("vol_step", 0.01)
        digits = info.get("digits", 5)
        atr = float(curr["atr"])

        # Entry Price Determination
        current_market_price = ask if signal["direction"] == "LONG" else bid
        order_type = signal.get("order_type", "MARKET")
        entry_price = signal.get("price", current_market_price)

        if order_type == "MARKET":
            entry_price = current_market_price
            signal_close_price = curr["close"]
            pct_diff = abs(current_market_price - signal_close_price) / signal_close_price

            # Stricter thresholds
            symbol_type = provider.get_symbol_type(symbol)
            if symbol_type == "CRYPTO":
                max_diff = 0.005  # 0.5% tolerance for Crypto
            elif symbol_type == "INDICES":
                max_diff = 0.003  # 0.3% tolerance for Indices
            else:
                max_diff = 0.001  # 0.1% tolerance for Forex

            if pct_diff > max_diff:
                self._log_once(
                    f"runaway_{symbol}",
                    f"Skipping {symbol}: Price Runaway. Signal: {signal_close_price} vs Now: {current_market_price}",
                )
                return None

        # Dynamic TP / SL calculation
        sl_multiplier = 1.4 if (self._is_high_volatility_symbol(symbol) or atr > (curr["close"] * 0.005)) else 1.0
        sl_dist = max(atr * sl_multiplier, point * 50)

        # Use Dynamic SL if provided
        if "suggested_sl" in signal:
            suggested_dist = abs(signal["suggested_sl"] - entry_price)
            if (atr * 0.3) < suggested_dist < (atr * 5.0):
                sl_dist = suggested_dist

        # Probability calibration hook
        prob = nn_result.get("prob", 0.5)
        pred_exit_atr = max(1, min(float(nn_result.get("pred_exit_atr", 2.0)), 6.0))
        tp_multiplier = max(pred_exit_atr, 3.0) if self.risk_pct > 3.0 else pred_exit_atr
        tp_dist = max(atr * tp_multiplier, sl_dist * 1.2)

        rr = (tp_dist / sl_dist) if sl_dist > 0 else 1.0
        expected_ev = prob * rr - (1.0 - prob)

        # Kelly-informed adjustment (small, capped multiplier)
        kelly = prob - ((1 - prob) / (rr + 1e-9))
        kelly_factor = min(1.5, max(0.5, 1.0 + (kelly * 2.0))) if kelly > 0 else 0.5

        # If EV is clearly negative, reduce risk_mult / skip
        if expected_ev < 0:
            nn_result["risk_mult"] = max(0.25, nn_result.get("risk_mult", 1.0) * 0.5)
            if expected_ev < -0.25:
                self._log_once(f"ev_gate_{symbol}", f"Skipping {symbol}: Negative EV ({expected_ev:.2f})")
                return None

        tp1_dist = tp_dist * 0.33
        tp2_dist = tp_dist * 0.66
        tp3_dist = tp_dist

        # Absolute Prices
        if signal["signal"] == "BUY":
            ref_price = entry_price if order_type == "LIMIT" else ask
            sl_price = ref_price - sl_dist
            tp1_price = ref_price + tp1_dist
            tp2_price = ref_price + tp2_dist
            tp3_price = ref_price + tp3_dist
        else:
            ref_price = entry_price if order_type == "LIMIT" else bid
            sl_price = ref_price + sl_dist
            tp1_price = ref_price - tp1_dist
            tp2_price = ref_price - tp2_dist
            tp3_price = ref_price - tp3_dist

        # Risk sizing
        risk_mult = nn_result.get("risk_mult", 1.0) * kelly_factor
        target_risk_account = self.user_balance_account * ((self.risk_pct * risk_mult) / 100)
        points_risk = sl_dist / point
        risk_per_lot = points_risk * tick_value
        if risk_per_lot == 0:
            return None

        # Lot Sizing
        lots = target_risk_account / risk_per_lot

        try:
            steps = math.floor(lots / vol_step)
            lots = steps * vol_step
        except Exception:
            lots = round(lots / vol_step) * vol_step

        lots = round(max(min_vol, min(lots, max_vol, self.max_lot)), 2)
        actual_risk_account = risk_per_lot * lots

        # Convert Account Balance to USD for tier checks if necessary
        balance_in_usd = self.user_balance_account
        if self.currency == "ZAR":
            balance_in_usd = self.user_balance_account / usdzar_rate

        # Safety Cap
        max_allowed_pct = get_account_risk_caps(balance_in_usd, self.currency)

        # Absolute hard cap in ZAR
        max_allowed_val = self.user_balance_account * (max_allowed_pct / 100.0)

        # Check if risk exceeds cap
        if actual_risk_account > max_allowed_val:
            # Try to reduce lots
            while actual_risk_account > max_allowed_val and lots > min_vol:
                lots -= vol_step
                actual_risk_account = risk_per_lot * lots

            # Final check after reduction
            lots = round(lots, 2)
            if lots <= min_vol and actual_risk_account > max_allowed_val * 1.2:
                self._log_once(
                    f"risk_{symbol}",
                    f"Skipping {symbol}: Risk (R{actual_risk_account:.2f}) > Cap (R{max_allowed_val * 1.2:.2f})",
                    logging.DEBUG,
                )
                return None

        # Profit Calculation
        points_profit = tp_dist / point
        profit_account = points_profit * tick_value * lots

        # Final Conversion for Reporting (Always provide ZAR for UI)
        actual_risk_zar = actual_risk_account * usdzar_rate if self.currency == "USD" else actual_risk_account
        profit_zar = profit_account * usdzar_rate if self.currency == "USD" else profit_account

        signal.update(
            {
                "price": round(entry_price, digits),
                "sl": round(sl_price, digits),
                "tp": round(tp3_price, digits),
                "tp1": round(tp1_price, digits),
                "tp2": round(tp2_price, digits),
                "tp3": round(tp3_price, digits),
                "lot_size": round(lots, 2),
                "risk_zar": round(actual_risk_zar, 2),
                "profit_zar": round(profit_zar, 2),
                "risk_account": round(actual_risk_account, 2),
                "currency": self.currency,
                "tick_value": tick_value,
                "point": point,
                "atr": atr,
                "is_high_risk": self._is_high_volatility_symbol(symbol),
                "order_type": order_type,
            }
        )
        return signal

    async def _get_htf_trend(self, symbol: str, provider: DataProvider) -> Literal["BULL", "BEAR", "FLAT"]:
        """Fetches trend from HTF using TechnicalAnalyzer strict logic."""
        symbol_info = provider.get_symbol_type(symbol)
        htf_tf = "4h" if symbol_info == "FOREX" else "1h"

        # Cache Check
        now = time.time()
        if symbol in self.htf_cache:
            cache_ts = self.htf_cache[symbol]["time"]
            cache_dt = datetime.fromtimestamp(cache_ts)
            curr_dt = datetime.fromtimestamp(now)

            # Invalidate if day changed
            if cache_dt.day != curr_dt.day and (now - cache_ts < 3600):
                return self.htf_cache[symbol]["trend"]

        klines = await provider.fetch_klines(symbol, htf_tf, 200)
        trend = "FLAT"
        if klines:
            df = pd.DataFrame(klines)
            df = TechnicalAnalyzer.calculate_indicators(df, heavy=False)
            trend = TechnicalAnalyzer.get_htf_trend(df)

        self.htf_cache[symbol] = {"trend": trend, "time": now}
        return trend

    def _get_session_status(self) -> Dict:
        """Returns allowed strategy types based on SAST time."""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()

        allowed_types = ["TREND", "REVERSION", "BREAKOUT"]  # Default allow all

        # Asian Session leans heavily to mean reversion
        if SESSION_CONFIG["ASIAN_START"] <= hour < SESSION_CONFIG["ASIAN_END"]:
            allowed_types = ["REVERSION", "BREAKOUT"]

        # NY Session is optimal for trend/breakouts
        if SESSION_CONFIG["NY_START"] <= hour < SESSION_CONFIG["NY_END"]:
            allowed_types = ["TREND", "BREAKOUT"]

        return {"allow_trade": True, "types": allowed_types}

    def _log_once(self, key: str, message: str, level=logging.INFO):
        """Prevents log spamming for the same event within 5 minutes."""
        now = time.time()
        if key in self._log_throttle:
            if now - self._log_throttle[key] < 300:
                return

        self._log_throttle[key] = now
        logger.log(level, message)

    def _is_high_volatility_symbol(self, symbol: str) -> bool:
        """Checks if symbol is considered High Volatility."""
        return any(x in symbol for x in HIGH_VOLATILITY_IDENTIFIERS)

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Checks if a symbol is on cooldown from last signal."""
        if symbol in self.signal_history:
            elapsed = time.time() - self.signal_history[symbol]
            if elapsed < PAIR_SIGNAL_COOLDOWN:
                return True
        return False

    async def analyze_market(self, symbol: str, klines: list, provider: DataProvider) -> Optional[Dict]:
        """
        Main Analysis pipeline focusing purely on SMC & HTF Alignment.
        """
        # 1. Volatility check
        is_volatile_pair = self._is_high_volatility_symbol(symbol)

        # 2. Cooldown Checks
        if self._is_on_cooldown(symbol) or symbol in self.active_features:
            return None

        # 3. Session Info
        session_info = self._get_session_status()
        if not session_info.get("allow_trade", True):
            return None
        allowed_session_types = session_info.get("types", ["TREND", "REVERSION", "BREAKOUT"])

        # 4. Data Preparation
        klines_m15 = await provider.fetch_klines(symbol, "15m", 200)
        klines_m5 = await provider.fetch_klines(symbol, "5m", 500)
        klines_h4 = await provider.fetch_klines(symbol, "4h", 100)

        df = await asyncio.to_thread(self.prepare_data, klines_m15, True)
        df_5m = await asyncio.to_thread(self.prepare_data, klines_m5, False)
        df_4h = pd.DataFrame(klines_h4)

        # Check if the preparation failed (returned None)
        if df is None or df_5m is None:
            return None

        # Check if the resulting DataFrames are empty
        if df.empty or df_5m.empty or df_4h.empty:
            return None

        curr = df.iloc[-1]

        if curr["atr"] <= 0:
            return None

        # Volatility Spike Check
        avg_atr = df["atr"].tail(24).mean() if len(df) >= 24 else df["atr"].mean()

        # 5. Spread Check
        symbol_info = await provider.get_symbol_info(symbol)
        if not symbol_info:
            return None

        point = symbol_info.get("point", 0.00001)
        spread_info = await provider.get_spread(symbol)
        spread_price = spread_info.get("spread", 0.0) * point

        max_allowed_spread = max(curr["atr"] * 1.5, point * 50)
        if spread_price > max_allowed_spread:
            return None

        # 6. Stateless SMC Detection
        active_fvgs = []
        active_obs = []
        active_breakers = []

        if len(df) >= 20:
            for i in range(len(df) - 20, len(df) - 1):
                c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]

                # --- FVG Detection ---
                if c1["high"] < c3["low"] and c2["close"] > c2["open"]:
                    fvg = {"type": "BULL", "high": c3["low"], "low": c1["high"]}
                    if not any(df.iloc[j]["low"] < fvg["low"] for j in range(i + 1, len(df))):
                        active_fvgs.append(fvg)
                elif c1["low"] > c3["high"] and c2["close"] < c2["open"]:
                    fvg = {"type": "BEAR", "high": c1["low"], "low": c3["high"]}
                    if not any(df.iloc[j]["high"] > fvg["high"] for j in range(i + 1, len(df))):
                        active_fvgs.append(fvg)

                # --- Order Block (OB) Detection ---
                ob_c1, ob_c2 = df.iloc[i - 1], df.iloc[i]

                # Bullish OB / Bearish Breaker
                if ob_c1["close"] < ob_c1["open"] and ob_c2["close"] > ob_c2["open"] and ob_c2["close"] > ob_c1["high"]:
                    ob = {"type": "BULL", "high": ob_c1["high"], "low": ob_c1["low"]}
                    is_broken = False
                    for j in range(i + 1, len(df)):
                        if df.iloc[j]["low"] < ob["low"]:
                            is_broken = True
                            breaker = {"type": "BEAR", "high": ob["high"], "low": ob["low"]}
                            if not any(df.iloc[k]["high"] > breaker["high"] for k in range(j + 1, len(df))):
                                active_breakers.append(breaker)
                            break
                    if not is_broken:
                        active_obs.append(ob)

                # Bearish OB / Bullish Breaker
                elif (
                    ob_c1["close"] > ob_c1["open"] and ob_c2["close"] < ob_c2["open"] and ob_c2["close"] < ob_c1["low"]
                ):
                    ob = {"type": "BEAR", "high": ob_c1["high"], "low": ob_c1["low"]}
                    is_broken = False
                    for j in range(i + 1, len(df)):
                        if df.iloc[j]["high"] > ob["high"]:
                            is_broken = True
                            breaker = {"type": "BULL", "high": ob["high"], "low": ob["low"]}
                            if not any(df.iloc[k]["low"] < breaker["low"] for k in range(j + 1, len(df))):
                                active_breakers.append(breaker)
                            break
                    if not is_broken:
                        active_obs.append(ob)

        # 7. Check Loss Cooldown (Database)
        try:
            if self.db_manager and await self.db_manager.check_recent_loss(symbol):
                self._log_once(f"loss_{symbol}", f"Skipping {symbol}: Loss Cooldown Active")
                return None
        except:
            pass

        structure_info = TechnicalAnalyzer.detect_structure(df)
        htf_trend = await self._get_htf_trend(symbol, provider)
        adx_strength = curr["adx"]

        final_signal = self.strategy_analyzer.analyze_router(
            curr,
            df,
            df_5m,
            df_4h,
            htf_trend,
            active_fvgs,
            active_obs,
            adx_strength,
            structure_info,
            allowed_session_types,
        )

        if not final_signal:
            return None

        # Set volatility flag for Telegram UI
        final_signal["is_high_risk"] = is_volatile_pair

        # Modify the signal confidence based on live BOS/CHoCH structural breaks
        if final_signal["direction"] == "LONG":
            if structure_info["bos"] == "BULL":
                final_signal["confidence"] += 5.0  # Strong trend continuation
            elif structure_info["choch"] == "BULL":
                final_signal["confidence"] += 10.0  # Perfect early entry on reversal
            elif structure_info["choch"] == "BEAR":
                final_signal["confidence"] -= 15.0  # Danger: Local market is reversing against HTF trend

        elif final_signal["direction"] == "SHORT":
            if structure_info["bos"] == "BEAR":
                final_signal["confidence"] += 5.0
            elif structure_info["choch"] == "BEAR":
                final_signal["confidence"] += 10.0
            elif structure_info["choch"] == "BULL":
                final_signal["confidence"] -= 15.0

        # 8. ML Prediction (Feature Extraction)
        now = datetime.now()
        dist_vwap = (curr["close"] - curr["vwap"]) / curr["vwap"] if curr["vwap"] != 0 else 0.0
        mtf_align = (
            1
            if (final_signal["direction"] == "LONG" and htf_trend == "BULL")
            else (-1 if (final_signal["direction"] == "SHORT" and htf_trend == "BEAR") else 0)
        )
        vol_ratio = curr["atr"] / avg_atr if avg_atr > 0 else 1.0

        dist_nearest_fvg = 0.0
        if active_fvgs:
            nearest = min(active_fvgs, key=lambda x: abs(x["high"] - curr["close"]))
            dist_nearest_fvg = abs(nearest["high"] - curr["close"]) / curr["close"]

        # Check if price is sitting inside a Breaker that aligns with our signal
        is_in_breaker_val = 0.0
        if final_signal["direction"] == "LONG":
            if any(b["low"] <= curr["close"] <= b["high"] for b in active_breakers if b["type"] == "BULL"):
                is_in_breaker_val = 1.0
        elif final_signal["direction"] == "SHORT":
            if any(b["low"] <= curr["close"] <= b["high"] for b in active_breakers if b["type"] == "BEAR"):
                is_in_breaker_val = 1.0

        features = {
            "dist_to_vwap": dist_vwap,
            "mtf_trend_alignment": mtf_align,
            "hour_norm": now.hour / 24.0,
            "volatility_ratio": vol_ratio,
            "dist_to_nearest_fvg": dist_nearest_fvg,
            "is_in_breaker": is_in_breaker_val,
            "htf_adx_strength": adx_strength,
            "poi_status": 1.0 if len(active_fvgs) > 0 or len(active_obs) > 0 else 0.0,
        }

        # Inference from read-only bundled model
        nn_result = self.nn_brain.predict(features)

        # Confidence Adjustment
        final_signal = await self._adjust_confidence(symbol, final_signal, nn_result["prob"], htf_trend, vol_ratio)
        if final_signal["confidence"] < self.min_confidence:
            return None

        # 9. Execution & Final Risk Sizing
        tick = await provider.get_current_tick(symbol)
        if not tick:
            return None

        result = await self._calculate_risk_metrics(symbol, final_signal, curr, tick, symbol_info, nn_result, provider)
        if result:
            self.signal_history[symbol] = time.time()
            self.active_features[symbol] = features
            return result

        return None

    def prepare_data(self, klines: list, heavy: bool = True) -> Optional[pd.DataFrame]:
        """Prepares DataFrame with Indicators for Analysis."""
        try:
            df = pd.DataFrame(klines)
            if df.empty:
                return None
            df = df.sort_values("time").reset_index(drop=True)
            df = TechnicalAnalyzer.calculate_indicators(df, heavy=heavy)
            df = CandleStickDetector.calculate_candles(df)
            return df
        except Exception as e:
            logger.error(f"Data prep error: {e}")
            return None

    def rank_symbols_by_volatility(self, symbols: List[str], data_map: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Sorts symbols by ATR (Volatility).
        """
        scored = []
        for sym in symbols:
            if sym in data_map:
                df = data_map[sym]
                if not df.empty and "atr" in df.columns:
                    if df.iloc[-1]["close"] > 0:
                        atr_pct = (df.iloc[-1]["atr"] / df.iloc[-1]["close"]) * 100
                        scored.append((sym, atr_pct))

        # Sort descending (Highest Volatility first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def record_trade_outcome(self, symbol: str, won: bool, pnl: float, excursion: float = 0.0):
        """Records a trade after it has been completed to update ML CSV"""
        logger.info(f"🏁 Trade Closed: {symbol} | PnL: {pnl} | Won: {won}")

        if symbol in self.active_features:
            features = self.active_features[symbol]
            target_win = 1 if won else 0

            collector = DataCollector()
            collector.log_training_data(symbol, features, target_win, pnl, excursion)
            logger.info(f"💾 Live features for {symbol} saved to training data.")

            del self.active_features[symbol]

    def register_active_trade(self, symbol: str):
        """Manually marks a symbol as active (used during recovery)."""
        if symbol not in self.active_features:
            self.active_features[symbol] = {}

    def set_context(self, balance: float, db: DatabaseManager, currency: str = "USD"):
        """Sets user balance and database manager."""
        self.user_balance_account = balance
        self.currency = currency
        self.db_manager = db
