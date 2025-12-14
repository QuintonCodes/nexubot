import logging
import pandas as pd
import time
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Literal, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.patterns import FakeBreakoutDetector, PatternRecognizer
from src.data.collector import DataCollector
from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ml_engine import NeuralPredictor
from src.engine.strategies import StrategyAnalyzer
from src.config import (
    FOREX_SYMBOLS,
    HIGH_RISK_SYMBOLS,
    MAX_LOT_SIZE,
    MIN_CONFIDENCE,
    PAIR_SIGNAL_COOLDOWN,
    RISK_PER_TRADE_PCT,
    SESSION_CONFIG,
)

logger = logging.getLogger(__name__)


class AITradingEngine:
    """
    Advanced Intelligence Engine.
    Integrates Multi-Timeframe Analysis, Regime Filtering, Pattern Recognition, and Neural Networks.
    """

    def __init__(self):
        self.strategy_analyzer = StrategyAnalyzer()
        self.fake_detector = FakeBreakoutDetector()
        self.pattern_recognizer = PatternRecognizer()
        self.nn_brain = NeuralPredictor()
        self.data_collector = DataCollector()
        self.user_balance_zar = 0.0
        self.db_manager = None
        self.signal_history = {}
        self.active_features = {}
        self.htf_cache = {}
        self.cached_usdzar_rate = 18.00

        self.nn_brain.train_network()

    async def _adjust_confidence(
        self, symbol: str, signal: dict, nn_prob: float, htf_trend: Literal["BULL", "BEAR", "FLAT"]
    ) -> Dict:
        """Dynamic Confidence Calculation.
        Factors:
        1. Base Strategy Confidence (Static)
        2. Neural Network Probability (AI)
        3. Trend Alignment (Context)
        4. Historical Pair Performance (Memory)"""
        strat_conf = signal["confidence"]

        # 1. Trend Alignment Bonus (Binary)
        # If trading with the H4 trend, +10% confidence. Against it, -10%.
        trend_bonus = 0
        if htf_trend == "BULL":
            trend_bonus = 10 if signal["direction"] == "LONG" else -10
        elif htf_trend == "BEAR":
            trend_bonus = 10 if signal["direction"] == "SHORT" else -10

        # 2. Historical Performance (Memory)
        # Fetch win rate for this specific pair from DB
        hist_win_rate = 0.5  # Default neutral
        strat_win_rate = 0.5
        if self.db_manager:
            hist_win_rate = await self.db_manager.get_pair_performance(symbol)
            strat_win_rate = await self.db_manager.get_strategy_performance(signal["strategy"])

        # If strategy sucks (WR < 40%), massive penalty (-20)
        # If strategy rules (WR > 60%), bonus (+10)
        strat_perf_factor = 0
        if strat_win_rate < 0.4:
            strat_perf_factor = -20
        elif strat_win_rate > 0.6:
            strat_perf_factor = 10

        # Scale historical factor: 0.5 (50%) -> 0 adjustment.
        # 0.8 (80%) -> +15 adjustment. 0.2 (20%) -> -15 adjustment.
        history_factor = (hist_win_rate - 0.5) * 50

        # 3. Neural Network Weighting
        # Map 0.0-1.0 prob to a -20 to +20 score
        nn_factor = (nn_prob - 0.5) * 40

        # --- FINAL CALCULATION ---
        # Base (Strategy) + Trend + History + AI
        final_conf = strat_conf + trend_bonus + history_factor + nn_factor + strat_perf_factor
        # Clamp between 0 and 99
        final_conf = max(0.0, min(99.9, final_conf))

        signal["confidence"] = final_conf
        return signal

    def _calculate_risk_metrics(
        self, symbol: str, signal: dict, curr: pd.Series, tick, info: dict, nn_result: dict
    ) -> Optional[Dict]:
        """
        Calculates Lot Size and Risk using Tick Value
        """
        if self.user_balance_zar <= 0:
            return None

        ask = tick.ask
        bid = tick.bid
        point = info["point"]
        tick_value = info.get("trade_tick_value", 0)
        min_vol = info.get("min_vol", 0.01)
        max_vol = info.get("max_vol", 100.0)
        atr = float(curr["atr"])

        tp_dist = atr * nn_result["pred_exit_atr"]
        sl_dist = atr * 1.5
        min_dist = point * 20
        if sl_dist < min_dist:
            sl_dist = min_dist

        if signal["signal"] == "BUY":
            entry_price = ask
            sl_price = ask - sl_dist
            tp_price = ask + tp_dist
        else:
            entry_price = bid
            sl_price = bid + sl_dist
            tp_price = bid - tp_dist

        # 2. Calculate Risk Amount per 1.0 Lot
        if tick_value == 0:
            return None

        # Risk Amount per 1 Lot = (Points Diff) * Tick Value
        # Points Diff = Distance / Point
        points_risk = sl_dist / point
        risk_per_lot = points_risk * tick_value
        if risk_per_lot == 0:
            return None

        # Target Risk
        risk_mult = nn_result["risk_mult"]
        target_risk_zar = self.user_balance_zar * ((RISK_PER_TRADE_PCT * risk_mult) / 100)

        # 3. Lot Sizing
        lots = target_risk_zar / risk_per_lot

        # Rounding
        step = info.get("vol_step", 0.01)
        lots = round(lots / step) * step
        lots = max(min_vol, min(lots, max_vol, MAX_LOT_SIZE))

        actual_risk_zar = risk_per_lot * lots

        if actual_risk_zar > (self.user_balance_zar * 0.20):
            logger.info(f"Filtering {symbol}: Risk R{actual_risk_zar:.2f} > 20% of Balance")
            return None

        # Profit Calculation
        points_profit = tp_dist / point
        profit_zar = points_profit * tick_value * lots

        signal.update(
            {
                "price": entry_price,
                "sl": round(sl_price, info["digits"]),
                "tp": round(tp_price, info["digits"]),
                "lot_size": round(lots, 2),
                "risk_zar": round(actual_risk_zar, 2),
                "profit_zar": round(profit_zar, 2),
                "tick_value": tick_value,
                "point": point,
                "atr": atr,
                "is_high_risk": symbol in HIGH_RISK_SYMBOLS,
            }
        )
        return signal

    async def _get_htf_trend(self, symbol: str, provider: DataProvider) -> Literal["BULL", "BEAR", "FLAT"]:
        """Fetches H4 EMA trend. Returns 'BULL', 'BEAR', or 'FLAT'."""

        # Cache Check (Valid for 1 Hour)
        now = time.time()
        if symbol in self.htf_cache:
            if now - self.htf_cache[symbol]["time"] < 3600:
                return self.htf_cache[symbol]["trend"]

        # Fetch New
        klines = await provider.fetch_klines(symbol, "4h", 200)
        trend = "FLAT"
        if klines:
            df = pd.DataFrame(klines)
            ema_200 = df["close"].ewm(span=200).mean().iloc[-1]
            if df["close"].iloc[-1] > ema_200:
                trend = "BULL"
            elif df["close"].iloc[-1] < ema_200:
                trend = "BEAR"

        # Save to Cache
        self.htf_cache[symbol] = {"trend": trend, "time": now}
        return trend

    def _is_market_session_open(self, symbol: str) -> bool:
        """
        Checks South African Time (SAST).
        Handles Weekends for Forex/Gold.
        """
        now = datetime.now()
        day = now.weekday()  # 0=Mon, 6=Sun
        hour = now.hour

        if symbol in FOREX_SYMBOLS:
            # Saturday (5) and Sunday (6) are closed
            if day >= 5:
                return False

            # Optional: Friday close logic (e.g., stop after 22:00)
            if day == 4 and hour >= 22:
                return False

            # Session Hours (e.g. 09:00 - 22:00) logic
            start = SESSION_CONFIG["FOREX_START"]
            end = SESSION_CONFIG["FOREX_END"]
            return start <= hour < end

        return True  # Crypto is 24/7

    def _prepare_data(self, klines: List[Dict]) -> Optional[pd.DataFrame]:
        """Prepares DataFrame with Indicators for Analysis."""
        try:
            df = pd.DataFrame(klines)
            if df.empty:
                return None
            df = df.sort_values("time").reset_index(drop=True)
            return TechnicalAnalyzer.calculate_indicators(df)
        except Exception as e:
            logger.error(f"Data prep error: {e}")
            return None

    async def analyze_market(self, symbol: str, klines: List[Dict], provider: DataProvider) -> Optional[Dict]:
        """
        Main Analysis Pipeline with Spread, Volatility, and Context filters.
        """
        # 1. Session & Cooldown
        if not self._is_market_session_open(symbol):
            return None

        # Check cooldown OR if currently in a trade (Active Features)
        if self.is_on_cooldown(symbol) or symbol in self.active_features:
            return None

        # Safe DB Check (Handles cancellation during shutdown)
        try:
            if self.db_manager and await self.db_manager.check_recent_loss(symbol):
                return None
        except Exception:
            pass

        # 2. Spread Check (New)
        spread_info = await provider.get_spread(symbol)
        if spread_info["spread_high"]:
            logger.info(f"High Spread {symbol}: {spread_info['spread']:.1f}")
            return None

        # Data Prep
        df = await asyncio.to_thread(self._prepare_data, klines)
        if df is None:
            return None
        curr = df.iloc[-1]

        # Stale Check
        if (time.time() - curr["time"]) > 1800:
            return None

        # Volatility Filter (News Filter)
        avg_atr = df["atr"].rolling(24).mean().iloc[-1]
        if curr["atr"] > (avg_atr * 3):
            logger.info(f"Skipping {symbol}: Extreme Volatility (ATR Spike)")
            self.signal_history[symbol] = time.time() + 1800  # 30 min ban
            return None

        # Context & Patterns
        volatility_ratio = curr["atr"] / avg_atr if avg_atr > 0 else 1.0
        htf_trend = await self._get_htf_trend(symbol, provider)
        patterns = self.pattern_recognizer.analyze_patterns(df)

        # Fakeout Check
        fake_analysis = self.fake_detector.analyze(df)
        fake_risk_penalty = 1.0
        if fake_analysis["risk_score"] >= 50:
            logger.info(f"Skipping {symbol}: Fakeout ({fake_analysis['reasons']})")
            return None
        elif fake_analysis["risk_score"] >= 30:
            fake_risk_penalty = 0.5  # Halve the risk for "sus" setups

        # Stratey Routing
        potential_signal = None
        if patterns:
            potential_signal = patterns[0]
            potential_signal["strategy"] = f"Chart Pattern ({potential_signal['pattern']})"
        elif symbol in FOREX_SYMBOLS:
            potential_signal = self.strategy_analyzer.analyze_forex(curr, df, htf_trend, patterns)
        else:
            potential_signal = self.strategy_analyzer.analyze_crypto(curr, df, patterns)

        if not potential_signal:
            return None

        # ML Features
        now = datetime.now()
        pivots = df[df["high"] == df["high"].rolling(10, center=True).max()]["high"]
        last_pivot = pivots.iloc[-1] if not pivots.empty else curr["high"]
        dist_to_pivot = abs(curr["close"] - last_pivot) / curr["close"]

        features = {
            "rsi": curr["rsi"],
            "adx": curr["adx"],
            "atr": curr["atr"],
            "ema_dist": (curr["close"] - curr["ema_50"]) / curr["close"],
            "bb_width": curr["bb_width"],
            "vol_ratio": curr["volume"] / curr["vol_sma"] if curr["vol_sma"] else 1,
            "htf_trend": 1 if htf_trend == "BULL" else (-1 if htf_trend == "BEAR" else 0),
            "dist_to_pivot": dist_to_pivot,
            "hour_norm": now.hour / 24.0,
            "day_norm": now.weekday() / 6.0,
            "wick_ratio": (curr["high"] - curr["close"]) / (curr["high"] - curr["low"]),
            "dist_ema200": (curr["close"] - curr["ema_200"]) / curr["close"],
            "volatility_ratio": volatility_ratio,
        }

        # Predict
        nn_result = self.nn_brain.predict(features)
        if nn_result["prob"] < 0.3:
            # Log as a "Loss" (0) so bot learns this was bad
            self.data_collector.log_training_data(symbol, features, 0, 0.0, 0.0)
            return None

        # Confidence Adjustment
        final_signal = await self._adjust_confidence(symbol, potential_signal, nn_result["prob"], htf_trend)
        if final_signal["confidence"] < MIN_CONFIDENCE:
            return None

        # Apply Fakeout Penalty to Risk Multiplier
        nn_result["risk_mult"] *= fake_risk_penalty

        # Risk & Targets
        symbol_info = await provider.get_symbol_info(symbol)
        if not symbol_info:
            return None

        # Get live Tick data for true Bid/Ask
        tick = await provider.get_current_tick(symbol)
        if not tick:
            return None

        result = self._calculate_risk_metrics(symbol, final_signal, curr, tick, symbol_info, nn_result)
        if result:
            self.signal_history[symbol] = time.time()
            self.active_features[symbol] = features
            return result

        return None

    def is_on_cooldown(self, symbol: str) -> bool:
        """Checks if a symbol is on cooldown from last signal."""
        last_time = self.signal_history.get(symbol, 0)
        return (time.time() - last_time) < PAIR_SIGNAL_COOLDOWN

    def rank_symbols_by_volatility(self, symbols: List[str], data_map: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Sorts symbols by ATR (Volatility).
        """
        scored = []
        for sym in symbols:
            if sym in data_map:
                df = data_map[sym]
                if not df.empty and "atr" in df.columns:
                    atr_pct = (df.iloc[-1]["atr"] / df.iloc[-1]["close"]) * 100
                    scored.append((sym, atr_pct))

        # Sort descending (Highest Volatility first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def record_trade_outcome(self, symbol: str, won: bool, pnl: float, excursion: float = 0.0):
        """Called by Console after trade finishes."""
        # 1. Update Learning
        # 2. Log Data for ML
        if symbol in self.active_features:
            self.data_collector.log_training_data(symbol, self.active_features[symbol], 1 if won else 0, pnl, excursion)
            del self.active_features[symbol]

    def register_active_trade(self, symbol: str):
        """Manually marks a symbol as active (used during recovery)."""
        # We place a placeholder dict so analyze_market skips this symbol
        if symbol not in self.active_features:
            self.active_features[symbol] = {}

    def set_context(self, balance: float, db: DatabaseManager):
        """Sets user balance and database manager."""
        self.user_balance_zar = balance
        self.db_manager = db

    def update_usdzar_rate(self, rate: float):
        """Updates cached USDZAR rate for Forex calculations."""
        if rate and rate > 10:
            self.cached_usdzar_rate = rate
