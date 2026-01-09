import asyncio
import logging
import math
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.patterns import FakeBreakoutDetector, PatternRecognizer
from src.data.collector import DataCollector
from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ml_engine import NeuralPredictor
from src.engine.strategies import StrategyAnalyzer
from src.utils.trainer import ModelTrainer
from src.config import (
    CHOP_THRESHOLD_TREND,
    CHOP_THRESHOLD_RANGE,
    CRYPTO_SYMBOLS,
    FOREX_SYMBOLS,
    HIGH_RISK_SYMBOLS,
    MAX_LOT_SIZE,
    MIN_CONFIDENCE,
    PAIR_SIGNAL_COOLDOWN,
    RISK_PER_TRADE_PCT,
    SESSION_CONFIG,
    get_account_risk_caps,
)

logger = logging.getLogger(__name__)


class AITradingEngine:
    """
    Advanced Intelligence Engine.
    Integrates Multi-Timeframe Analysis, Regime Filtering, Pattern Recognition, and Neural Networks.
    """

    def __init__(self):
        self.strategy_analyzer = StrategyAnalyzer()

        ModelTrainer.train_if_needed()
        self.nn_brain = NeuralPredictor()

        self._log_throttle = {}
        self.active_features = {}
        self.db_manager = None
        self.htf_cache = {}
        self.last_news_load_time = 0
        self.low_vol_candidates = {}
        self.news_blocks = self._load_news_blocks()
        self.signal_history = {}
        self.user_balance_zar = 0.0

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
        base_conf = min(signal["confidence"], MIN_CONFIDENCE)

        # 1. Trend Alignment
        trend_bonus = 0
        if htf_trend == "BULL" and signal["direction"] == "LONG":
            trend_bonus = 5
        elif htf_trend == "BEAR" and signal["direction"] == "SHORT":
            trend_bonus = 5
        elif htf_trend != "FLAT":
            if "reversion" in signal["strategy"].lower():
                trend_bonus = -5
            else:
                trend_bonus = -15  # Counter trend trade

        # 2. Historical Performance
        hist_win_rate = 0.5
        if self.db_manager:
            hist_win_rate = await self.db_manager.get_pair_performance(symbol)

        # Heavy Penalty for losers (< 40% win rate), Small Bonus for winners
        history_factor = -10 if hist_win_rate < 0.4 else (10 if hist_win_rate > 0.6 else 0)

        # 3. Neural Network Weighting
        nn_factor = (nn_prob - 0.5) * 40

        # 4. Volatility Penalty
        vol_penalty = -10 if volatility_ratio > 1.5 else (-20 if volatility_ratio > 2.0 else 0)

        # --- FINAL CALCULATION ---
        # Base (Strategy) + Trend + History + AI
        final_conf = base_conf + trend_bonus + history_factor + nn_factor + vol_penalty

        # Clamp between 0 and 99
        final_conf = max(0.0, min(99.0, final_conf))

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
        vol_step = info.get("vol_step", 0.01)
        digits = info.get("digits", 5)
        atr = float(curr["atr"])

        # --- 1. SMART ENTRY LOGIC ---
        current_market_price = ask if signal["direction"] == "LONG" else bid
        order_type = signal.get("order_type", "MARKET")
        entry_price = signal.get("price", current_market_price)

        # Determine Order Type
        if order_type == "MARKET":
            entry_price = current_market_price

        # Calculate Chase Distance (in ATR multiples)
        signal_close_price = curr["close"]
        chase_dist = abs(current_market_price - signal_close_price)
        if chase_dist > (atr * 0.5):
            logger.debug(f"Skipping {symbol}: Price moved too far ({chase_dist/atr:.2f} ATR)")
            return None

        # --- GHOST ORDER LOGIC ---
        strategy_name = signal.get("strategy", "").lower()

        # Momentum needs immediate execution. Limit orders cause missed trades here.
        is_momentum = any(x in strategy_name for x in ["breakout", "flow", "ichimoku"])

        if "fvg" not in strategy_name and "limit" not in str(order_type).lower() and not is_momentum:
            if signal["confidence"] <= 85.0:
                is_reversal = "reversion" in strategy_name or "divergence" in strategy_name
                order_type = "LIMIT"
                ghost_pips = 15 * point if is_reversal else 10 * point
                if signal["direction"] == "LONG":
                    entry_price = current_market_price - ghost_pips
                else:
                    entry_price = current_market_price + ghost_pips

        # --- 2. TP / SL CALCULATION ---
        # Base SL is tightened to 1.0 ATR
        # FOR XAUUSD: Tighten to 0.75 ATR to reduce absolute risk amount (cheaper trade).
        sl_multiplier = 0.75 if "XAU" in symbol else 1.0
        sl_dist = atr * sl_multiplier
        min_dist = point * 50
        sl_dist = max(sl_dist, min_dist)

        # Use Dynamic SL if provided
        if "suggested_sl" in signal:
            suggested_dist = abs(signal["suggested_sl"] - entry_price)
            # Safety Check: Don't allow SL to be dangerously tight (< 0.5 ATR) or too wide (> 3 ATR)
            if (atr * 0.5) < suggested_dist < (atr * 3.0):
                sl_dist = suggested_dist

        # Probability calibration hook
        prob = nn_result.get("prob", 0.5)
        pred_exit_atr = float(nn_result.get("pred_exit_atr", 2.0))
        # Bound predicted exit for sanity
        pred_exit_atr = max(0.8, min(pred_exit_atr, 6.0))

        # conservative TP based on predicted exit ATR but always at least 1.2x SL
        tp_dist = max(atr * pred_exit_atr, sl_dist * 1.2)

        rr = (tp_dist / sl_dist) if sl_dist > 0 else 1.0
        expected_ev = prob * rr - (1.0 - prob)

        # Kelly-informed adjustment (small, capped multiplier)
        kelly = prob - ((1 - prob) / (rr + 1e-9))
        if kelly > 0:
            kelly_factor = min(1.5, max(0.5, 1.0 + (kelly * 2.0)))  # modest scaling
        else:
            kelly_factor = 0.5  # shrink size for negative Kelly

        # If EV is clearly negative, reduce risk_mult / skip
        if expected_ev < 0:
            # degrade risk multiplier to limit exposure
            nn_result["risk_mult"] = max(0.25, nn_result.get("risk_mult", 1.0) * 0.5)
            # hard skip if very negative
            if expected_ev < -0.25:
                self._log_once(f"ev_gate_{symbol}", f"Skipping {symbol}: Negative EV ({expected_ev:.2f})")
                return None

        # Enforce minimum acceptable RR for low-prob trades
        if rr < 1.2 and prob < 0.65:
            self._log_once(
                f"rr_bad_{symbol}", f"Skipping {symbol}: Low RR {rr:.2f} with low prob {prob:.2f}", logging.DEBUG
            )
            return None

        # Calculate Absolute Prices
        if signal["signal"] == "BUY":
            # For Limit orders, SL/TP relative to Limit Price
            ref_price = entry_price if order_type == "LIMIT" else ask
            sl_price = ref_price - sl_dist
            tp_price = ref_price + tp_dist
        else:
            ref_price = entry_price if order_type == "LIMIT" else bid
            sl_price = ref_price + sl_dist
            tp_price = ref_price - tp_dist

        # --- 3. RISK SIZING ---
        risk_mult = nn_result.get("risk_mult", 1.0) * kelly_factor
        if signal.get("is_shadow", False):
            risk_mult = min(risk_mult, 0.1)

        target_risk_zar = self.user_balance_zar * ((RISK_PER_TRADE_PCT * risk_mult) / 100)
        points_risk = sl_dist / point
        risk_per_lot = points_risk * tick_value
        if risk_per_lot == 0:
            return None

        # Lot Sizing
        lots = target_risk_zar / risk_per_lot

        # Round lots to exchange vol_step safely using floor to avoid over-risk
        try:
            steps = math.floor(lots / vol_step)
            lots = steps * vol_step
        except Exception:
            lots = round(lots / vol_step) * vol_step
        lots = round(lots / vol_step) * vol_step

        lots = max(min_vol, min(lots, max_vol, MAX_LOT_SIZE))

        actual_risk_zar = risk_per_lot * lots

        # --- SAFETY CAP ---
        max_allowed_pct = get_account_risk_caps(self.user_balance_zar)
        if risk_mult > 1.0:
            max_allowed_pct *= risk_mult

        # Absolute hard cap in ZAR
        max_allowed_zar = self.user_balance_zar * (max_allowed_pct / 100.0)

        # Check if risk exceeds cap
        if actual_risk_zar > max_allowed_zar and not signal.get("is_shadow", False):
            # Try to reduce lots
            while actual_risk_zar > max_allowed_zar and lots > min_vol:
                lots -= vol_step
                actual_risk_zar = risk_per_lot * lots

            # Final check after reduction
            lots = round(lots, 2)
            if lots <= min_vol and actual_risk_zar > max_allowed_zar:
                emergency_cap_pct = 0.35 if "XAU" in symbol else 0.08
                max_emergency_risk = self.user_balance_zar * emergency_cap_pct
                if actual_risk_zar < max_emergency_risk:
                    # Allow trade but log warning
                    self._log_once(
                        f"high_risk_{symbol}",
                        f"⚠️ High Risk Accepted for {symbol}: R{actual_risk_zar:.2f} ({actual_risk_zar/self.user_balance_zar*100:.1f}%)",
                    )
                    pass
                else:
                    self._log_once(
                        f"risk_{symbol}",
                        f"Skipping {symbol}: Min Lot Risk (R{actual_risk_zar:.2f}) > 8% Safety Cap",
                        logging.DEBUG,
                    )
                    return None
            elif actual_risk_zar > (max_allowed_zar * 1.1):
                self._log_once(
                    f"risk_{symbol}",
                    f"Skipping {symbol}: Risk (R{actual_risk_zar:.2f}) > Cap (R{max_allowed_zar:.2f})",
                    logging.DEBUG,
                )
                return None

        # Profit Calculation
        points_profit = tp_dist / point
        profit_zar = points_profit * tick_value * lots

        signal.update(
            {
                "price": round(entry_price, digits),
                "sl": round(sl_price, digits),
                "tp": round(tp_price, digits),
                "lot_size": round(lots, 2),
                "risk_zar": round(actual_risk_zar, 2),
                "profit_zar": round(profit_zar, 2),
                "tick_value": tick_value,
                "point": point,
                "atr": atr,
                "is_high_risk": symbol in HIGH_RISK_SYMBOLS,
                "order_type": order_type,
            }
        )
        return signal

    def _check_low_vol_candidates(self, symbol: str, curr: pd.Series) -> Optional[Dict]:
        """Checks if a previously rejected 'Low Vol' trade is now valid (Late Bloomer)."""
        if symbol not in self.low_vol_candidates:
            return None

        cached = self.low_vol_candidates[symbol]
        # Expire after 15 mins
        if (time.time() - cached["time"]) > 900:
            del self.low_vol_candidates[symbol]
            return None

        # Logic: If Volume is now strong AND price is still near original entry
        vol_is_strong = curr["volume"] > curr["vol_sma"]
        price_near_entry = abs(curr["close"] - cached["entry"]) < (curr["atr"] * 0.5)

        if vol_is_strong and price_near_entry:
            logger.info(f"🌱 Late Bloomer Activated: {symbol} volume spike detected!")
            del self.low_vol_candidates[symbol]
            return cached["signal"]

        return None

    def _check_news_update(self):
        """Checks for updates to news_block.txt every 5 minutes."""
        now = time.time()
        if now - self.last_news_load_time < 300:
            return

        if os.path.exists("news_block.txt"):
            mtime = os.path.getmtime("news_block.txt")
            if mtime > self.last_news_load_time:
                self.news_blocks = self._load_news_blocks()
                self.last_news_load_time = mtime
                logger.info("📅 News blocks updated dynamically.")
        else:
            self.last_news_load_time = now

    async def _get_htf_trend(self, symbol: str, provider: DataProvider) -> Literal["BULL", "BEAR", "FLAT"]:
        """Fetches H4 EMA trend."""
        htf_tf = "4h" if symbol in FOREX_SYMBOLS else "1h"

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
            # Calculate Indicators
            df = TechnicalAnalyzer.calculate_indicators(df, heavy=False)
            # Use Shared Logic
            trend = TechnicalAnalyzer.get_htf_trend(df)

        self.htf_cache[symbol] = {"trend": trend, "time": now}
        return trend

    def _get_session_status(self) -> Dict:
        """Returns allowed strategy types based on SAST time."""
        # 1. News Block Check
        if self._is_news_blocked():
            return {"allow_trade": False, "reason": "High Impact News"}

        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()  # 0=Mon, 6=Sun

        # Monday Morning Block (First 2 hours)
        if weekday == 0 and hour < 2:
            return {"allow_trade": False, "reason": "Monday Open"}

        # Pre-London Trap Zone
        if SESSION_CONFIG["PRE_LONDON_START"] <= hour < SESSION_CONFIG["PRE_LONDON_END"]:
            return {"allow_trade": False, "reason": "Pre-London Trap"}

        allowed_types = []

        # Asian Session (Mean Reversion Only)
        if SESSION_CONFIG["ASIAN_START"] <= hour < SESSION_CONFIG["ASIAN_END"]:
            allowed_types.append("REVERSION")
        else:
            # London/NY (Trend + Breakout)
            allowed_types.append("TREND")

            # Block Volatility Breakouts at 13:00 SAST
            if hour != SESSION_CONFIG["NO_VOLATILITY_HOUR"]:
                allowed_types.append("BREAKOUT")

        return {"allow_trade": True, "types": allowed_types}

    def _load_news_blocks(self) -> List[Dict]:
        """
        Parses news_block.txt for blocked time ranges.
        """
        blocks = []
        if not os.path.exists("news_block.txt"):
            return blocks
        try:
            with open("news_block.txt", "r") as f:
                for line in f:
                    if "->" in line and not line.strip().startswith("#"):
                        parts = line.split("#")[0].strip().split("->")
                        if len(parts) == 2:
                            try:
                                start = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M")
                                end = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M")

                                # Add 30-minute safety buffer
                                start_buffer = start - timedelta(minutes=30)
                                end_buffer = end + timedelta(minutes=30)

                                blocks.append({"start": start_buffer, "end": end_buffer})
                            except ValueError:
                                logger.warning(f"Invalid date format in news_block.txt: {line.strip()}")
        except Exception as e:
            logger.error(f"Failed to load news blocks: {e}")
        return blocks

    def _log_once(self, key: str, message: str, level=logging.INFO):
        """Prevents log spamming for the same event within 5 minutes."""
        now = time.time()
        if key in self._log_throttle:
            if now - self._log_throttle[key] < 300:  # 5 minutes
                return

        self._log_throttle[key] = now
        logger.log(level, message)

    def _is_news_blocked(self) -> bool:
        """
        Checks if current time is inside a news block window (including buffers).
        """
        now = datetime.now()
        for block in self.news_blocks:
            if block["start"] <= now <= block["end"]:
                return True
        return False

    def _prepare_data(self, klines: List[Dict], heavy: bool = True) -> Optional[pd.DataFrame]:
        """Prepares DataFrame with Indicators for Analysis."""
        try:
            df = pd.DataFrame(klines)
            if df.empty:
                return None
            df = df.sort_values("time").reset_index(drop=True)

            analyzer = TechnicalAnalyzer()
            return analyzer.calculate_indicators(df, heavy=heavy)
        except Exception as e:
            logger.error(f"Data prep error: {e}")
            return None

    async def analyze_market(self, symbol: str, klines: List[Dict], provider: DataProvider) -> Optional[Dict]:
        """
        Main Analysis with step-by-step pipeline
        """
        # --- 1. Cooldown & Active states ---
        # Check simple memory-based flags first to avoid expensive calls
        if self.is_on_cooldown(symbol) or symbol in self.active_features:
            return None

        # --- 2. Session & News Filter ---
        # Check static time blocks (CPU only)
        if self._is_news_blocked():
            self._log_once(f"news_static_{symbol}", f"Skipping {symbol}: 🔴 Static News Block Active")
            return None

        session_info = self._get_session_status()
        if not session_info["allow_trade"]:
            self._log_once("session_block", f"Skipping analysis: {session_info['reason']}")
            return None

        # Check live news (Requires API/Network)
        is_news_blocked = await provider.check_live_news_block(symbol, [])
        if is_news_blocked:
            self._log_once(f"news_{symbol}", f"Skipping {symbol}: 🔴 Live High-Impact News Detected")
            return None

        # --- 3. Data Preparation & Integrity ---
        df = await asyncio.to_thread(self._prepare_data, klines, True)
        if df is None:
            return None
        curr = df.iloc[-1]

        # Stale Check
        if (time.time() - curr["time"]) > 1800:
            return None

        # --- 4. Basic Volatility & Spread ---
        # ATR Check
        if curr["atr"] <= 0:
            self._log_once(f"atr_low_{symbol}", f"⏳ Initializing {symbol}: ATR is 0 (Waiting for more data)")
            return None

        avg_atr = df["atr"].tail(24).mean() if len(df) >= 24 else df["atr"].mean()

        # Volatility Spike Check
        if curr["atr"] > (avg_atr * 3):
            self._log_once(f"vol_{symbol}", f"Skipping {symbol}: Extreme Volatility (ATR Spike)")
            self.signal_history[symbol] = time.time() + 1800  # 30 min ban
            return None

        # Spread Check (Network Call)
        symbol_info = await provider.get_symbol_info(symbol)
        if not symbol_info:
            return None
        point = symbol_info.get("point", 0.00001)

        # Filter: Spread > 90% of 14-period ATR
        spread_info = await provider.get_spread(symbol)
        spread_points = spread_info.get("spread", 0.0)
        spread_price = spread_points * point

        if spread_price > (curr["atr"] * 0.9):
            self._log_once(
                f"spread_{symbol}",
                f"Skipping {symbol}: Spread {spread_points:.0f}pts ({spread_price:.5f}) > 0.5 ATR ({curr['atr']:.5f})",
                logging.DEBUG,
            )
            return None

        volatility_ratio = curr["atr"] / avg_atr if avg_atr > 0 else 1.0
        htf_trend = await self._get_htf_trend(symbol, provider)

        # --- 5. Recovery Check (Late Bloomers / Loss Cooldown) ---
        late_signal = self._check_low_vol_candidates(symbol, curr)
        final_signal_candidate = None
        is_late_recovery = False

        if late_signal:
            final_signal_candidate = late_signal
            is_late_recovery = True
            logger.info(f"🚀 Processing Late Bloomer for {symbol} | Trend: {htf_trend}")
        else:
            # Adaptive Cooldown logic
            current_cooldown_req = PAIR_SIGNAL_COOLDOWN * (2 if volatility_ratio > 1.5 else 1)
            last_time = self.signal_history.get(symbol, 0)
            if (time.time() - last_time) < current_cooldown_req:
                return None

            # Check Loss Cooldown (Database)
            try:
                if self.db_manager:
                    is_loss = await self.db_manager.check_recent_loss(symbol)
                    if is_loss:
                        self._log_once(f"loss_{symbol}", f"Skipping {symbol}: Loss Cooldown Active")
                        return None
            except Exception:
                pass

        # --- 6. Technical Analysis & Context ---
        # Define shared variables
        rsi = curr["rsi"]
        stoch_k = curr.get("stoch_k", 50)
        chop_idx = curr["chop_idx"]

        # Regime Logic
        market_regime = "NEUTRAL"
        if chop_idx > CHOP_THRESHOLD_RANGE:
            market_regime = "RANGE"
        elif chop_idx < CHOP_THRESHOLD_TREND:
            market_regime = "TREND"

        # Session/Regime Filter (Skip if not recovering late signal)
        if not is_late_recovery:
            if "REVERSION" in session_info["types"] and "TREND" not in session_info["types"]:
                if market_regime == "TREND":
                    return None  # Don't trade trend in Asia

        # Structure & Patterns
        htf_trend = await self._get_htf_trend(symbol, provider)
        pattern_recognizer = PatternRecognizer()
        structure = pattern_recognizer.check_market_structure(df)
        patterns = pattern_recognizer.analyze_patterns(df, structure)
        pattern_signal = patterns[0] if patterns else None

        # Support & Resistance Context
        sr_levels = TechnicalAnalyzer.get_support_resistance_levels(df)
        dist_threshold = curr["atr"] * 0.5
        near_support = any(abs(curr["close"] - lvl) < dist_threshold for lvl in sr_levels if lvl < curr["close"])
        near_resistance = any(abs(curr["close"] - lvl) < dist_threshold for lvl in sr_levels if lvl > curr["close"])

        bullish_momentum = rsi > 50 and stoch_k < 80  # Not overbought yet
        bearish_momentum = rsi < 50 and stoch_k > 20  # Not oversold yet
        oversold_condition = stoch_k < 20 or rsi < 30
        overbought_condition = stoch_k > 80 or rsi > 70

        context_bias = "NEUTRAL"
        if structure == "BULL":
            if near_support and oversold_condition:
                context_bias = "LONG_BOUNCE"  # High Probability
            elif bullish_momentum:
                context_bias = "LONG_CONTINUATION"
        elif structure == "BEAR":
            if near_resistance and overbought_condition:
                context_bias = "SHORT_REJECTION"  # High Probability
            elif bearish_momentum:
                context_bias = "SHORT_CONTINUATION"
        elif structure == "RANGE":
            if near_support and oversold_condition:
                context_bias = "LONG_RANGE"
            elif near_resistance and overbought_condition:
                context_bias = "SHORT_RANGE"

        # --- 7. Strategy Routing & Signal Generation ---
        fake_risk_penalty = 1.0  # Default

        if not is_late_recovery:
            strat_signal = None
            if symbol in FOREX_SYMBOLS:
                # BB Reversion
                if market_regime == "RANGE" and curr["bb_slope"] < (curr["atr"] * 0.1):
                    strat_signal = self.strategy_analyzer._fx_bb_reversion(curr)

                # Continuation / Trend
                if "CONTINUATION" in context_bias:
                    strat_signal = self.strategy_analyzer._fx_volatility_breakout(curr, df)
                elif market_regime in ["TREND", "NEUTRAL"] and "TREND" in session_info["types"]:
                    strat_signal = self.strategy_analyzer._fx_fvg_entry(curr, df)
                    if not strat_signal:
                        strat_signal = self.strategy_analyzer._fx_golden_pullback(curr, htf_trend)

                # Breakout (Only if allowed hour)
                if not strat_signal and "BREAKOUT" in session_info["types"]:
                    strat_signal = self.strategy_analyzer._fx_volatility_breakout(curr, df)

                # Fallback FVG
                if not strat_signal:
                    strat_signal = self.strategy_analyzer._fx_fvg_entry(curr, df)
            else:
                strat_signal = self.strategy_analyzer.analyze_crypto(curr, df, patterns)

            # --- 8. Confluence & Conflict Checks ---
            if strat_signal and pattern_signal:
                if strat_signal["direction"] == pattern_signal["direction"]:
                    strat_signal["confidence"] += 10
                    strat_signal["strategy"] += f" + {pattern_signal['pattern']}"
                    final_signal_candidate = strat_signal
                else:
                    # Conflict Logic
                    s_name = strat_signal["strategy"].lower()
                    s_type = "REVERSION" if "reversion" in s_name or "divergence" in s_name else "TREND"
                    p_name = pattern_signal["pattern"].lower()
                    p_type = "TREND" if "flag" in p_name else "REVERSION"

                    if s_type == p_type:
                        self._log_once(f"conflict_{symbol}", f"Skipping {symbol}: Strategy/Pattern Conflict")
                        return None
                    else:
                        strat_signal["confidence"] -= 15
                        strat_signal["strategy"] += f" - {pattern_signal['pattern']} (Conflict)"
                        final_signal_candidate = strat_signal
            elif strat_signal:
                final_signal_candidate = strat_signal
            elif pattern_signal:
                pattern_signal["strategy"] = pattern_signal["pattern"]
                final_signal_candidate = pattern_signal

        if not final_signal_candidate:
            return None

        # --- 9. Signal Vetting (Structure & Crypto) ---
        # We skip deep vetting for Late Bloomers as they are already vetted survivors
        if not is_late_recovery:
            penalty_score = 0
            is_reversion = "reversion" in final_signal_candidate["strategy"].lower()

            if not is_reversion:
                if final_signal_candidate["direction"] == "LONG" and structure == "BEAR" and "LONG" not in context_bias:
                    penalty_score += 15
                if (
                    final_signal_candidate["direction"] == "SHORT"
                    and structure == "BULL"
                    and "SHORT" not in context_bias
                ):
                    penalty_score += 15

            # Bitcoin Correlation Veto
            if symbol in CRYPTO_SYMBOLS and symbol != "BTCUSDm":
                if "BTCUSDm" not in self.htf_cache:
                    await self._get_htf_trend("BTCUSDm", provider)

                btc_trend_data = self.htf_cache.get("BTCUSDm")
                if btc_trend_data:
                    btc_trend = btc_trend_data["trend"]
                    if btc_trend == "BEAR" and final_signal_candidate["direction"] == "LONG":
                        penalty_score += 20
                    if btc_trend == "BULL" and final_signal_candidate["direction"] == "SHORT":
                        penalty_score += 20

            # Fakeout Detector
            breakout_detector = FakeBreakoutDetector()
            fake_analysis = breakout_detector.analyze(df)

            if fake_analysis["risk_score"] >= 50:
                # If reason is Low Vol Breakout, cache it for 15 mins
                if "Low Vol Breakout" in fake_analysis["reasons"]:
                    self.low_vol_candidates[symbol] = {
                        "time": time.time(),
                        "signal": final_signal_candidate,
                        "entry": curr["close"],
                    }
                    return None
                else:
                    penalty_score += 25
            elif fake_analysis["risk_score"] >= 30:
                fake_risk_penalty = 0.5
                penalty_score += 10

            # Apply Penalties
            final_signal_candidate["confidence"] -= penalty_score

        # --- 10. ML Prediction (Feature Extraction) ---
        now = datetime.now()
        recent_df = df.iloc[-60:]
        pivots = recent_df[recent_df["high"] == recent_df["high"].rolling(10, center=True).max()]["high"]
        last_pivot = pivots.iloc[-1] if not pivots.empty else curr["high"]
        dist_to_pivot = abs(curr["close"] - last_pivot) / curr["close"]

        range_len = curr["high"] - curr["low"]
        wick_ratio = (curr["high"] - curr["close"]) / range_len if range_len > 0 else 0.0
        dist_to_vwap = (curr["close"] - curr["vwap"]) / curr["vwap"] if curr["vwap"] != 0 else 0.0
        day_norm_val = 0.0 if symbol in CRYPTO_SYMBOLS else now.weekday() / 6.0

        rolling_acc = 0.5  # Default neutral
        if self.db_manager:
            rolling_acc = await self.db_manager.get_pair_performance(symbol)

        avg_atr_24 = avg_atr
        atr_ratio = curr["atr"] / (avg_atr_24 + 1e-9)
        vol_sma_ratio = curr["volume"] / (curr["vol_sma"] + 1e-9)
        recent_range_std = recent_df["high"].sub(recent_df["low"]).tail(20).std()

        features = {
            "rsi": rsi,
            "adx": curr["adx"],
            "atr": curr["atr"],
            "atr_ratio": atr_ratio,
            "avg_atr_24": avg_atr_24,
            "ema_dist": (curr["close"] - curr["ema_50"]) / curr["close"],
            "bb_width": curr["bb_width"],
            "vol_ratio": vol_sma_ratio,
            "htf_trend": 1 if htf_trend == "BULL" else (-1 if htf_trend == "BEAR" else 0),
            "dist_to_pivot": dist_to_pivot,
            "hour_norm": now.hour / 24.0,
            "day_norm": day_norm_val,
            "wick_ratio": wick_ratio,
            "dist_ema200": (curr["close"] - curr["ema_200"]) / curr["close"],
            "volatility_ratio": volatility_ratio,
            "dist_to_vwap": dist_to_vwap,
            "rolling_acc": rolling_acc,
            "recent_range_std": recent_range_std if not math.isnan(recent_range_std) else 0.0,
        }

        # Predict
        nn_result = self.nn_brain.predict(features)

        # Shadow training logic
        is_shadow = False
        if nn_result["prob"] < 0.45:
            is_shadow = True
            final_signal_candidate["is_shadow"] = True
            final_signal_candidate["confidence"] = 40.0  # Force low confidence

        # Confidence Adjustment
        final_signal = await self._adjust_confidence(
            symbol, final_signal_candidate, nn_result["prob"], htf_trend, volatility_ratio
        )
        if final_signal["confidence"] < MIN_CONFIDENCE and not is_shadow:
            return None

        # Apply Fakeout Penalty to Risk Multiplier
        nn_result["risk_mult"] *= fake_risk_penalty

        # --- 11. Execution & Risk Sizing ---
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
                    if df.iloc[-1]["close"] > 0:
                        atr_pct = (df.iloc[-1]["atr"] / df.iloc[-1]["close"]) * 100
                        scored.append((sym, atr_pct))

        # Sort descending (Highest Volatility first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def record_trade_outcome(self, symbol: str, won: bool, pnl: float, excursion: float = 0.0, is_shadow: bool = False):
        """Called by Console after trade finishes."""
        # Update Learning
        if is_shadow:
            logger.debug(f"👻 Shadow trade outcome ignored for training: {symbol}")
            if symbol in self.active_features:
                del self.active_features[symbol]
            return

        if symbol in self.active_features:
            data_collector = DataCollector()
            data_collector.log_training_data(symbol, self.active_features[symbol], 1 if won else 0, pnl, excursion)
            del self.active_features[symbol]

    def register_active_trade(self, symbol: str):
        """Manually marks a symbol as active (used during recovery)."""
        if symbol not in self.active_features:
            self.active_features[symbol] = {}

    def set_context(self, balance: float, db: DatabaseManager):
        """Sets user balance and database manager."""
        self.user_balance_zar = balance
        self.db_manager = db
