import asyncio
import logging
import math
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional

from src.analysis.indicators import TechnicalAnalyzer
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

    async def _adjust_confidence(
        self,
        symbol: str,
        signal: dict,
        nn_prob: float,
        htf_trend: float,
        session_multiplier: float,
    ) -> Dict:
        """
        Calculates realistic confidence score.
        """
        base_conf = min(signal["confidence"], DEFAULT_MIN_CONFIDENCE)

        # 1. MTF Trend Alignment (1.0 = Bullish, -1.0 = Bearish)
        trend_bonus = (
            15
            if (htf_trend == 1.0 and signal["direction"] == "LONG")
            or (htf_trend == -1.0 and signal["direction"] == "SHORT")
            else 0
        )

        # 2. Historical Performance
        hist_win_rate = 0.5
        if self.db_manager:
            hist_win_rate = await self.db_manager.get_pair_performance(symbol)

        # Heavy Penalty for losers (< 40% win rate), Small Bonus for winners
        history_factor = -10 if hist_win_rate < 0.4 else (10 if hist_win_rate > 0.6 else 0)

        # 3. Neural Network Weighting
        nn_factor = (nn_prob - 0.5) * 40

        # Apply Advanced Session Multiplier
        final_conf = (base_conf + trend_bonus + history_factor + nn_factor) * session_multiplier
        final_conf = max(0.0, min(99.0, final_conf))

        signal["confidence"] = final_conf
        return signal

    async def _calculate_risk_metrics(
        self,
        symbol: str,
        signal: dict,
        curr: pd.Series,
        tick,
        info: dict,
        nn_result: dict,
        provider: DataProvider,
        active_fvgs: list,
        active_ifvgs: list,
        active_obs: list,
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
        point, tick_value = info["point"], info.get("trade_tick_value", 0)
        min_vol, max_vol, vol_step = info.get("min_vol", 0.01), info.get("max_vol", 100.0), info.get("vol_step", 0.01)
        digits = info.get("digits", 5)

        atr = float(curr.get("atr", 1.0))
        if atr <= 0:
            atr = 1.0

        # Entry Price Determination
        current_market_price = ask if signal["direction"] == "LONG" else bid
        order_type = signal.get("order_type", "MARKET")
        entry_price = signal.get("price", current_market_price)

        # Dynamic TP / SL calculation
        is_volatile_asset = self._is_high_volatility_symbol(symbol)
        sl_multiplier = 1.3 if (is_volatile_asset or atr > (curr["close"] * 0.005)) else 1.0
        sl_dist = max(atr * sl_multiplier, point * 50)

        # Use Dynamic SL if provided
        if "suggested_sl" in signal:
            suggested_dist = abs(signal["suggested_sl"] - entry_price)
            if (atr * 0.2) < suggested_dist < (atr * 6.0):
                sl_dist = suggested_dist

        # Excursion / Stop-Loss Cap Filter
        max_sl_cap = atr * 3.5
        if sl_dist > max_sl_cap:
            self._log_once(
                f"excursion_{symbol}",
                f"Skipping {symbol}: Setup requires SL ({sl_dist:.5f}) exceeding max ATR cap ({max_sl_cap:.5f})",
            )
            return None

        # Determine Opposing POIs to dynamically cap Take Profit
        opposing_pois = []
        all_zones = active_fvgs + active_ifvgs + active_obs
        if signal["direction"] == "LONG":
            opposing_pois = [p["low"] for p in all_zones if p["type"] == "BEAR" and p["low"] > entry_price]
        else:
            opposing_pois = [p["high"] for p in all_zones if p["type"] == "BULL" and p["high"] < entry_price]

        nearest_opposing_poi = None
        if opposing_pois:
            nearest_opposing_poi = min(opposing_pois) if signal["direction"] == "LONG" else max(opposing_pois)

        # Probability calibration hook
        prob = nn_result.get("prob", 0.5)
        pred_exit_atr = max(1.0, min(float(nn_result.get("pred_exit_atr", 2.0)), 6.0))
        tp_multiplier = max(pred_exit_atr, 3.0) if self.risk_pct > 3.0 else pred_exit_atr
        base_tp_dist = max(atr * tp_multiplier, sl_dist * 1.2)

        # Absolute Prices
        if signal["signal"] == "BUY":
            ref_price = entry_price if order_type == "LIMIT" else ask
            sl_price = ref_price - sl_dist

            # Cap TP just beneath the nearest opposing liquidity wall
            max_structural_tp = (
                (nearest_opposing_poi - point * 15) if nearest_opposing_poi else (ref_price + base_tp_dist)
            )
            tp3_price = min(ref_price + base_tp_dist, max_structural_tp)
            actual_tp_dist = tp3_price - ref_price

            tp1_price = ref_price + (actual_tp_dist * 0.33)
            tp2_price = ref_price + (actual_tp_dist * 0.66)
        else:
            ref_price = entry_price if order_type == "LIMIT" else bid
            sl_price = ref_price + sl_dist

            max_structural_tp = (
                (nearest_opposing_poi + point * 15) if nearest_opposing_poi else (ref_price - base_tp_dist)
            )
            tp3_price = max(ref_price - base_tp_dist, max_structural_tp)
            actual_tp_dist = ref_price - tp3_price

            tp1_price = ref_price - (actual_tp_dist * 0.33)
            tp2_price = ref_price - (actual_tp_dist * 0.66)

        # Verify true Risk/Reward against POI blockades
        rr = (actual_tp_dist / sl_dist) if sl_dist > 0 else 1.0
        MIN_RR = 1.5
        if rr < MIN_RR:
            self._log_once(f"rr_gate_{symbol}", f"Skipping {symbol}: Poor Risk/Reward ratio ({rr:.2f}) < {MIN_RR}")
            return None

        expected_ev = prob * rr - (1.0 - prob)
        if expected_ev < -0.25:
            self._log_once(f"ev_gate_{symbol}", f"Skipping {symbol}: Negative EV ({expected_ev:.2f})")
            return None

        # Lot Sizing based on the final SL distance
        risk_mult = nn_result.get("risk_mult", 1.0)
        target_risk_account = self.user_balance_account * ((self.risk_pct * risk_mult) / 100)
        points_risk = sl_dist / point
        risk_per_lot = points_risk * tick_value
        if risk_per_lot == 0:
            return None

        lots = target_risk_account / risk_per_lot
        try:
            steps = math.floor(lots / vol_step)
            lots = steps * vol_step
        except:
            lots = round(lots / vol_step) * vol_step

        lots = round(max(min_vol, min(lots, max_vol, self.max_lot)), 2)
        actual_risk_account = risk_per_lot * lots

        # Convert Account Balance to USD for tier checks if necessary
        balance_in_usd = (
            self.user_balance_account if self.currency == "USD" else self.user_balance_account / usdzar_rate
        )

        # Safety Cap
        max_allowed_pct = get_account_risk_caps(balance_in_usd, self.currency)

        # Absolute hard cap in ZAR
        max_allowed_val = self.user_balance_account * (max_allowed_pct / 100.0)

        # Check if risk exceeds cap
        if actual_risk_account > max_allowed_val:
            while actual_risk_account > max_allowed_val and lots > min_vol:
                lots -= vol_step
                actual_risk_account = risk_per_lot * lots

            # Final check after reduction
            lots = round(lots, 2)

            if lots <= min_vol and actual_risk_account > max_allowed_val * 1.2:
                if is_volatile_asset and (actual_risk_account <= self.user_balance_account * 0.20):
                    pass  # Micro-Account Override
                else:
                    self._log_once(
                        f"risk_{symbol}",
                        f"Skipping {symbol}: Risk (R{actual_risk_account:.2f}) > Cap (R{max_allowed_val * 1.2:.2f})",
                        logging.DEBUG,
                    )
                    return None

        # Profit Calculation
        profit_account = (actual_tp_dist / point) * tick_value * lots
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
                "digits": digits,
                "lot_size": lots,
                "risk_zar": round(actual_risk_zar, 2),
                "profit_zar": round(profit_zar, 2),
                "risk_account": round(actual_risk_account, 2),
                "currency": self.currency,
                "tick_value": tick_value,
                "point": point,
                "atr": atr,
                "is_high_risk": is_volatile_asset,
                "order_type": order_type,
            }
        )
        return signal

    async def _get_htf_trend(self, symbol: str, provider: DataProvider) -> float:
        """Fetches trend from HTF using TechnicalAnalyzer strict logic. Returns float."""
        htf_tf = "1h"

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
        trend = 0.0
        if klines:
            df = TechnicalAnalyzer.calculate_indicators(pd.DataFrame(klines))
            trend = TechnicalAnalyzer.get_htf_trend(df)

        self.htf_cache[symbol] = {"trend": trend, "time": now}
        return trend

    def _get_session_status(self, symbol: str) -> Dict:
        """Determines active session and assigns volatility multipliers per symbol class."""
        now = datetime.now()
        hour = now.hour

        # Check against mapped SAST sessions
        is_asian = SESSION_CONFIG["ASIAN_START"] <= hour < SESSION_CONFIG["ASIAN_END"]
        is_london = SESSION_CONFIG["LONDON_START"] <= hour < SESSION_CONFIG["LONDON_END"]
        is_ny = SESSION_CONFIG["NY_START"] <= hour < SESSION_CONFIG["NY_END"]

        active_session = "NONE"
        if is_ny:
            active_session = "NY"
        elif is_london:
            active_session = "LONDON"
        elif is_asian:
            active_session = "ASIAN"

        # Apply Session-Specific Volatility Boosts for target instruments
        session_multiplier = 1.0

        # Catch specific London pairs first
        if any(x in symbol for x in ["EUR", "GBP"]):
            if active_session == "LONDON":
                session_multiplier = 1.15
        # Then catch NY-specific pairs and indices
        elif any(x in symbol for x in ["XAU", "US30", "NAS", "USD", "BTC", "ETH"]):
            if active_session in ["NY", "LONDON"]:
                session_multiplier = 1.25
        # Asian pairs
        elif any(x in symbol for x in ["JPY", "AUD", "NZD"]):
            if active_session == "ASIAN":
                session_multiplier = 1.15

        allow_trade = is_asian or is_london or is_ny
        return {"allow_trade": allow_trade, "active_session": active_session, "multiplier": session_multiplier}

    def _log_once(self, key: str, message: str, level=logging.INFO):
        """Prevents log spamming for the same event within 5 minutes."""
        now = time.time()
        if key in self._log_throttle and now - self._log_throttle[key] < 300:
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
        session_info = self._get_session_status(symbol)
        if not session_info.get("allow_trade", True):
            return None

        # 4. Data Preparation
        df = await asyncio.to_thread(self.prepare_data, klines)
        if df is None or df.empty:
            return None

        curr = df.iloc[-1]

        # Centralize highly used row metrics
        atr = float(curr.get("atr", 1.0))
        if atr <= 0:
            return None
        close_price = curr["close"]

        # 5. Spread Check
        symbol_info = await provider.get_symbol_info(symbol)
        if not symbol_info:
            return None

        point = symbol_info.get("point", 0.00001)
        spread_info = await provider.get_spread(symbol)
        spread_price = spread_info.get("spread", 0.0) * point

        max_allowed_spread = max(atr * 2.0, point * 50)
        if spread_price > max_allowed_spread:
            return None

        # 6. Unified SMC POI Detection
        active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.extract_active_pois(df)

        # 7. Check Loss Cooldown (Database)
        try:
            if self.db_manager and await self.db_manager.check_recent_loss(symbol):
                self._log_once(f"loss_{symbol}", f"Skipping {symbol}: Loss Cooldown Active")
                return None
        except:
            pass

        # 8. Structural & HTF Analysis
        structure_info = TechnicalAnalyzer.detect_structure(df)
        htf_trend = await self._get_htf_trend(symbol, provider)

        # Prepare Daily Levels for Sweep Logic
        df_temp = df.copy()
        df_temp["date"] = pd.to_datetime(df_temp["time"], unit="s").dt.date
        prev_days = df_temp[df_temp["date"] < df_temp["date"].iloc[-1]]

        pdl, pdh = None, None
        if not prev_days.empty:
            last_day = prev_days["date"].iloc[-1]
            yesterday_df = prev_days[prev_days["date"] == last_day]
            pdh, pdl = yesterday_df["high"].max(), yesterday_df["low"].min()

        daily_levels = {"pdh": pdh, "pdl": pdl}

        is_liquidity_swept, sweep_depth_atr = TechnicalAnalyzer.detect_liquidity_sweeps(
            curr, structure_info, daily_levels
        )

        # Route through Strategy Engine
        final_signal = self.strategy_analyzer.analyze_router(
            curr, active_fvgs, active_ifvgs, active_obs, structure_info, is_liquidity_swept, htf_trend
        )

        if not final_signal:
            return None

        # Set volatility flag for Telegram UI
        final_signal["is_high_risk"] = is_volatile_pair

        # Feature Extraction calculations

        # A. Determine Killzone Session Logic
        hour = datetime.now().hour
        active_killzone = 0.0
        if SESSION_CONFIG["ASIAN_START"] <= hour < SESSION_CONFIG["ASIAN_END"]:
            active_killzone = 1.0
        elif SESSION_CONFIG["LONDON_START"] <= hour < SESSION_CONFIG["LONDON_END"]:
            active_killzone = 2.0
        elif SESSION_CONFIG["NY_START"] <= hour < SESSION_CONFIG["NY_END"]:
            active_killzone = 3.0

        # B. Distance to Nearest Active POI & Mitigation Freshness Tracking
        all_pois = active_fvgs + active_ifvgs + active_obs
        dist_nearest_poi_atr = 0.0
        mitigation_count = 0

        if all_pois:
            # Find the absolute closest POI boundary (High or Low edge) to Current Price
            nearest_poi = min(all_pois, key=lambda x: min(abs(x["high"] - close_price), abs(x["low"] - close_price)))
            raw_distance = min(abs(nearest_poi["high"] - close_price), abs(nearest_poi["low"] - close_price))

            dist_nearest_poi_atr = raw_distance / atr if atr > 0 else 0.0
            mitigation_count = nearest_poi.get("mitigations", 0)

        # C. Feature Construction (Types are cleanly managed, formatting occurs in collector)
        features = {
            "is_htf_aligned": htf_trend,
            "is_liquidity_swept": float(is_liquidity_swept),
            "is_in_fvg": 1.0 if any(f["low"] <= close_price <= f["high"] for f in active_fvgs) else 0.0,
            "is_in_ifvg": 1.0 if any(i_f["low"] <= close_price <= i_f["high"] for i_f in active_ifvgs) else 0.0,
            "is_in_orderblock": 1.0 if any(o["low"] <= close_price <= o["high"] for o in active_obs) else 0.0,
            "structural_break": structure_info.get("structural_break", 0.0),
            "active_killzone": active_killzone,
            "distance_to_poi": dist_nearest_poi_atr,
            "pd_array_status": structure_info.get("pd_array", 0.5),
            "mitigation_count": float(mitigation_count),
            "sweep_depth_atr": sweep_depth_atr,
        }

        # Inference from read-only bundled model
        nn_result = self.nn_brain.predict(features)

        # Confidence Adjustment
        final_signal = await self._adjust_confidence(
            symbol, final_signal, nn_result["prob"], htf_trend, session_info["multiplier"]
        )

        required_conf = self.min_confidence
        if is_volatile_pair:
            required_conf -= 5.0

        if final_signal["confidence"] < required_conf:
            return None

        # 9. Execution & Final Risk Sizing
        tick = await provider.get_current_tick(symbol)
        if not tick:
            return None

        result = await self._calculate_risk_metrics(
            symbol, final_signal, curr, tick, symbol_info, nn_result, provider, active_fvgs, active_ifvgs, active_obs
        )
        if result:
            self.signal_history[symbol] = time.time()
            self.active_features[symbol] = features
            return result

        return None

    def prepare_data(self, klines: list) -> Optional[pd.DataFrame]:
        """Prepares DataFrame with Indicators for Analysis."""
        try:
            df = pd.DataFrame(klines)
            if df.empty:
                return None
            df = df.sort_values("time").reset_index(drop=True)
            df = TechnicalAnalyzer.calculate_indicators(df)
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
