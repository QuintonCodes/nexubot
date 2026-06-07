import collections
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
    CANDLE_LIMIT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MAX_LOT,
    DEFAULT_RISK_PCT,
    HIGH_VOLATILITY_IDENTIFIERS,
    MIN_RR,
    PAIR_SIGNAL_COOLDOWN,
    SESSION_CONFIG,
    TIMEFRAME,
    get_account_risk_caps,
)

logger = logging.getLogger(__name__)


class AITradingEngine:
    """AI Trading Engine for analyzing market data and generating trade signals."""

    def __init__(self):
        self.strategy_analyzer = StrategyAnalyzer()
        self.nn_brain = NeuralPredictor(auto_load=True)
        self.collector = DataCollector()

        self._acct_cache = {"data": None, "time": 0}
        self._log_throttle = collections.OrderedDict()
        self._poi_cache = collections.OrderedDict()
        self.htf_cache = collections.OrderedDict()
        self.signal_history = collections.OrderedDict()

        self.active_features = {}
        self.db_manager = None
        self.user_balance_account = 0.0
        self.currency = "USD"

        # Dynamic Configuration
        self.risk_pct = DEFAULT_RISK_PCT
        self.max_lot = DEFAULT_MAX_LOT
        self.min_confidence = DEFAULT_MIN_CONFIDENCE

    def _calculate_structural_rr(
        self,
        signal: dict,
        close_price: float,
        atr: float,
        point: float,
        is_volatile_asset: bool,
        active_ifvgs: list,
        active_obs: list,
    ) -> float:
        """Calculates the baseline structural Risk/Reward without ML influence for feature encoding."""
        sl_multiplier = 1.3 if (is_volatile_asset or atr > (close_price * 0.005)) else 1.0
        sl_dist = max(atr * sl_multiplier, point * 50)

        if "suggested_sl" in signal:
            suggested_dist = abs(signal["suggested_sl"] - close_price)
            if (atr * 0.2) < suggested_dist < (atr * 6.0):
                sl_dist = suggested_dist

        hard_blockades = active_ifvgs + [ob for ob in active_obs if ob.get("tier") == "MAJOR"]

        if signal["direction"] == "LONG":
            opposing = [p["low"] for p in hard_blockades if p["type"] == "BEAR" and p["low"] > close_price]
            nearest = min(opposing) if opposing else None
            base_tp_dist = max(atr * 4.0, sl_dist * MIN_RR)
            max_tp = (nearest - point * 15) if nearest else (close_price + base_tp_dist)
            actual_tp_dist = min(close_price + base_tp_dist, max_tp) - close_price
        else:
            opposing = [p["high"] for p in hard_blockades if p["type"] == "BULL" and p["high"] < close_price]
            nearest = max(opposing) if opposing else None
            base_tp_dist = max(atr * 4.0, sl_dist * MIN_RR)
            max_tp = (nearest + point * 15) if nearest else (close_price - base_tp_dist)
            actual_tp_dist = close_price - max(close_price - base_tp_dist, max_tp)

        return (actual_tp_dist / sl_dist) if sl_dist > 0 else 1.0

    async def _calculate_risk_metrics(
        self,
        symbol: str,
        signal: dict,
        curr: pd.Series,
        tick,
        info: dict,
        nn_result: dict,
        provider: DataProvider,
        active_ifvgs: list,
        active_obs: list,
    ) -> Optional[Dict]:
        """Calculates dynamic SL/TP levels, lot sizing, and risk metrics based on current market context and NN output."""
        # 1. Fetch Account Currency & Live Rates
        acct_summary = await provider.get_account_summary()
        self.currency = acct_summary.get("currency", "USD")
        self.user_balance_account = acct_summary.get("balance", 0.0)

        if self.user_balance_account <= 0:
            return None

        usdzar_rate = await provider.get_usdzar_rate() if self.currency == "USD" else 1.0

        ask, bid = tick.ask, tick.bid
        point, tick_value = info["point"], info.get("trade_tick_value", 0)
        min_vol, max_vol, vol_step = info.get("min_vol", 0.01), info.get("max_vol", 100.0), info.get("vol_step", 0.01)
        digits = info.get("digits", 5)

        # Dynamically scale the minimum ATR fallback using the asset's point value
        min_atr = point * 10
        atr = float(curr.get("atr", min_atr))
        if atr <= 0 or pd.isna(atr):
            atr = min_atr

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
            sl_on_correct_side = (signal["direction"] == "LONG" and signal["suggested_sl"] < entry_price) or (
                signal["direction"] == "SHORT" and signal["suggested_sl"] > entry_price
            )
            if sl_on_correct_side and (atr * 0.2) < suggested_dist < (atr * 6.0):
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
        hard_blockades = active_ifvgs + [ob for ob in active_obs if ob.get("tier") == "MAJOR"]
        if signal["direction"] == "LONG":
            opposing_pois = [p["low"] for p in hard_blockades if p["type"] == "BEAR" and p["low"] > entry_price]
            nearest_opposing_poi = min(opposing_pois) if opposing_pois else None
        else:
            opposing_pois = [p["high"] for p in hard_blockades if p["type"] == "BULL" and p["high"] < entry_price]
            nearest_opposing_poi = max(opposing_pois) if opposing_pois else None

        # Probability calibration hook
        prob = nn_result.get("prob", 0.5)
        pred_exit_atr = max(MIN_RR, min(float(nn_result.get("pred_exit_atr", 2.5)), 6.0))
        tp_multiplier = max(pred_exit_atr, 3.5) if self.risk_pct > 3.0 else pred_exit_atr
        base_tp_dist = max(atr * tp_multiplier, sl_dist * MIN_RR)

        # Absolute Prices
        if signal["signal"] == "BUY":
            ref_price = entry_price if order_type == "LIMIT" else ask
            sl_price = ref_price - sl_dist

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
        if rr < MIN_RR:
            self._log_once(f"rr_gate_{symbol}", f"Skipping {symbol}: Poor Risk/Reward ratio ({rr:.2f}) < {MIN_RR}")
            return None

        expected_ev = prob * rr - (1.0 - prob)
        if expected_ev < 0.0:
            self._log_once(
                f"ev_gate_{symbol}",
                f"Skipping {symbol}: Negative EV ({expected_ev:.2f}) at prob={prob:.2f}, rr={rr:.2f}",
            )
            return None

        # Lot Sizing based on the final SL distance
        risk_mult = nn_result.get("risk_mult", 1.0)
        target_risk_account = self.user_balance_account * ((self.risk_pct * risk_mult) / 100)
        points_risk = sl_dist / point
        risk_per_lot = points_risk * tick_value
        if risk_per_lot == 0:
            return None

        lots = target_risk_account / risk_per_lot
        balance_in_usd = (
            self.user_balance_account if self.currency == "USD" else self.user_balance_account / usdzar_rate
        )
        max_allowed_pct = get_account_risk_caps(balance_in_usd, self.currency)
        max_allowed_val = self.user_balance_account * (max_allowed_pct / 100.0)

        actual_risk_account = risk_per_lot * lots
        if actual_risk_account > max_allowed_val:
            lots = min(lots, max_allowed_val / risk_per_lot)

        try:
            steps = math.floor(lots / vol_step)
            lots = steps * vol_step
        except Exception:
            lots = round(lots / vol_step) * vol_step

        lots = round(max(min_vol, min(lots, max_vol, self.max_lot)), 2)
        actual_risk_account = risk_per_lot * lots

        # Check if risk exceeds cap
        if lots <= min_vol and actual_risk_account > max_allowed_val * 1.2:
            if not (is_volatile_asset and actual_risk_account <= self.user_balance_account * 0.20):
                self._log_once(f"risk_{symbol}", f"Skipping {symbol}: Risk exceeds cap", logging.DEBUG)
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
                "profit_account": round(profit_account, 2),
                "currency": self.currency,
                "tick_value": tick_value,
                "point": point,
                "atr": atr,
                "is_high_risk": is_volatile_asset,
                "order_type": order_type,
                "realized_rr": rr,
            }
        )
        return signal

    def _enforce_cache_limits(self, cache_dict: collections.OrderedDict, max_size: int = 200):
        """Safely pops the oldest items if caches grow too large."""
        while len(cache_dict) > max_size:
            cache_dict.popitem(last=False)

    async def _get_cached_account(self, provider: DataProvider) -> dict:
        now = time.time()
        if self._acct_cache["data"] and now - self._acct_cache["time"] < 30:
            return self._acct_cache["data"]
        data = await provider.get_account_summary()
        self._acct_cache = {"data": data, "time": now}
        return data

    async def _get_htf_trend(self, symbol: str, provider: DataProvider) -> float:
        """Determines the HTF trend direction for the given symbol using cached data when possible."""
        now = time.time()
        self._enforce_cache_limits(self.htf_cache)
        cache_entry = self.htf_cache.get(symbol)

        if cache_entry and (now - cache_entry["time"]) < 3600:
            return cache_entry["trend"]

        klines = await provider.fetch_klines(symbol, "1h", 1000)
        trend, last_time = 0.0, 0
        if klines:
            df = TechnicalAnalyzer.calculate_indicators(pd.DataFrame(klines))
            trend = TechnicalAnalyzer.get_htf_trend(df)
            last_time = klines[-1]["time"]

        self.htf_cache[symbol] = {"trend": trend, "time": now, "last_h1_time": last_time}
        return trend

    def _log_once(self, key: str, message: str, level=logging.INFO):
        """Prevents log spamming for the same event within 5 minutes."""
        now = time.time()
        self._enforce_cache_limits(self._log_throttle, max_size=500)
        if key in self._log_throttle and now - self._log_throttle[key] < 300:
            return
        self._log_throttle[key] = now
        logger.log(level, message)

    def _is_high_volatility_symbol(self, symbol: str) -> bool:
        """Checks if symbol is considered High Volatility."""
        return any(x in symbol for x in HIGH_VOLATILITY_IDENTIFIERS)

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Checks if a symbol is on cooldown from last signal."""
        self._enforce_cache_limits(self.signal_history)
        if symbol in self.signal_history:
            elapsed = time.time() - self.signal_history[symbol]
            if elapsed < PAIR_SIGNAL_COOLDOWN:
                return True
        return False

    async def adjust_confidence(
        self,
        symbol: str,
        signal: dict,
        nn_prob: float,
        htf_trend: float,
        session_multiplier: float,
    ) -> Dict:
        """Adjusts the confidence score of a signal based on multiple factors:"""
        base_conf = signal["confidence"]
        direction = signal.get("direction", "")

        # 1. Trend Boost
        trend_bonus = (
            10.0
            if ((direction == "LONG" and htf_trend == 1.0) or (direction == "SHORT" and htf_trend == -1.0))
            else (-5.0 if htf_trend != 0.0 else 0.0)
        )

        # 2. Historical Performance
        hist_win_rate = await self.db_manager.get_pair_performance(symbol) if self.db_manager else 0.5

        # Heavy Penalty for losers (< 40% win rate), Small Bonus for winners
        history_factor = -10 if hist_win_rate < 0.4 else (10 if hist_win_rate > 0.6 else 0)

        # 3. Neural Network Weighting
        nn_factor = (nn_prob - 0.5) * 40

        # Apply Advanced Session Multiplier
        final_conf = (base_conf + trend_bonus + history_factor + nn_factor) * session_multiplier
        final_conf = max(0.0, min(99.0, final_conf))

        signal["confidence"] = final_conf
        return signal

    async def analyze_for_report(self, symbol: str, provider: DataProvider) -> Optional[str]:
        """Provides the detailed breakdown logic extracted from cmd_analyze to enforce DRY."""
        snapshot = await self.compute_market_snapshot(symbol, provider)
        if not snapshot:
            return None

        curr, structure, htf_trend = snapshot["curr"], snapshot["structure"], snapshot["htf_trend"]
        active_fvgs, active_ifvgs, active_obs = (
            snapshot["active_fvgs"],
            snapshot["active_ifvgs"],
            snapshot["active_obs"],
        )
        is_liquidity_swept, sweep_depth_atr, killzone_name = (
            snapshot["is_liquidity_swept"],
            snapshot["sweep_depth_atr"],
            snapshot["killzone_name"],
        )

        final_signal = self.strategy_analyzer.analyze_router(
            curr, active_fvgs, active_ifvgs, active_obs, structure, is_liquidity_swept
        )

        if not final_signal:
            final_signal = {
                "signal": "BUY" if structure["structure"] == "BULL" else "SELL",
                "direction": "LONG" if structure["structure"] == "BULL" else "SHORT",
                "confidence": 60.0,
                "structural_break_val": structure.get("structural_break", 0.0),
            }
        else:
            final_signal["structural_break_val"] = structure.get("structural_break", 0.0)

        symbol_info = await provider.get_symbol_info(symbol)
        point = symbol_info.get("point", 0.00001) if symbol_info else 0.00001
        is_volatile_pair = self._is_high_volatility_symbol(symbol)

        # 1. Provide the structural pre-ML RR feature calculation
        structural_rr = self._calculate_structural_rr(
            final_signal, snapshot["close_price"], snapshot["atr"], point, is_volatile_pair, active_ifvgs, active_obs
        )

        raw_features = TechnicalAnalyzer.compile_features(
            curr=curr,
            signal_direction=final_signal["direction"],
            active_fvgs=active_fvgs,
            active_ifvgs=active_ifvgs,
            active_obs=active_obs,
            structure_info=structure,
            is_liquidity_swept=is_liquidity_swept,
            sweep_depth_atr=sweep_depth_atr,
            atr=snapshot["atr"],
            htf_trend=htf_trend,
            rr_at_entry=structural_rr,
        )

        engineered_features = self.collector.engineer_features(raw_features)
        nn_result = self.nn_brain.predict(engineered_features, strategy=final_signal.get("strategy", "Unknown"))
        session_info = self.get_session_status()

        adjusted_signal = await self.adjust_confidence(
            symbol, final_signal, nn_result["prob"], htf_trend, session_info["multiplier"]
        )

        win_prob = adjusted_signal["confidence"]
        trend_bonus = (
            10
            if (
                (final_signal["direction"] == "LONG" and htf_trend == 1.0)
                or (final_signal["direction"] == "SHORT" and htf_trend == -1.0)
            )
            else 0
        )

        htf_text = "BULLISH 🐂" if htf_trend == 1.0 else ("BEARISH 🐻" if htf_trend == -1.0 else "FLAT ➖")
        pd_status = structure.get("pd_array", 0.5)
        pd_text = "Equilibrium"
        if pd_status <= 0.4:
            pd_text = f"Discount 🟢 ({pd_status:.2f})"
        elif pd_status >= 0.6:
            pd_text = f"Premium 🔴 ({pd_status:.2f})"

        struct_break = final_signal.get("structural_break_val", 0.0)
        break_text = "None"
        if struct_break == 1.0:
            break_text = "Bullish BOS 📈"
        elif struct_break == -1.0:
            break_text = "Bearish BOS 📉"
        elif struct_break == 2.0:
            break_text = "Bullish CHoCH 🔄"
        elif struct_break == -2.0:
            break_text = "Bearish CHoCH 🔄"

        sweep_text = "None"
        if is_liquidity_swept == 3:
            sweep_text = f"Daily/Asian High/Low Swept 🔥 ({sweep_depth_atr:.1f} ATR)"
        elif is_liquidity_swept == 2:
            sweep_text = f"Major Swing/Psych Swept ⚡ ({sweep_depth_atr:.1f} ATR)"
        elif is_liquidity_swept == 1:
            sweep_text = f"Internal Pivot Swept ({sweep_depth_atr:.1f} ATR)"

        vwap_trend = "BULL" if snapshot["close_price"] > curr.get("vwap", 0) else "BEAR"

        if win_prob > 80:
            ai_thought = "Highly favorable setup forming. Aligning with HTF institutional flow and Killzone momentum."
        elif win_prob > 60:
            ai_thought = "Viable environment. Awaiting clear liquidity sweep or decisive structural confirmation."
        else:
            ai_thought = "Poor conditions. High probability of chop or false breakouts. Avoiding."

        return (
            f"🧠 *Deep SMC Analysis: {symbol}*\n\n"
            f"🌍 *Session Profile:* ({killzone_name})\n"
            f"📊 *HTF Flow (1H):* {htf_text}\n"
            f"🧭 *Local Flow ({TIMEFRAME}):* {structure['structure']} | Break: {break_text}\n"
            f"📏 *PD Array:* Price in {pd_text}\n"
            f"💧 *Liquidity Profile:* {len(active_fvgs)} FVGs | {len(active_ifvgs)} IFVGs | {len(active_obs)} OBs\n"
            f"🎯 *Nearest POI Status:* {raw_features.get('distance_to_poi', 0.0):.1f} ATR Away\n"
            f"🧹 *Sweep Status:* {sweep_text}\n"
            f"🌊 *VWAP State:* {vwap_trend}\n\n"
            f"⚡ *Momentum & Vol:* Vol Ratio {engineered_features.get('vol_ratio', 1.0):.2f}x | Body Ratio {engineered_features.get('body_ratio', 0.0):.2f}\n\n"
            f"🤖 *Neural Output:*\n"
            f"• _Trend Alignment:_ {'Aligned ✅' if trend_bonus > 0 else 'Counter ⚠️'}\n"
            f"• _Calculated AI Confidence:_ *{win_prob:.1f}%*\n"
            f"• _AI Conclusion:_ {ai_thought}"
        )

    async def analyze_market(self, symbol: str, klines: list, provider: DataProvider) -> Optional[Dict]:
        """Main analysis function that processes market data and generates trade signals with dynamic risk management."""
        # Volatility check
        is_volatile_pair = self._is_high_volatility_symbol(symbol)

        # Cooldown Checks
        if self._is_on_cooldown(symbol) or symbol in self.active_features:
            return None

        # Session Info
        session_info = self.get_session_status()
        if not session_info.get("allow_trade", True):
            return None

        # Check Loss Cooldown (Database)
        try:
            if self.db_manager and await self.db_manager.check_recent_loss(symbol):
                self._log_once(f"loss_{symbol}", f"Skipping {symbol}: Loss Cooldown Active")
                return None
        except Exception as e:
            logger.warning(f"Database check failed for {symbol}: {e}. Failing closed.")
            return None

        # Defer intensive logic directly to the generalized snapshot creator
        snapshot = await self.compute_market_snapshot(symbol, provider, klines)
        if not snapshot:
            return None

        curr = snapshot["curr"]

        # Spread Check
        symbol_info = await provider.get_symbol_info(symbol)
        if not symbol_info:
            return None

        # Route through Strategy Engine
        final_signal = self.strategy_analyzer.analyze_router(
            curr,
            snapshot["active_fvgs"],
            snapshot["active_ifvgs"],
            snapshot["active_obs"],
            snapshot["structure"],
            snapshot["is_liquidity_swept"],
        )

        if not final_signal:
            return None

        final_signal["is_high_risk"] = is_volatile_pair
        final_signal["structural_break_val"] = snapshot["structure"].get("structural_break", 0.0)

        structural_rr = self._calculate_structural_rr(
            final_signal,
            curr["close"],
            snapshot["atr"],
            symbol_info.get("point", 0.00001),
            is_volatile_pair,
            snapshot["active_ifvgs"],
            snapshot["active_obs"],
        )

        # Apply raw features then pipe directly into feature engineer
        raw_features = TechnicalAnalyzer.compile_features(
            curr=curr,
            signal_direction=final_signal["direction"],
            active_fvgs=snapshot["active_fvgs"],
            active_ifvgs=snapshot["active_ifvgs"],
            active_obs=snapshot["active_obs"],
            structure_info=snapshot["structure"],
            is_liquidity_swept=snapshot["is_liquidity_swept"],
            sweep_depth_atr=snapshot["sweep_depth_atr"],
            atr=snapshot["atr"],
            htf_trend=snapshot["htf_trend"],
            rr_at_entry=structural_rr,
        )

        engineered_features = self.collector.engineer_features(raw_features)

        # Inference from read-only bundled model
        nn_result = self.nn_brain.predict(engineered_features, strategy=final_signal.get("strategy", "Unknown"))

        # Confidence Adjustment
        final_signal = await self.adjust_confidence(
            symbol, final_signal, nn_result["prob"], snapshot["htf_trend"], session_info["multiplier"]
        )

        required_conf = self.min_confidence - 5.0 if is_volatile_pair else self.min_confidence
        if final_signal["confidence"] < required_conf:
            return None

        # Execution & Final Risk Sizing
        tick = await provider.get_current_tick(symbol)
        if not tick:
            return None

        result = await self._calculate_risk_metrics(
            symbol,
            final_signal,
            curr,
            tick,
            symbol_info,
            nn_result,
            provider,
            snapshot["active_ifvgs"],
            snapshot["active_obs"],
        )
        if result:
            self.signal_history[symbol] = time.time()
            self.active_features[symbol] = {
                "features": engineered_features,
                "strategy": final_signal.get("strategy", "Unknown"),
            }
            return result

        return None

    async def compute_market_snapshot(
        self, symbol: str, provider: DataProvider, klines: Optional[list] = None
    ) -> Optional[dict]:
        """Computes a comprehensive snapshot of the market for a given symbol."""
        if not klines:
            klines = await provider.fetch_klines(symbol, TIMEFRAME, CANDLE_LIMIT)
            if not klines:
                return None

        df = self.prepare_data(klines)
        if df is None or df.empty:
            return None

        curr = df.iloc[-1]
        close_price = curr["close"]
        atr = float(curr.get("atr", 1.0))
        if atr <= 0:
            atr = 1.0

        records = df.to_dict("records")
        self._enforce_cache_limits(self._poi_cache)

        # Fix Med 6: O(1) Incremental POI Update
        cache = self._poi_cache.get(symbol, {})
        if cache and len(records) >= 3 and cache.get("last_time") == records[-2]["time"]:
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.update_pois(
                records[-3], records[-2], curr, cache["fvgs"], cache["ifvgs"], cache["obs"]
            )
        else:
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.extract_active_pois(df)

        MAX_POI_PER_TYPE = 50
        self._poi_cache[symbol] = {
            "last_time": curr["time"],
            "fvgs": active_fvgs[-MAX_POI_PER_TYPE:],
            "ifvgs": active_ifvgs[-MAX_POI_PER_TYPE:],
            "obs": active_obs[-MAX_POI_PER_TYPE:],
        }

        htf_trend = await self._get_htf_trend(symbol, provider)
        structure = TechnicalAnalyzer.detect_structure(df)

        curr_time = pd.to_datetime(curr["time"], unit="s")
        last_date = curr_time.date()
        prev_days = df[pd.to_datetime(df["time"], unit="s").dt.date < last_date]

        pdl, pdh = None, None
        if not prev_days.empty:
            yesterday_date = pd.to_datetime(prev_days.iloc[-1]["time"], unit="s").date()
            yesterday_df = prev_days[pd.to_datetime(prev_days["time"], unit="s").dt.date == yesterday_date]
            pdh, pdl = yesterday_df["high"].max(), yesterday_df["low"].min()

        daily_levels = {"pdh": pdh, "pdl": pdl}
        is_liquidity_swept, sweep_depth_atr = TechnicalAnalyzer.detect_liquidity_sweeps(curr, structure, daily_levels)

        hour = datetime.now().hour
        killzone_name = "Dead Zone"
        if SESSION_CONFIG["ASIAN_START"] <= hour < SESSION_CONFIG["ASIAN_END"]:
            killzone_name = "Asian Killzone"
        elif SESSION_CONFIG["LONDON_START"] <= hour < SESSION_CONFIG["LONDON_END"]:
            killzone_name = "London Killzone"
        elif SESSION_CONFIG["NY_START"] <= hour < SESSION_CONFIG["NY_END"]:
            killzone_name = "New York Killzone"

        return {
            "curr": curr,
            "close_price": close_price,
            "atr": atr,
            "htf_trend": htf_trend,
            "structure": structure,
            "active_fvgs": active_fvgs,
            "active_ifvgs": active_ifvgs,
            "active_obs": active_obs,
            "is_liquidity_swept": is_liquidity_swept,
            "sweep_depth_atr": sweep_depth_atr,
            "killzone_name": killzone_name,
        }

    def get_session_status(self) -> Dict:
        """Determines if current time falls within active trading sessions and applies session-based confidence multipliers."""
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

        allow_trade = is_asian or is_london or is_ny

        SESSION_MULTIPLIERS = {"NY": 1.05, "LONDON": 1.03, "ASIAN": 0.97, "NONE": 0.90}
        return {
            "allow_trade": allow_trade,
            "active_session": active_session,
            "multiplier": SESSION_MULTIPLIERS[active_session],
        }

    def prepare_data(self, klines: list) -> Optional[pd.DataFrame]:
        """Prepares DataFrame with indicators for analysis."""
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
        """Sorts symbols by ATR (Volatility)."""
        scored = []
        for sym in symbols:
            if sym in data_map:
                df = data_map[sym]
                if not df.empty and "atr" in df.columns and df.iloc[-1]["close"] > 0:
                    atr_pct = (df.iloc[-1]["atr"] / df.iloc[-1]["close"]) * 100
                    scored.append((sym, atr_pct))

        # Sort descending (Highest Volatility first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def record_trade_outcome(self, symbol: str, won: bool, pnl: float, excursion: float = 0.0) -> None:
        """Records a trade after it has been completed to update ML CSV"""
        logger.info(f"🏁 Trade Closed: {symbol} | PnL: {pnl} | Won: {won}")

        if symbol in self.active_features:
            data = self.active_features[symbol]
            features = data.get("features", {})
            strategy = data.get("strategy", "Unknown")

            # Skip logging corrupted empty feature sets from recovered offline trades
            if not features:
                logger.info(f"⏭️ Skipping ML log for {symbol}: Offline recovered trade has no stored features.")
                del self.active_features[symbol]
                return

            self.collector.log_training_data(symbol, strategy, features, 1 if won else 0, pnl, excursion)
            logger.info(f"💾 Live features for {symbol} saved to training data.")
            del self.active_features[symbol]

    def register_active_trade(self, symbol: str, strategy: str = "Unknown") -> None:
        """Manually marks a symbol as active (used during recovery)."""
        if symbol not in self.active_features:
            self.active_features[symbol] = {"strategy": strategy}

    def set_context(self, balance: float, db: DatabaseManager, currency: str = "USD") -> None:
        """Sets user balance and database manager."""
        self.user_balance_account = balance
        self.currency = currency
        self.db_manager = db
