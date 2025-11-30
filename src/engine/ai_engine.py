import os
import pickle
import logging
import time
import pandas as pd
from typing import Dict, List, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.patterns import FakeBreakoutDetector
from src.database.manager import DatabaseManager
from src.config import (
    MODEL_FILE, MIN_CONFIDENCE, DEFAULT_BALANCE_ZAR,
    RISK_PER_TRADE, LEVERAGE, HIGH_RISK_SYMBOLS,
    USD_ZAR_RATE, PAIR_SIGNAL_COOLDOWN, SPREAD_COST_PCT,
    LOT_MIN, LOT_MAX, CONTRACT_SIZES,
    ADX_TREND_THRESHOLD, BB_SQUEEZE_THRESHOLD, VWAP_DEV_THRESHOLD
)

logger = logging.getLogger(__name__)

class AITradingEngine:
    """
    The core brain of Nexubot.
    """
    def __init__(self):
        self.learned_patterns = {}
        self.fake_detector = FakeBreakoutDetector()
        self.user_balance_zar = DEFAULT_BALANCE_ZAR
        self.signal_history: Dict[str, float] = {} # { 'BTCUSDT': timestamp }
        self.db_manager = None
        self.load_models()

    def load_models(self) -> None:
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    self.learned_patterns = pickle.load(f)
                logger.info("📚 AI models loaded")
            except Exception:
                self.learned_patterns = {}

    def save_models(self) -> None:
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(self.learned_patterns, f)
            logger.info("💾 AI models saved successfully")
        except Exception as e:
            logger.error(f"Model save error: {e}")

    def set_user_balance(self, balance_zar: float) -> None:
        self.user_balance_zar = balance_zar

    def set_db_manager(self, db: DatabaseManager):
        self.db_manager = db

    async def analyze_market(self, symbol: str, klines: List) -> Optional[Dict]:
        """
        Main Analysis Pipeline with Regime Filtering
        """
        # Cooldown Check
        if self.is_on_cooldown(symbol): return None
        if self.db_manager:
            # Prevent overusage if recently lost
            if await self.db_manager.check_recent_loss(symbol):
                return None

        # Data Prep
        df = self._prepare_data(klines)
        if df is None: return None

        # Fakeout Detection
        fake_analysis = self.fake_detector.analyze(df)
        if fake_analysis['is_fake']: return None

        # Market Regime & Strategy Routing
        curr = df.iloc[-1]

        # Determine Market State
        is_trending = curr['adx'] > ADX_TREND_THRESHOLD
        is_squeezing = curr['bb_width'] < BB_SQUEEZE_THRESHOLD

        # Calculate Deviation from VWAP (Z-Score proxy)
        vwap_dist = abs(curr['close'] - curr['vwap'])
        std_dev = (curr['vwap_upper'] - curr['vwap']) / 2
        is_extended = vwap_dist > (std_dev * VWAP_DEV_THRESHOLD)

        potential_signal = None

        # Routing Logic
        if is_extended:
            # Priority 1: Mean Reversion (Fair Value Return)
            potential_signal = self._strat_fair_value_return(curr, df)
        elif is_squeezing:
            # Priority 2: Volatility Breakout
            potential_signal = self._strat_vol_squeeze(curr, df)
        elif is_trending:
            # Priority 3: Smart Trend Follow
            potential_signal = self._strat_smart_trend(curr, df)

        if not potential_signal: return None

        # AI Confidence Adjustment
        final_signal = await self._adjust_confidence(symbol, potential_signal)

        # Risk Calculation
        if final_signal['confidence'] >= MIN_CONFIDENCE:
            result = self._calculate_risk_metrics(symbol, final_signal, curr)

            if result:
                self.signal_history[symbol] = time.time()

            return result

        return None

    # ---------------------------------------------------
    # STRATEGIES
    # ---------------------------------------------------
    def _strat_smart_trend(self, curr, df) -> Optional[Dict]:
        """
        Strategy 1: Smart Trend (Pullback + Market Structure)
        Condition: Price > VWAP (Uptrend).
        Setup: Liquidity Sweep (Low dipped below previous swing low).
        Trigger: Break back above recent high.
        """
        prev = df.iloc[-2]
        trend_direction = "LONG" if curr['close'] > curr['vwap'] else "SHORT"

        if trend_direction == "LONG":
            # Trend Filter
            if curr['close'] < curr['ema_200']: return None

            # Setup: Price dipped recently (RSI Pullback)
            if curr['rsi'] > 50: return None # Must be pulling back

            # Trigger: Price bouncing off EMA50 or VWAP
            if curr['low'] <= curr['ema_50'] and curr['close'] > curr['ema_50']:
                return {
                    'strategy': 'Smart Trend',
                    'signal': 'BUY',
                    'direction': 'LONG',
                    'confidence': 80.0
                }

        elif trend_direction == "SHORT":
            if curr['close'] > curr['ema_200']: return None
            if curr['rsi'] < 50: return None
            if curr['high'] >= curr['ema_50'] and curr['close'] < curr['ema_50']:
                return {
                    'strategy': 'Smart Trend',
                    'signal': 'SELL',
                    'direction': 'SHORT',
                    'confidence': 80.0
                }
        return None

    def _strat_vol_squeeze(self, curr, df) -> Optional[Dict]:
        """
        Strategy 2: Vol-Squeeze (Bollinger + Volume Profile)
        Condition: BB Squeeze (< Threshold).
        Confirm: Breakout of Volume Profile High/Low.
        Filter: Positive Volume Delta.
        """
        # Note: We are already inside 'is_squeezing' block from main router

        # Long Breakout
        if curr['close'] > curr['bb_upper'] and curr['cum_delta'] > 0:
            # Check if breaking recent Volume High
            if curr['close'] > curr['vol_profile_vah']:
                return {
                    'strategy': 'Vol-Squeeze',
                    'signal': 'BUY',
                    'direction': 'LONG',
                    'confidence': 85.0
                }

        # Short Breakout
        elif curr['close'] < curr['bb_lower'] and curr['cum_delta'] < 0:
            return {
                'strategy': 'Vol-Squeeze',
                'signal': 'SELL',
                'direction': 'SHORT',
                'confidence': 85.0
            }

        return None

    def _strat_fair_value_return(self, curr, df) -> Optional[Dict]:
        """
        Strategy 3: Fair Value Return (VWAP Reversion)
        Condition: Price extended > 2 StDev from VWAP.
        Trigger: Reversal Candle (Close back inside bands).
        """
        prev = df.iloc[-2]

        # Short Reversion (Price was above Top Band, now closing below)
        if prev['close'] > prev['vwap_upper'] and curr['close'] < curr['vwap_upper']:
             return {
                'strategy': 'Fair Value Return',
                'signal': 'SELL',
                'direction': 'SHORT',
                'confidence': 78.0 # Counter-trend is riskier
            }

        # Long Reversion (Price was below Bottom Band, now closing above)
        if prev['close'] < prev['vwap_lower'] and curr['close'] > curr['vwap_lower']:
             return {
                'strategy': 'Fair Value Return',
                'signal': 'BUY',
                'direction': 'LONG',
                'confidence': 78.0
            }

        return None

    # ---------------------------------------------------
    # AI & RISK LOGIC
    # ---------------------------------------------------
    async def _adjust_confidence(self, symbol: str, signal: Dict) -> Dict:
        """
        Adjust confidence based on DB history and Volatility.
        """
        # 1. DB History Check
        if self.db_manager:
            hist_win_rate = await self.db_manager.get_strategy_performance(signal['strategy'])
            if hist_win_rate > 0.6: signal['confidence'] += 5.0
            elif hist_win_rate < 0.4: signal['confidence'] -= 10.0

        # 2. Local Learning (Pickle)
        key = f"{symbol}|{signal['strategy']}"
        if key in self.learned_patterns:
            data = self.learned_patterns[key]
            if data['trades'] > 5 and (data['wins']/data['trades']) < 0.4:
                signal['confidence'] -= 20.0 # Heavy penalty for recurring losers

            # Cap at 99%
            signal['confidence'] = min(99.0, signal['confidence'])

        return signal

    def _calculate_risk_metrics(self, symbol: str, signal: Dict, curr) -> Optional[Dict]:
        """
        Improved Lot Sizing to prevent Max Lot usage.
        """
        close_price = float(curr['close'])
        atr = float(curr['atr'])
        contract_size = CONTRACT_SIZES.get(symbol, CONTRACT_SIZES['DEFAULT'])

        # Spread
        bid_price = close_price
        ask_price = close_price * (1 + (SPREAD_COST_PCT/100))

        # SL / TP Settings based on Strategy Type
        if signal['strategy'] == 'Fair Value Return':
            # Tighter stops for mean reversion
            sl_mult, tp_mult = 1.5, 2.0
        elif signal['strategy'] == 'Vol-Squeeze':
            # Wider stops for volatility
            sl_mult, tp_mult = 2.5, 4.5
        else:
            sl_mult, tp_mult = 1.5, 2.5

        if signal['signal'] == 'BUY':
            entry_price = ask_price # Enter Negative
            sl_price = bid_price - (atr * sl_mult)
            tp_price = bid_price + (atr * tp_mult)
        else: # SELL
            entry_price = bid_price
            sl_price = ask_price + (atr * sl_mult)
            tp_price = ask_price - (atr * tp_mult)

        #  # --- LOT SIZE CALCULATION ---
        # Calculate Monetary Risk
        balance_usd = self.user_balance_zar / USD_ZAR_RATE
        risk_per_trade_usd = balance_usd * (RISK_PER_TRADE / 100)

        # Distance Calculation (Absolute Value)
        sl_dist = abs(entry_price - sl_price)
        if sl_dist == 0: return None

        # Formula: Lots = (Risk Amount) / (Distance * Contract Size)
        raw_lots = risk_per_trade_usd / (sl_dist * contract_size)

        # Clamp Lots
        lot_size = max(LOT_MIN, min(raw_lots, LOT_MAX))
        lot_size = round(lot_size, 2)

        # Total Risk (If SL hit)
        risk_usd = sl_dist * lot_size * contract_size
        risk_zar = risk_usd * USD_ZAR_RATE

        # Safety Check: If Lot Max forced risk to be > 3% of balance, skip trade
        if risk_zar > (self.user_balance_zar * 0.03):
            return None

        # Potential Profit
        tp_dist = abs(entry_price - tp_price)
        profit_zar = (tp_dist * lot_size * contract_size) * USD_ZAR_RATE
        spread_cost_zar = (abs(ask_price - bid_price) * lot_size * contract_size) * USD_ZAR_RATE
        margin_zar = ((entry_price * lot_size * contract_size) / LEVERAGE) * USD_ZAR_RATE

        signal.update({
            'price': round(entry_price, 6),
            'sl': round(sl_price, 6),
            'tp': round(tp_price, 6),
            'lot_size': lot_size,
            'spread_cost_zar': round(spread_cost_zar, 2),
            'risk_zar': round(risk_zar, 2),
            'profit_zar': round(profit_zar, 2),
            'required_margin_zar': round(margin_zar, 2),
            'contract_size': contract_size,
            'is_high_risk': symbol in HIGH_RISK_SYMBOLS,
        })
        return signal

    # ---------------------------------------------------
    # UTILS
    # ---------------------------------------------------
    def is_on_cooldown(self, symbol: str) -> bool:
        last_time = self.signal_history.get(symbol, 0)
        return (time.time() - last_time) < PAIR_SIGNAL_COOLDOWN

    def _prepare_data(self, klines: List) -> Optional[pd.DataFrame]:
        try:
            df = pd.DataFrame(klines)
            df = df[[0, 1, 2, 3, 4, 5]].copy()
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Apply new indicators
            df = TechnicalAnalyzer.calculate_indicators(df)
            return df
        except Exception:
            return None

    def learn(self, symbol: str, strategy_name: str, won: bool, pnl_amount: float) -> None:
        """
        Records trade outcome.
        """
        key = f"{symbol}|{strategy_name}"
        if key not in self.learned_patterns:
            self.learned_patterns[key] = {
                'wins': 0, 'trades': 0, 'pnl_accumulated': 0.0
            }

        self.learned_patterns[key]['trades'] += 1
        self.learned_patterns[key]['pnl_accumulated'] += pnl_amount
        if won: self.learned_patterns[key]['wins'] += 1
        self.save_models()