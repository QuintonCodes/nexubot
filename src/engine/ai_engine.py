import os
import pickle
import logging
import pandas as pd
from typing import Dict, List

from src.database.manager import DatabaseManager
from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.patterns import FakeBreakoutDetector
from src.config import (
    MODEL_FILE, MIN_CONFIDENCE, ACCOUNT_BALANCE,
    RISK_PER_TRADE, LEVERAGE
)

logger = logging.getLogger(__name__)

class AITradingEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.learned_patterns = {}
        self.fake_detector = FakeBreakoutDetector()
        self.load_models()

    def load_models(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    self.learned_patterns = pickle.load(f)
                logger.info("📚 AI models loaded")
            except Exception:
                self.learned_patterns = {}

    def save_models(self):
        """Save AI models to file"""
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(self.learned_patterns, f)
            logger.info("💾 AI models saved successfully")
        except Exception as e:
            logger.error(f"Model save error: {e}")

    def analyze_market(self, symbol: str, klines: List) -> Dict:
        """
        Parses Binance Klines: [Open Time, Open, High, Low, Close, Volume, ...]
        """
        if not klines or len(klines) < 100: return None

        try:
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'q_vol', 'trades', 'buy_vol', 'buy_q_vol', 'ignore'
            ])

            cols = ['open', 'high', 'low', 'close', 'volume']
            for col in cols:
                df[col] = df[cols].astype(float)

            df = TechnicalAnalyzer.calculate_indicators(df)

            # Check Fake Breakout
            fake_analysis = self.fake_detector.analyze(df)
            if fake_analysis['is_fake']:
                return None # Skip dangerous setups

            # Run Strategies
            strategies = [
                self._strat_trend_pullback,
                self._strat_volatility_breakout,
                self._strat_reversal
            ]

            best_signal = None
            highest_conf = 0

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            for strategy in strategies:
                sig = strategy(curr, prev, df)
                if sig and sig['confidence'] > highest_conf:
                    highest_conf = sig['confidence']
                    best_signal = sig

            if not best_signal: return None

            # AI Adjustment
            best_signal = self._apply_ai_learning(symbol, best_signal)

            if best_signal['confidence'] >= MIN_CONFIDENCE:
                # Calculate Risk Management (SL/TP/Lots)
                return self._calculate_risk_metrics(symbol, best_signal, curr)

        except Exception as e:
            logger.error(f"Analysis Data Error {symbol}: {e}")
            return None

        return None

    # ---------------------------------------------------
    # STRATEGIES
    # ---------------------------------------------------

    def _strat_trend_pullback(self, curr, _prev, _df) -> Dict:
        """Strategy 1: Trend Pullback"""
        if curr['ema_50'] > curr['ema_200']:
            if curr['rsi'] < 45 and curr['close'] > curr['ema_50']:
                return {
                    'strategy': 'Trend Pullback',
                    'signal': 'BUY',
                    'confidence': 85
                }
        elif curr['ema_50'] < curr['ema_200']:
            if curr['rsi'] > 55 and curr['close'] < curr['ema_50']:
                return {
                    'strategy': 'Trend Pullback',
                    'signal': 'SELL',
                    'confidence': 85
                }
        return None

    def _strat_volatility_breakout(self, curr, prev, _df) -> Dict:
        """Strategy 2: Volatility Breakout"""
        if prev['bb_width'] < 0.10: # Tight bands
            if curr['volume'] > curr['vol_sma'] * 2.0:
                if curr['close'] > curr['bb_upper']:
                    return {
                        'strategy': 'Vol Breakout',
                        'signal': 'BUY',
                        'confidence': 90
                    }
                elif curr['close'] < curr['bb_lower']:
                    return {
                        'strategy': 'Vol Breakout',
                        'signal': 'SELL',
                        'confidence': 90,
                    }
        return None

    def _strat_reversal(self, curr, _prev, _df) -> Dict:
        """Strategy 3: Reversal Scalp"""
        if curr['rsi'] > 75 and curr['close'] > curr['bb_upper']:
            return {
                'strategy': 'Reversal Scalp',
                'signal': 'SELL',
                'confidence': 80,
            }
        elif curr['rsi'] < 25 and curr['close'] < curr['bb_lower']:
            return {
                'strategy': 'Reversal Scalp',
                'signal': 'BUY',
                'confidence': 80,
            }
        return None

    # ---------------------------------------------------
    # AI & RISK UTILS
    # ---------------------------------------------------

    def _apply_ai_learning(self, symbol, signal):
        """Adjust confidence based on past win rate"""
        # Key: "BTCUSDT|Trend Pullback"
        key = f"{symbol}|{signal['strategy']}"

        if key in self.learned_patterns:
            data = self.learned_patterns[key]
            if data['trades'] > 3:
                win_rate = data['wins'] / data['trades']
                if win_rate > 0.65: signal['confidence'] += 5
                elif win_rate < 0.45: signal['confidence'] -= 15

        return signal

    def _calculate_risk_metrics(self, symbol, signal, curr):
        """Calculate SL, TP, and Lot Size for MT5 using Leverage"""
        price = float(curr['close'])
        atr = float(curr['atr'])

        # Risk Settings
        sl_multiplier = 1.5
        tp_multiplier = 2.0

        if signal['signal'] == 'BUY':
            sl_price = price - (atr * sl_multiplier)
            tp_price = price + (atr * tp_multiplier)
            stop_loss_dist = price - sl_price
        else: # SELL
            sl_price = price + (atr * sl_multiplier)
            tp_price = price - (atr * tp_multiplier)
            stop_loss_dist = sl_price - price

        # Lot Size Calculation
        # Dtermine how much money we risk
        # Risk Amount = Balance * (Risk% / 100)
        risk_amount = ACCOUNT_BALANCE * (RISK_PER_TRADE / 100)

        try:
            # Example: Risk $20. SL distance is $100 on BTC.
            # Position needed: 0.2 BTC.
            # Does 0.2 BTC exceed Balance * Leverage?

            sl_percent = stop_loss_dist / price
            position_value = risk_amount / sl_percent

            max_buying_power = ACCOUNT_BALANCE * LEVERAGE

            if position_value > max_buying_power:
                position_value = max_buying_power

            # Convert USD position to Units (Lots)
            # For Crypto, 1 Lot usually = 1 Unit
            lot_size = round(position_value / price, 4)

        except ZeroDivisionError:
            lot_size = 0.01

        signal.update({
            'price': price,
            'sl': round(sl_price, 5),
            'tp': round(tp_price, 5),
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'est_profit': round(risk_amount * (tp_multiplier/sl_multiplier), 2)
        })
        return signal

    def learn(self, symbol, strategy_name, won):
        key = f"{symbol}|{strategy_name}"

        if key not in self.learned_patterns:
            self.learned_patterns[key] = {'wins': 0, 'trades': 0}

        self.learned_patterns[key]['trades'] += 1
        if won: self.learned_patterns[key]['wins'] += 1
        self.save_models()