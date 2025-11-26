import os
import pickle
import logging
import time
import pandas as pd
from typing import Dict, List, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.patterns import FakeBreakoutDetector
from src.config import (
    MODEL_FILE, MIN_CONFIDENCE, DEFAULT_BALANCE_ZAR,
    RISK_PER_TRADE, LEVERAGE, HIGH_RISK_SYMBOLS,
    USD_ZAR_RATE, PAIR_SIGNAL_COOLDOWN,
    SPREAD_COST_PCT, MIN_LOT_SIZE
)

logger = logging.getLogger(__name__)

class AITradingEngine:
    """
    The core brain of Nexubot.
    """
    def __init__(self):
        self.learned_patterns = {}
        self.fake_detector = FakeBreakoutDetector()
        self.load_models()
        self.user_balance_zar = DEFAULT_BALANCE_ZAR
        self.signal_history: Dict[str, float] = {} # { 'BTCUSDT': timestamp }

    def load_models(self) -> None:
        """Loads historical performance data for AI scoring."""
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    self.learned_patterns = pickle.load(f)
                logger.info("📚 AI models loaded")
            except Exception:
                self.learned_patterns = {}

    def save_models(self) -> None:
        """Saves performance data to disk."""
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(self.learned_patterns, f)
            logger.info("💾 AI models saved successfully")
        except Exception as e:
            logger.error(f"Model save error: {e}")

    def set_user_balance(self, balance_zar: float) -> None:
        """Sets the user balance in ZAR for risk calculations."""
        self.user_balance_zar = balance_zar

    def is_on_cooldown(self, symbol: str) -> bool:
        """Checks if the specific pair is on a signal cooldown."""
        last_time = self.signal_history.get(symbol, 0)
        return (time.time() - last_time) < PAIR_SIGNAL_COOLDOWN

    def analyze_market(self, symbol: str, klines: List) -> Optional[Dict]:
        """
        Orchestrates the analysis pipeline.
        """
        # Cooldown Check
        if self.is_on_cooldown(symbol): return None

        # Data Validation
        if not klines or not isinstance(klines, list) or len(klines) < 50:
            return None

        try:
            # Robust Parsing
            df = pd.DataFrame(klines)
            if df.shape[1] < 6: return None

            # Safe slicing and typing
            df = df[[0, 1, 2, 3, 4, 5]].copy()
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            cols = ['open', 'high', 'low', 'close', 'volume']
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Technical Analysis
            df = TechnicalAnalyzer.calculate_indicators(df)

            # Fakeout Detection
            fake_analysis = self.fake_detector.analyze(df)
            if fake_analysis['is_fake']: return None

            # Strategy Hunting
            strategies = [
                self._strat_trend_pullback,
                self._strat_volatility_breakout,
                self._strat_reversal,
                self._strat_macd_momentum
            ]

            best_signal = None
            highest_conf = 0.0
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            for strategy in strategies:
                sig = strategy(curr, prev, df)
                if sig and sig['confidence'] > highest_conf:
                    highest_conf = sig['confidence']
                    best_signal = sig

            if not best_signal: return None

            # AI Scoring Adjustment (Dynamic Confidence)
            best_signal = self._apply_ai_learning(symbol, best_signal)

            # Final Filter & Risk Calc
            if best_signal['confidence'] >= MIN_CONFIDENCE:
                # Calculate Risk with HFM/MT5 Logic
                final_signal = self._calculate_risk_metrics(symbol, best_signal, curr)

                # Only return if the Risk/Reward is still valid after spread calculation
                if final_signal:
                    self.signal_history[symbol] = time.time() # Start cooldown
                    return final_signal

        except Exception as e:
            logger.error(f"Analysis Error {symbol}: {e}")
            return None

        return None

    # ---------------------------------------------------
    # STRATEGIES
    # ---------------------------------------------------

    def _strat_trend_pullback(self, curr, _prev, _df) -> Optional[Dict]:
        """Strategy: Trend Following Pullback"""
        # Long: EMA50 > EMA200 (Uptrend) + RSI < 45 (Pullback)
        if curr['ema_50'] > curr['ema_200']:
            if curr['rsi'] < 45 and curr['close'] > curr['ema_50']:
                return {
                    'strategy': 'Trend Pullback',
                    'signal': 'BUY',
                    'direction': 'LONG',
                    'confidence': 82.0
                }
        # Short: EMA50 < EMA200 (Downtrend) + RSI > 55 (Pullback)
        elif curr['ema_50'] < curr['ema_200']:
            if curr['rsi'] > 55 and curr['close'] < curr['ema_50']:
                return {
                    'strategy': 'Trend Pullback',
                    'signal': 'SELL',
                    'direction': 'SHORT',
                    'confidence': 82.0
                }
        return None

    def _strat_volatility_breakout(self, curr, prev, _df) -> Optional[Dict]:
        """Strategy: Bollinger Band Squeeze Breakout"""
        # Trigger: Bands tight (< 0.08) + Volume Spike + Band Pierce
        if prev['bb_width'] < 0.08:
            if curr['volume'] > curr['vol_sma'] * 1.5:
                if curr['close'] > curr['bb_upper']:
                    return {
                        'strategy': 'Vol Breakout',
                        'signal': 'BUY',
                        'direction': 'LONG',
                        'confidence': 88.0
                    }
                elif curr['close'] < curr['bb_lower']:
                    return {
                        'strategy': 'Vol Breakout',
                        'signal': 'SELL',
                        'direction': 'SHORT',
                        'confidence': 88.0,
                    }
        return None

    def _strat_reversal(self, curr, _prev, _df) -> Optional[Dict]:
        """Strategy: Mean Reversion / Counter-Trend"""
        # Short: RSI Extreme High + Price above Upper Band
        if curr['rsi'] > 75 and curr['close'] > curr['bb_upper']:
            return {
                'strategy': 'Reversal Scalp',
                'signal': 'SELL',
                'direction': 'SHORT',
                'confidence': 75.0,
            }
        # Long: RSI Extreme Low + Price below Lower Band
        elif curr['rsi'] < 25 and curr['close'] < curr['bb_lower']:
            return {
                'strategy': 'Reversal Scalp',
                'signal': 'BUY',
                'direction': 'LONG',
                'confidence': 75.0,
            }
        return None

    def _strat_macd_momentum(self, curr, prev, _df) -> Optional[Dict]:
        """Strategy: MACD Crossover Momentum"""
        if prev['macd'] < prev['signal'] and curr['macd'] > curr['signal']:
            if curr['hist'] > 0:
                return {
                    'strategy': 'MACD Momentum',
                    'signal': 'BUY',
                    'direction': 'LONG',
                    'confidence': 85.0
                }
        elif prev['macd'] > prev['signal'] and curr['macd'] < curr['signal']:
            if curr['hist'] < 0:
                return {
                    'strategy': 'MACD Momentum',
                    'signal': 'SELL',
                    'direction': 'SHORT',
                    'confidence': 85.0
                }
        return None

    # ---------------------------------------------------
    # RISK MANAGEMENT (MT5 LOGIC: ASK vs BID)
    # ---------------------------------------------------
    def _calculate_risk_metrics(self, symbol: str, signal: Dict, curr) -> Optional[Dict]:
        """
        Calculates Entry, Stop Loss, and Lot Size considering Spread.
        """
        # We assume the 'close' price from data is the BID price (Market Price)
        bid_price = float(curr['close'])

        # Calculate Ask Price based on Spread Config
        # If Spread is 0.1%, Ask is 0.1% higher than Bid.
        spread_multiplier = SPREAD_COST_PCT / 100
        ask_price = bid_price * (1 + spread_multiplier)

        atr = float(curr['atr'])

        # Volatility-Based Stops
        is_high_risk = symbol in HIGH_RISK_SYMBOLS
        sl_mult = 2.5 if is_high_risk else 1.5
        tp_mult = 3.0 if is_high_risk else 2.0

        if signal['signal'] == 'BUY':
            # ENTRY: We Buy at ASK
            entry_price = ask_price

            # SL/TP: Triggered by BID price (Chart Price)
            # We want SL to be below the current BID
            sl_price = bid_price - (atr * sl_mult)
            tp_price = bid_price + (atr * tp_mult)
        else: # SELL
            # ENTRY: We Sell at BID
            entry_price = bid_price

            # SL/TP: Triggered by ASK price (Chart Price + Spread)
            # To close a Short, we must BUY back at ASK.
            # So SL must be ABOVE the current Ask Price
            sl_price = ask_price + (atr * sl_mult)
            tp_price = ask_price - (atr * tp_mult)

        # Distance Calculation (Absolute Value)
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(entry_price - tp_price)

        if sl_dist == 0: return None

        # Financial Calculation (ZAR -> USD)
        balance_usd = self.user_balance_zar / USD_ZAR_RATE

        # Determine Risk Amount (USD)
        risk_amount_usd = balance_usd * (RISK_PER_TRADE / 100)

        # Lot Size Calculation
        calculated_lots = risk_amount_usd / sl_dist

        # Normaliza Lots (HFM Logic)
        if bid_price > 500: # High value assets (BTC) need decimals
            lot_size = round(calculated_lots, 3)
        else:
            lot_size = round(calculated_lots, 1)

        if lot_size < MIN_LOT_SIZE:
            lot_size = MIN_LOT_SIZE

        # --- MARGIN CHECK ---
        # Margin = (Entry * Lots) / Leverage
        required_margin = (entry_price * lot_size) / LEVERAGE

        # If margin exceeds balance, scale down
        if required_margin > balance_usd:
            # Max possible lots = Balance * Leverage / Entry
            max_lots = (balance_usd * LEVERAGE) / entry_price
            # Take 95% of max to be safe
            lot_size = round(max_lots * 0.95, 3)
            if lot_size < MIN_LOT_SIZE: return None # Can't afford min trade

        # --- FINAL PROFIT/LOSS ESTIMATION ---
        # Recalculate Risk based on finalized Lot Size
        actual_risk_usd = sl_dist * lot_size

        # Estimated Profit is based on distance to TP
        potential_profit_usd = tp_dist * lot_size

        # Convert to ZAR
        risk_zar = actual_risk_usd * USD_ZAR_RATE
        profit_zar = potential_profit_usd * USD_ZAR_RATE

        # Trade Viability Check
        # If Profit is less than Risk (due to wide spread eating the gain), skip.
        if profit_zar < risk_zar:
            return None # Risk Reward bad due to spread

        signal.update({
            'price': round(entry_price, 2),
            'sl': round(sl_price, 2),
            'tp': round(tp_price, 2),
            'lot_size': lot_size,
            'risk_zar': risk_zar,
            'profit_zar': profit_zar,
            'is_high_risk': is_high_risk
        })
        return signal

    def _apply_ai_learning(self, symbol: str, signal: Dict) -> Dict:
        """
        Adjusts confidence based on Profit/Loss history.
        """
        key = f"{symbol}|{signal['strategy']}"
        if key in self.learned_patterns:
            data = self.learned_patterns[key]

            # Require minimum sample size
            if data['trades'] > 3:
                # Metric 1: Win Rate
                win_rate = data['wins'] / data['trades']

                # Metric 2: Profitability (Avg PnL per trade)
                net_pnl = data.get('pnl_accumulated', 0)

                if win_rate > 0.6 and net_pnl > 0:
                    signal['confidence'] += 10.0 # Strong Boost
                elif win_rate < 0.4:
                    signal['confidence'] -= 15.0 # Penalty

                # Cap at 99%
                signal['confidence'] = min(99.0, signal['confidence'])

        return signal

    def learn(self, symbol: str, strategy_name: str, won: bool, pnl_amount: float) -> None:
        """
        Records trade outcome.

        Args:
            symbol: Asset name
            strategy_name: Strategy used
            won: True if trade was profitable
            pnl_amount: Actual profit/loss amount (can be negative)
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