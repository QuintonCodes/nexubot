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
    MODEL_FILE, MIN_CONFIDENCE, RISK_PER_TRADE_PCT,
    SMALL_ACCOUNT_THRESHOLD_ZAR, MAX_SURVIVAL_CAP_PCT,
    BROKER_SPECS, SPREAD_POINTS, HIGH_RISK_SYMBOLS,
    PAIR_SIGNAL_COOLDOWN
)

logger = logging.getLogger(__name__)

class AITradingEngine:
    """
    The core brain of Nexubot.
    """
    def __init__(self):
        self.learned_patterns = {}
        self.fake_detector = FakeBreakoutDetector()
        self.user_balance_zar = 0.0
        self.usd_zar_rate = 18.00
        self.db_manager = None
        self.signal_history = {}
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

    def set_context(self, balance: float, rate: float, db: DatabaseManager):
        self.user_balance_zar = balance
        self.usd_zar_rate = rate if rate > 0 else 18.00
        self.db_manager = db

    async def analyze_market(self, symbol: str, klines: List) -> Optional[Dict]:
        """
        Main Analysis Pipeline
        """
        if self.is_on_cooldown(symbol): return None
        if self.db_manager and await self.db_manager.check_recent_loss(symbol):
            return None

        df = self._prepare_data(klines)
        if df is None: return None

        # Fakeout Detection
        fake_analysis = self.fake_detector.analyze(df)
        if fake_analysis['is_fake']: return None

        # Strategy Routing
        curr = df.iloc[-1]
        is_forex ='/' in symbol
        potential_signal = self._evaluate_strategies(curr, df, is_forex)

        if not potential_signal: return None

        # Confidence & Risk
        final_signal = await self._adjust_confidence(symbol, potential_signal)

        if final_signal['confidence'] >= MIN_CONFIDENCE:
            result = self._calculate_risk_metrics(symbol, final_signal, curr, is_forex)
            if result:
                self.signal_history[symbol] = time.time()
                return result

        return None

    def _evaluate_strategies(self, curr, df, is_forex: bool) -> Optional[Dict]:
        """
        Central Router: Directs data to the correct market strategy set.
        """
        if is_forex:
            return self._evaluate_forex_strategies(curr, df)
        else:
            return self._evaluate_crypto_strategies(curr, df)

    def _evaluate_crypto_strategies(self, curr, df) -> Optional[Dict]:
        """Specific strategies for Crypto (Volatility/Volume focus)"""
        prev = df.iloc[-2]
        trend_direction = "LONG" if curr['close'] > curr['vwap'] else "SHORT"

        # --- STRATEGY 1: Smart Trend ---
        vwap_band_width = (curr['vwap_upper'] - curr['vwap'])

        if abs(curr['close'] - curr['vwap']) > (vwap_band_width * 0.5):
            if trend_direction == "LONG" and curr['close'] > curr['ema_200']:
                if curr['low'] <= curr['ema_50'] and curr['close'] > curr['ema_50']:
                    return {
                        'strategy': 'Smart Trend',
                        'signal': 'BUY',
                        'direction': 'LONG',
                        'confidence': 80.0
                    }

            elif trend_direction == "SHORT" and curr['close'] < curr['ema_200']:
                if curr['high'] >= curr['ema_50'] and curr['close'] < curr['ema_50']:
                    return {
                        'strategy': 'Smart Trend',
                        'signal': 'SELL',
                        'direction': 'SHORT',
                        'confidence': 80.0
                    }

        # --- Strategy 2: Vol-Squeeze (Bollinger + Volume Profile) ---
        if curr['bb_width'] < 0.10:
            if curr['close'] > curr['bb_upper'] and curr['cum_delta'] > 0:
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

        # --- Strategy 3: Fair Value Return (VWAP Reversion) --
        if curr['close'] > curr['vwap'] and curr['rsi'] > 55 and curr['adx'] > 20:
            return {
                'strategy': 'Crypto Momentum',
                'signal': 'BUY',
                'direction': 'LONG',
                'confidence': 75.0
            }

        # Long Reversion (Price was below Bottom Band, now closing above)
        if curr['close'] < curr['vwap'] and curr['rsi'] < 45 and curr['adx'] > 20:
            return {
                'strategy': 'Crypto Momentum',
                'signal': 'SELL',
                'direction': 'SHORT',
                'confidence': 75.0
            }

        return None

    def _evaluate_forex_strategies(self, curr, df) -> Optional[Dict]:
        """Specific strategies for Forex (Trend/Reversion focus)"""
        prev = df.iloc[-2]

        # --- STRATEGY 1: Fair Value Reversion (Mean Reversion) ---
        # Good for ranging forex pairs or overextended moves
        vwap_dist = abs(curr['close'] - curr['vwap'])
        std_dev = (curr['vwap_upper'] - curr['vwap']) / 2

        if vwap_dist > (std_dev * 2.0):
            # Price is overextended
            # Check for reversion (closing back inside bands)
            if prev['close'] > prev['vwap_upper'] and curr['close'] < curr['vwap_upper']:
                return {
                    'strategy': 'Fair Value Reversion',
                    'signal': 'SELL',
                    'direction': 'SHORT',
                    'confidence': 82.0
                }
            if prev['close'] < prev['vwap_lower'] and curr['close'] > curr['vwap_lower']:
                return {
                    'strategy': 'Fair Value Reversion',
                    'signal': 'BUY',
                    'direction': 'LONG',
                    'confidence': 82.0
                }

        # --- STRATEGY 2: FX Trend Flow (EMA Stack) ---
        # Buy: Price > EMA50 > EMA200
        if curr['close'] > curr['ema_50'] > curr['ema_200']:
            if 50 < curr['rsi'] < 70:
                if curr['low'] <= curr['ema_9']:
                    return {
                        'strategy': 'FX Trend Flow',
                        'signal': 'BUY',
                        'direction': 'LONG',
                        'confidence': 80.0
                    }

        # Sell: Price < EMA50 < EMA200:
        if curr['close'] < curr['ema_50'] < curr['ema_200']:
             if 30 < curr['rsi'] < 50:
                 if curr['high'] >= curr['ema_9']:
                    return {
                        'strategy': 'FX Trend Flow',
                        'signal': 'SELL',
                        'direction': 'SHORT',
                        'confidence': 80.0
                    }

        return None

    async def _adjust_confidence(self, symbol: str, signal: Dict) -> Dict:
        """
        Adjust confidence based on DB history and Volatility.
        """
        # DB History Check
        if self.db_manager:
            hist_win_rate = await self.db_manager.get_strategy_performance(signal['strategy'])
            if hist_win_rate > 0.6: signal['confidence'] += 5.0
            elif hist_win_rate < 0.4: signal['confidence'] -= 10.0

        # Local Learning
        key = f"{symbol}|{signal['strategy']}"
        if key in self.learned_patterns:
            data = self.learned_patterns[key]
            if data['trades'] > 5 and (data['wins']/data['trades']) < 0.35:
                signal['confidence'] -= 20.0
            # Cap at 99%
            signal['confidence'] = min(99.0, signal['confidence'])

        return signal

    def _calculate_risk_metrics(self, symbol: str, signal: Dict, curr, is_forex: bool) -> Optional[Dict]:
        """
        Calculates Lot Size, SL/TP Prices, and Risk/Reward in ZAR.
        """
        # 1. Asset Specs & Spreads
        default_spec = {'contract_size': 1, 'digits': 2, 'min_vol': 0.01, 'max_vol': 100.0, 'stop_level': 0, 'commission_per_lot': 0}
        spec = BROKER_SPECS.get(symbol, default_spec)

        contract_size = spec['contract_size']
        decimals = spec['digits']
        min_vol = spec['min_vol']
        max_vol = spec['max_vol']
        stop_level_points = spec['stop_level']
        comm_per_lot = spec['commission_per_lot']

        point = 1 / (10 ** decimals)

        # 2. Get Spread (In Points)
        spread_points = SPREAD_POINTS.get(symbol, 20)
        spread_val_usd = spread_points * point # Price difference

        close_price = float(curr['close'])

        # 3. Entry Price (Simulating Bid/Ask with Spread)
        if signal['signal'] == 'BUY':
            entry_price = close_price + spread_val_usd # Ask
        else:
            entry_price = close_price # Bid

        # 4. Stop Loss Calculation (ATR Based)
        atr = float(curr['atr'])
        if atr == 0: return None

        sl_mult = 1.5 if is_forex else 2.0
        raw_sl_dist = atr * sl_mult

        # Enforce Minimum Stop Level
        min_sl_dist = stop_level_points * point
        sl_dist = max(raw_sl_dist, min_sl_dist * 1.1) # 10% buffer above stop level

        if signal['signal'] == 'BUY':
            sl_price = entry_price - sl_dist
            tp_price = entry_price + (sl_dist * 1.5)
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - (sl_dist * 1.5)

        # 5. Lot Sizing (Risk Based)
        risk_per_trade_zar = self.user_balance_zar * (RISK_PER_TRADE_PCT / 100)
        risk_per_trade_usd = risk_per_trade_zar / self.usd_zar_rate

        dist_to_sl = abs(entry_price - sl_price)
        if dist_to_sl == 0: return None

        # Formula: Risk = (Dist * Contract * Lots)
        raw_lots = risk_per_trade_usd / (contract_size * dist_to_sl)

        # 6. Apply Volume Limits
        lots = max(min_vol, min(raw_lots, max_vol))

        # Rounding (Forex 2 decimals, Crypto might differ but HFM standardizes volume step usually)
        # HFM Vol Step is usually 0.01 or 1.0 depending on Min Vol.
        if min_vol >= 1.0:
            lots = round(lots, 0) # Integer lots (e.g., 1, 2, 3)
        else:
            lots = round(lots, 2) # Fractional lots (e.g., 0.01)

        # 7. Recalculate Actual Risk
        actual_risk_usd = lots * contract_size * dist_to_sl
        actual_risk_zar = actual_risk_usd * self.usd_zar_rate

        # 8. Cost Calculation (Spread + Commission)
        # Spread Cost = (SpreadPriceDiff * Contract * Lots)
        spread_cost_usd = spread_val_usd * contract_size * lots
        # Comm Cost = CommPerLot * Lots
        comm_cost_usd = comm_per_lot * lots

        total_cost_usd = spread_cost_usd + comm_cost_usd
        total_cost_zar = total_cost_usd * self.usd_zar_rate

        # 9. Small Account / Survival Logic
        survival_cap_zar = self.user_balance_zar * (MAX_SURVIVAL_CAP_PCT / 100)
        is_forced_trade = False

        # Small Account Logic
        if actual_risk_zar > risk_per_trade_zar:
            if self.user_balance_zar < SMALL_ACCOUNT_THRESHOLD_ZAR:
                if actual_risk_zar <= survival_cap_zar:
                    is_forced_trade = True
                else:
                    return None # Risk > 15% of account
            else:
                if lots == min_vol:
                    return None
                else:
                    pass

        # 10. Profit Est
        profit_usd = (sl_dist * 1.5) * lots * contract_size
        profit_zar = profit_usd * self.usd_zar_rate

        signal.update({
            'price': round(entry_price, decimals),
            'sl': round(sl_price, decimals),
            'tp': round(tp_price, decimals),
            'lot_size': lots,
            'risk_zar': round(actual_risk_zar, 2),
            'profit_zar': round(profit_zar, 2),
            'spread_cost_zar': round(total_cost_zar, 2),
            'is_high_risk': symbol in HIGH_RISK_SYMBOLS,
            'is_small_account_boost': is_forced_trade
        })
        return signal

    def is_on_cooldown(self, symbol: str) -> bool:
        last_time = self.signal_history.get(symbol, 0)
        return (time.time() - last_time) < PAIR_SIGNAL_COOLDOWN

    def _prepare_data(self, klines: List) -> Optional[pd.DataFrame]:
        try:
            df = pd.DataFrame(klines)
            if df.empty: return None
            df = df.sort_values('time').reset_index(drop=True)
            return TechnicalAnalyzer.calculate_indicators(df)
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