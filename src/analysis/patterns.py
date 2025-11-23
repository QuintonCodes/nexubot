from typing import Dict
import pandas as pd

class FakeBreakoutDetector:
    """
    Analyzes price action for trap patterns.
    Focuses on 'Wick Rejection' and Volume Anomalies.
    """

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Returns a risk dictionary.
        If risk_score > 50, it's likely a fakeout.
        """
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        risk_score = 0
        reasons = []

        # 1. Wick Rejection Analysis
        # If trying to go UP (green candle) but has huge top wick -> Selling pressure (Fakeout risk)
        body_size = abs(curr['close'] - curr['open'])
        upper_wick = curr['high'] - max(curr['close'], curr['open'])
        lower_wick = min(curr['close'], curr['open']) - curr['low']

        # Breakout Trap (Bullish Trap)
        if curr['close'] > prev['high'] and upper_wick > body_size * 1.5:
            risk_score += 40
            reasons.append('WICK_REJECTION_TOP')

        # Breakdown Trap (Bearish Trap)
        if curr['close'] < prev['low'] and lower_wick > body_size * 1.5:
            risk_score += 40
            reasons.append('WICK_REJECTION_BOTTOM')

        # 2. Volume Divergence
        # Price makes new high, but Volume is lower than average
        if curr['high'] > prev['high'] and curr['volume'] < curr['vol_sma']:
            risk_score += 30
            reasons.append('LOW_VOL_BREAKOUT')

        # 3. Over-extension (Rubber Band effect)
        # Price is too far from EMA 20 (Bollinger middle)
        dist_from_mean = abs(curr['close'] - curr['sma20']) / curr['sma20'] * 100
        if dist_from_mean > 2.5:  # > 2.5% away from mean
            risk_score += 20
            reasons.append('OVEREXTENDED')

        return {
            'is_fake': risk_score >= 50,
            'risk_score': risk_score,
            'reasons': reasons
        }