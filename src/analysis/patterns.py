from typing import Dict
import pandas as pd

class FakeBreakoutDetector:
    """
    Analyzes price action for trap patterns.
    Focuses on 'Wick Rejection' and Volume Anomalies.
    """

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Scans the latest candles for signs of market manipulation or exhaustion.

        Args:
            df (pd.DataFrame): Market data with OHLCV.

        Returns:
            Dict: Contains 'is_fake' (bool), 'risk_score' (int), and 'reasons' (list).
        """
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        risk_score = 0
        reasons = []

        # --- Wick Rejection Analysis ---
        # Calculate wick sizes relative to body
        body_size = abs(curr['close'] - curr['open'])
        upper_wick = curr['high'] - max(curr['close'], curr['open'])
        lower_wick = min(curr['close'], curr['open']) - curr['low']

        # Bull Trap: Price broke High but closed low with huge upper wick
        if curr['close'] > prev['high'] and upper_wick > body_size * 1.5:
            risk_score += 40
            reasons.append('WICK_REJECTION_TOP')

        # Bear Trap: Price broke Low but closed high with huge lower wick
        if curr['close'] < prev['low'] and lower_wick > body_size * 1.5:
            risk_score += 40
            reasons.append('WICK_REJECTION_BOTTOM')

        # --- Volume Divergence ---
        # Price makes new highs on dropping volume indicates weakness
        if curr['high'] > prev['high'] and curr['volume'] < curr['vol_sma']:
            risk_score += 30
            reasons.append('LOW_VOL_BREAKOUT')

        # --- Over-extension (Mean Reversion) ---
        # Price is statistically too far from the average (Rubber band snap-back risk)
        dist_from_mean = abs(curr['close'] - curr['sma20']) / curr['sma20'] * 100
        if dist_from_mean > 3.0:  # > 3.0% deviation
            risk_score += 20
            reasons.append('OVEREXTENDED')

        return {
            'is_fake': risk_score >= 50,
            'risk_score': risk_score,
            'reasons': reasons
        }