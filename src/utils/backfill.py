import logging
import pandas as pd
import MetaTrader5 as mt5
from typing import List, Optional

from src.data.provider import DataProvider
from src.engine.ai_engine import AITradingEngine
from src.data.collector import DataCollector
from src.analysis.indicators import TechnicalAnalyzer
from src.analysis.candle_sticks import CandleStickDetector
from src.config import FALLBACK_CRYPTO, FALLBACK_FOREX

logger = logging.getLogger(__name__)


async def backfill_data(provider, engine, target_symbols: Optional[List[str]] = None):
    """
    Simulates historical price action to build a reliable ML dataset
    using the new SMC logic and strict feature sets.
    """
    collector = DataCollector()

    symbols = []
    if target_symbols:
        symbols = target_symbols
    else:
        logger.info("✅ Fetching User's Market Watch...")
        dynamic_symbols = await provider.get_dynamic_symbols()
        symbols = dynamic_symbols.get("crypto", []) + dynamic_symbols.get("forex", [])

    if not symbols:
        symbols = FALLBACK_CRYPTO + FALLBACK_FOREX

    logger.info(f"🔄 Starting SMC Backfill for {len(symbols)} symbols...")

    for symbol in symbols:
        logger.info(f"⏳ Backfilling {symbol}...")

        # Force MT5 to recognize the symbol in Market Watch
        mt5.symbol_select(symbol, True)

        # Fetch large history for robust simulation (5000 candles on M15)
        klines = await provider.fetch_klines(symbol, "M15", 5000)
        if not klines or len(klines) < 200:
            logger.warning(f"⚠️ Not enough data for {symbol}. Skipping.")
            continue

        # Load into DataFrame and pre-calculate indicators for blazing fast simulation
        df_full = pd.DataFrame(klines)
        df_full = df_full.sort_values("time").reset_index(drop=True)

        df_full = TechnicalAnalyzer.calculate_indicators(df_full, heavy=True)
        df_full = CandleStickDetector.calculate_candles(df_full)

        # Stateful memory for this specific symbol during the simulation
        active_fvgs = []
        active_obs = []

        # Iterate through historical data (Start at 200 to allow indicators to warm up)
        # End 50 candles early to allow forward-looking for Trade Outcome Simulation
        for i in range(200, len(df_full) - 50):
            window = df_full.iloc[:i].copy()
            curr = window.iloc[-1]

            # -----------------------------------------------------------------
            # 1. Update SMC Stateful Memory (FVGs)
            # -----------------------------------------------------------------
            if len(window) > 3:
                c1, c2, c3 = window.iloc[-3], window.iloc[-2], window.iloc[-1]
                # Bullish FVG (Gap between C1 High and C3 Low)
                if c1["high"] < c3["low"] and c2["close"] > c2["open"]:
                    active_fvgs.append({"type": "BULL", "high": c3["low"], "low": c1["high"]})
                # Bearish FVG (Gap between C1 Low and C3 High)
                elif c1["low"] > c3["high"] and c2["close"] < c2["open"]:
                    active_fvgs.append({"type": "BEAR", "high": c1["low"], "low": c3["high"]})

            # Clean up mitigated FVGs (Price swept completely through them)
            active_fvgs = [
                fvg
                for fvg in active_fvgs
                if (fvg["type"] == "BULL" and curr["low"] > fvg["low"])
                or (fvg["type"] == "BEAR" and curr["high"] < fvg["high"])
            ]
            active_fvgs = active_fvgs[-5:]  # Keep only the 5 most recent to prevent bloat

            # -----------------------------------------------------------------
            # 2. Determine HTF Trend (Simulation Proxy)
            # -----------------------------------------------------------------
            htf_trend = (
                "BULL" if curr["close"] > curr["ema_200"] else ("BEAR" if curr["close"] < curr["ema_200"] else "FLAT")
            )
            adx_strength = curr["adx"]

            # -----------------------------------------------------------------
            # 3. Strategy Router Execution
            # -----------------------------------------------------------------
            signal = engine.strategy_analyzer.analyze_router(
                curr, window, htf_trend, active_fvgs, active_obs, adx_strength
            )

            if not signal:
                continue

            # -----------------------------------------------------------------
            # 4. ML Feature Extraction (Strict 8 Features)
            # -----------------------------------------------------------------
            dt = pd.to_datetime(curr["time"], unit="s")
            dist_vwap = (curr["close"] - curr["vwap"]) / curr["vwap"] if curr["vwap"] != 0 else 0.0
            mtf_align = (
                1
                if (signal["direction"] == "LONG" and htf_trend == "BULL")
                else (-1 if (signal["direction"] == "SHORT" and htf_trend == "BEAR") else 0)
            )

            avg_atr = window["atr"].tail(24).mean()
            vol_ratio = curr["atr"] / avg_atr if avg_atr > 0 else 1.0

            dist_nearest_fvg = 0.0
            if active_fvgs:
                nearest = min(active_fvgs, key=lambda x: abs(x["high"] - curr["close"]))
                dist_nearest_fvg = abs(nearest["high"] - curr["close"]) / curr["close"]

            features = {
                "dist_to_vwap": dist_vwap,
                "mtf_trend_alignment": mtf_align,
                "hour_norm": dt.hour / 24.0,
                "volatility_ratio": vol_ratio,
                "dist_to_nearest_fvg": dist_nearest_fvg,
                "is_in_breaker": 0.0,  # Placeholder until OB logic expands
                "htf_adx_strength": adx_strength,
                "poi_status": 1.0 if len(active_fvgs) > 0 else 0.0,
            }

            # -----------------------------------------------------------------
            # 5. Forward Simulation (Outcome resolution)
            # -----------------------------------------------------------------
            future_window = df_full.iloc[i : i + 50]  # Look ahead next 50 candles
            entry_price = curr["close"]
            atr = curr["atr"]

            # Dynamic SL based on strategy suggestion or default ATR
            sl = signal.get(
                "suggested_sl",
                entry_price - (atr * 1.5) if signal["direction"] == "LONG" else entry_price + (atr * 1.5),
            )

            # Simulated 2:1 RR
            tp_dist = abs(entry_price - sl) * 2.0
            tp = entry_price + tp_dist if signal["direction"] == "LONG" else entry_price - tp_dist

            won = 0
            pnl = 0.0
            max_favorable = 0.0

            # Step through the future to find what gets hit first
            for _, f_curr in future_window.iterrows():
                if signal["direction"] == "LONG":
                    max_favorable = max(max_favorable, f_curr["high"] - entry_price)
                    if f_curr["low"] <= sl:
                        won = 0
                        pnl = sl - entry_price
                        break
                    if f_curr["high"] >= tp:
                        won = 1
                        pnl = tp - entry_price
                        break
                else:
                    max_favorable = max(max_favorable, entry_price - f_curr["low"])
                    if f_curr["high"] >= sl:
                        won = 0
                        pnl = entry_price - sl
                        break
                    if f_curr["low"] <= tp:
                        won = 1
                        pnl = entry_price - tp
                        break

            target_excursion = max_favorable / atr if atr > 0 else 0.0

            # -----------------------------------------------------------------
            # 6. Log Validated Data
            # -----------------------------------------------------------------
            collector.log_training_data(symbol=symbol, features=features, won=won, pnl=pnl, excursion=target_excursion)

    logger.info("✅ SMC Backfill Complete. New 'training_data.csv' generated securely.")
    await provider.shutdown()
