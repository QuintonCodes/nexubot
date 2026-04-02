import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import List, Optional

from src.analysis.candle_sticks import CandleStickDetector
from src.analysis.indicators import TechnicalAnalyzer
from src.data.collector import DataCollector
from src.config import FALLBACK_CRYPTO, FALLBACK_FOREX, FALLBACK_INDICES

logger = logging.getLogger(__name__)


async def backfill_data(provider, engine, target_symbols: Optional[List[str]] = None):
    """
    Simulates historical price action to build a reliable ML dataset
    using the new SMC logic and strict feature sets.
    """
    collector = DataCollector()
    total_rows_collected = 0
    max_total_rows = 20000
    max_rows_per_symbol = 1000

    symbols = []
    if target_symbols:
        symbols = target_symbols
    else:
        logger.info("✅ Fetching User's Market Watch...")
        dynamic_symbols = await provider.get_dynamic_symbols()
        symbols = (
            dynamic_symbols.get("crypto", []) + dynamic_symbols.get("forex", []) + dynamic_symbols.get("indices", [])
        )

    if not symbols:
        symbols = FALLBACK_CRYPTO + FALLBACK_FOREX + FALLBACK_INDICES

    logger.info(f"🔄 Starting SMC Backfill for {len(symbols)} symbols...")

    for symbol in symbols:
        if total_rows_collected >= max_total_rows:
            logger.info("🎯 Global limit of 20,000 rows reached. Stopping backfill.")
            break

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

        symbol_rows_collected = 0

        # Iterate through historical data (Start at 200 to allow indicators to warm up)
        # End 50 candles early to allow forward-looking for Trade Outcome Simulation
        for i in range(200, len(df_full) - 50):
            if symbol_rows_collected >= max_rows_per_symbol:
                logger.info(f"✅ {symbol} reached 1,000 row cap.")
                break

            window = df_full.iloc[:i].copy()
            curr = window.iloc[-1]

            # -----------------------------------------------------------------
            # 1. Stateless SMC Detection (Mirrors Live Engine)
            # -----------------------------------------------------------------
            active_fvgs = []
            active_obs = []
            active_breakers = []

            if len(window) >= 20:
                for j in range(len(window) - 20, len(window) - 1):
                    c1, c2, c3 = window.iloc[j - 2], window.iloc[j - 1], window.iloc[j]

                    # --- FVG Detection ---
                    if c1["high"] < c3["low"] and c2["close"] > c2["open"]:
                        fvg = {"type": "BULL", "high": c3["low"], "low": c1["high"]}
                        if not any(window.iloc[k]["low"] < fvg["low"] for k in range(j + 1, len(window))):
                            active_fvgs.append(fvg)
                    elif c1["low"] > c3["high"] and c2["close"] < c2["open"]:
                        fvg = {"type": "BEAR", "high": c1["low"], "low": c3["high"]}
                        if not any(window.iloc[k]["high"] > fvg["high"] for k in range(j + 1, len(window))):
                            active_fvgs.append(fvg)

                    # --- OB & Breaker Detection ---
                    ob_c1, ob_c2 = window.iloc[j - 1], window.iloc[j]
                    if (
                        ob_c1["close"] < ob_c1["open"]
                        and ob_c2["close"] > ob_c2["open"]
                        and ob_c2["close"] > ob_c1["high"]
                    ):
                        ob = {"type": "BULL", "high": ob_c1["high"], "low": ob_c1["low"]}
                        is_broken = False
                        for k in range(j + 1, len(window)):
                            if window.iloc[k]["low"] < ob["low"]:
                                is_broken = True
                                breaker = {"type": "BEAR", "high": ob["high"], "low": ob["low"]}
                                if not any(window.iloc[m]["high"] > breaker["high"] for m in range(k + 1, len(window))):
                                    active_breakers.append(breaker)
                                break
                        if not is_broken:
                            active_obs.append(ob)

                    elif (
                        ob_c1["close"] > ob_c1["open"]
                        and ob_c2["close"] < ob_c2["open"]
                        and ob_c2["close"] < ob_c1["low"]
                    ):
                        ob = {"type": "BEAR", "high": ob_c1["high"], "low": ob_c1["low"]}
                        is_broken = False
                        for k in range(j + 1, len(window)):
                            if window.iloc[k]["high"] > ob["high"]:
                                is_broken = True
                                breaker = {"type": "BULL", "high": ob["high"], "low": ob["low"]}
                                if not any(window.iloc[m]["low"] < breaker["low"] for m in range(k + 1, len(window))):
                                    active_breakers.append(breaker)
                                break
                        if not is_broken:
                            active_obs.append(ob)
            # -----------------------------------------------------------------
            # 2. Extract Data & Route
            # -----------------------------------------------------------------
            htf_trend = (
                "BULL" if curr["close"] > curr["ema_200"] else ("BEAR" if curr["close"] < curr["ema_200"] else "FLAT")
            )
            adx_strength = curr["adx"]
            structure_info = TechnicalAnalyzer.detect_structure(window)

            allowed_session_types = ["TREND", "REVERSION", "BREAKOUT"]

            signal = engine.strategy_analyzer.analyze_router(
                curr, window, htf_trend, active_fvgs, active_obs, adx_strength, structure_info, allowed_session_types
            )

            if not signal:
                continue

            # -----------------------------------------------------------------
            # 3. ML Feature Extraction
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

            is_in_breaker_val = 0.0
            if signal["direction"] == "LONG":
                if any(b["low"] <= curr["close"] <= b["high"] for b in active_breakers if b["type"] == "BULL"):
                    is_in_breaker_val = 1.0
            elif signal["direction"] == "SHORT":
                if any(b["low"] <= curr["close"] <= b["high"] for b in active_breakers if b["type"] == "BEAR"):
                    is_in_breaker_val = 1.0

            features = {
                "dist_to_vwap": dist_vwap,
                "mtf_trend_alignment": mtf_align,
                "hour_norm": dt.hour / 24.0,
                "volatility_ratio": vol_ratio,
                "dist_to_nearest_fvg": dist_nearest_fvg,
                "is_in_breaker": is_in_breaker_val,
                "htf_adx_strength": adx_strength,
                "poi_status": 1.0 if len(active_fvgs) > 0 or len(active_obs) > 0 else 0.0,
            }

            # -----------------------------------------------------------------
            # 4. Forward Simulation (Outcome resolution)
            # -----------------------------------------------------------------
            future_window = df_full.iloc[i : i + 50]  # Look ahead next 50 candles
            entry_price = curr["close"]
            atr = curr["atr"]

            # Relax SL multiplier for high volatility pairs during training
            is_high_vol = any(x in symbol for x in ["XAU", "XAG", "BTC", "US30", "NAS"])
            backfill_sl_mult = 2.0 if is_high_vol else 1.5

            # Dynamic SL based on strategy suggestion or default ATR
            sl = signal.get(
                "suggested_sl",
                (
                    entry_price - (atr * backfill_sl_mult)
                    if signal["direction"] == "LONG"
                    else entry_price + (atr * backfill_sl_mult)
                ),
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
            collector.log_training_data(symbol, features, won, pnl, target_excursion)
            symbol_rows_collected += 1
            total_rows_collected += 1

    logger.info("✅ SMC Backfill Complete. New dataset generated.")
