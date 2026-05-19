import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import List, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.data.collector import DataCollector
from src.config import FALLBACK_CRYPTO, FALLBACK_FOREX, FALLBACK_INDICES, FALLBACK_METALS, TIMEFRAME

logger = logging.getLogger(__name__)


async def backfill_data(provider, engine, target_symbols: Optional[List[str]] = None):
    """
    Simulates historical price action to build a reliable ML dataset
    using the new SMC logic and strict feature sets.
    """
    collector = DataCollector()
    total_rows_collected = 0
    max_total_rows = 14000
    max_rows_per_symbol = 2000

    # Prioritize focused targets
    symbols = target_symbols or (FALLBACK_CRYPTO + FALLBACK_FOREX + FALLBACK_INDICES + FALLBACK_METALS)

    logger.info(f"🔄 Starting SMC Backfill for {len(symbols)} symbols...")

    for symbol in symbols:
        if total_rows_collected >= max_total_rows:
            logger.info(f"🎯 Global limit of {max_total_rows} rows reached. Stopping backfill.")
            break

        logger.info(f"⏳ Backfilling {symbol}...")

        # Force MT5 to recognize the symbol in Market Watch
        mt5.symbol_select(symbol, True)

        # Fetch M1 Data (Main execution TF) and H1 Data (HTF Bias)
        klines_main = await provider.fetch_klines(symbol, TIMEFRAME, 50000)
        klines_htf = await provider.fetch_klines(symbol, "1h", 1000)

        if not klines_main or len(klines_main) < 1000:
            logger.warning(f"⚠️ Not enough data for {symbol}. Skipping.")
            continue

        # Load into DataFrame and pre-calculate indicators for blazing fast simulation
        df_full = pd.DataFrame(klines_main).sort_values("time").reset_index(drop=True)
        df_htf = pd.DataFrame(klines_htf).sort_values("time").reset_index(drop=True)

        # Pre-calculate indicators
        df_full = TechnicalAnalyzer.calculate_indicators(df_full)
        df_htf = TechnicalAnalyzer.calculate_indicators(df_htf)

        symbol_rows_collected = 0

        # Iterate through historical data
        # Start at 200 to allow warming up, end 200 early to allow forward-looking for outcome
        for i in range(200, len(df_full) - 50):
            if symbol_rows_collected >= max_rows_per_symbol:
                logger.info(f"✅ {symbol} reached {max_rows_per_symbol} row cap.")
                break

            window = df_full.iloc[:i].copy()
            curr = window.iloc[-1]

            #   -----------------------------------------------------------------
            # 1. Stateless SMC Detection (Mirrors Live Engine)
            #   -----------------------------------------------------------------
            active_fvgs = []
            active_obs = []

            if len(window) >= 20:
                for j in range(len(window) - 20, len(window) - 1):
                    c1, c2, c3 = window.iloc[j - 2], window.iloc[j - 1], window.iloc[j]

                    #   --- FVG Detection   ---
                    if c1["high"] < c3["low"] and c2["close"] > c2["open"]:
                        fvg = {"type": "BULL", "high": c3["low"], "low": c1["high"]}
                        if not any(window.iloc[k]["low"] < fvg["low"] for k in range(j + 1, len(window))):
                            active_fvgs.append(fvg)
                    elif c1["low"] > c3["high"] and c2["close"] < c2["open"]:
                        fvg = {"type": "BEAR", "high": c1["low"], "low": c3["high"]}
                        if not any(window.iloc[k]["high"] > fvg["high"] for k in range(j + 1, len(window))):
                            active_fvgs.append(fvg)

                    #   --- OB & Breaker Detection   ---
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
                                break
                        if not is_broken:
                            active_obs.append(ob)
            #   -----------------------------------------------------------------
            # 2. Extract Data & Route
            #   -----------------------------------------------------------------
            structure_info = TechnicalAnalyzer.detect_structure(window)

            # Align HTF trend by timestamp
            curr_time = curr["time"]
            htf_window = df_htf[df_htf["time"] <= curr_time]
            htf_trend = TechnicalAnalyzer.get_htf_trend(htf_window) if not htf_window.empty else "FLAT"

            signal = engine.strategy_analyzer.analyze_router(
                curr,
                window,
                htf_trend,
                active_fvgs,
                active_obs,
                structure_info,
            )

            if not signal:
                continue

            #   -----------------------------------------------------------------
            # 3. ML Feature Extraction
            #   -----------------------------------------------------------------
            all_pois = active_fvgs + active_obs
            dist_nearest_poi = 0.0
            if all_pois:
                nearest = min(
                    all_pois, key=lambda x: min(abs(x["high"] - curr["close"]), abs(x["low"] - curr["close"]))
                )
                dist_nearest_poi = (
                    min(abs(nearest["high"] - curr["close"]), abs(nearest["low"] - curr["close"])) / curr["close"]
                )

            strat_name = signal.get("strategy", "")

            features = {
                "is_htf_aligned": (
                    1
                    if (signal["direction"] == "LONG" and htf_trend == "BULL")
                    or (signal["direction"] == "SHORT" and htf_trend == "BEAR")
                    else -1
                ),
                "is_liquidity_swept": 1 if "Sweep" in strat_name else 0,
                "is_in_fvg": 1 if "FVG" in strat_name else 0,
                "is_in_orderblock": 1 if "OB" in strat_name else 0,
                "structural_break": 1 if structure_info["bos"] else (2 if structure_info["choch"] else 0),
                "session_volume_spike": 1 if curr["volume"] > window["volume"].tail(20).mean() * 1.5 else 0,
                "distance_to_poi": dist_nearest_poi,
            }

            # -----------------------------------------------------------------
            # 4. Forward Simulation (Outcome resolution on M1)
            # -----------------------------------------------------------------
            future_window = df_full.iloc[i : i + 200]  # Look ahead next 200 candles (~3 hours)
            entry_price = curr["close"]
            atr = curr["atr"]

            # Dynamic SL based on strategy suggestion
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

            #   -----------------------------------------------------------------
            # 6. Log Validated Data
            #   -----------------------------------------------------------------
            collector.log_training_data(symbol, features, won, pnl, target_excursion)
            symbol_rows_collected += 1
            total_rows_collected += 1

    logger.info("✅ SMC Backfill Complete. New dataset generated.")
