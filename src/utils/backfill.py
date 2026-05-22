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
    using highly optimized state-tracking and vectorized preprocessing.
    """

    collector = DataCollector()
    total_rows_collected = 0
    max_total_rows = 14000
    max_rows_per_symbol = 1500

    symbols = target_symbols or (FALLBACK_CRYPTO + FALLBACK_FOREX + FALLBACK_INDICES + FALLBACK_METALS)
    logger.info(f"🔄 Starting SMC Backfill for {len(symbols)} symbols...")

    for symbol in symbols:
        if total_rows_collected >= max_total_rows:
            logger.info(f"🎯 Global limit of {max_total_rows} rows reached. Stopping backfill.")
            break

        logger.info(f"⏳ Backfilling {symbol}...")
        mt5.symbol_select(symbol, True)

        symbol_info = await provider.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"⚠️ Could not fetch MT5 specs for {symbol}. Skipping.")
            continue

        point = symbol_info.get("point", 0.00001)
        tick_value = symbol_info.get("trade_tick_value", 0.0)

        # Fetch M1 Data (Main execution TF) and H1 Data (HTF Bias)
        klines_main = await provider.fetch_klines(symbol, TIMEFRAME, 50000)
        klines_htf = await provider.fetch_klines(symbol, "1h", 1000)

        if not klines_main or len(klines_main) < 1000:
            logger.warning(f"⚠️ Not enough data for {symbol}. Skipping.")
            continue

        # Load into DataFrame and pre-calculate indicators for blazing fast simulation
        df_full = pd.DataFrame(klines_main).sort_values("time").reset_index(drop=True)
        df_htf = pd.DataFrame(klines_htf).sort_values("time").reset_index(drop=True)

        # 1. Pre-calculate indicators
        df_full = TechnicalAnalyzer.calculate_indicators(df_full)
        df_htf = TechnicalAnalyzer.calculate_indicators(df_htf)

        # 2. Vectorized HTF Trend Mapping
        htf_trends = []
        last_hs, last_ls = [], []
        curr_htf = "FLAT"
        for row in df_htf.itertuples():
            if row.pivot_high:
                last_hs.append(row.high)
            if row.pivot_low:
                last_ls.append(row.low)
            if len(last_hs) >= 2 and len(last_ls) >= 2:
                if last_hs[-1] > last_hs[-2] and last_ls[-1] > last_ls[-2]:
                    curr_htf = "BULL"
                elif last_hs[-1] < last_hs[-2] and last_ls[-1] < last_ls[-2]:
                    curr_htf = "BEAR"
            htf_trends.append(curr_htf)

        df_htf["htf_trend"] = htf_trends
        df_full = pd.merge_asof(df_full, df_htf[["time", "htf_trend"]], on="time", direction="backward")
        df_full["htf_trend"] = df_full["htf_trend"].fillna("FLAT")

        # 3. Vectorized Daily Levels
        df_full["date"] = pd.to_datetime(df_full["time"], unit="s").dt.date
        daily_highs = df_full.groupby("date")["high"].max().shift(1)
        daily_lows = df_full.groupby("date")["low"].min().shift(1)
        df_full["pdh"] = df_full["date"].map(daily_highs)
        df_full["pdl"] = df_full["date"].map(daily_lows)

        # 4. Precalculate Local Structure for O(1) Lookups
        bos_list, choch_list, struct_list = [], [], []
        last_hs_m1, last_ls_m1 = [], []
        prev_h, last_h, prev_l, last_l = None, None, None, None

        for i in range(len(df_full)):
            if i >= 5:
                conf_row = df_full.iloc[i - 5]
                if conf_row["pivot_high"]:
                    prev_h, last_h = last_h, conf_row["high"]
                if conf_row["pivot_low"]:
                    prev_l, last_l = last_l, conf_row["low"]

            bos, choch, structure = None, None, "FLAT"
            if prev_h is not None and prev_l is not None:
                if last_h > prev_h and last_l > prev_l:
                    structure = "BULL"
                elif last_h < prev_h and last_l < prev_l:
                    structure = "BEAR"

                curr_close = df_full["close"].iloc[i]
                if structure == "BULL":
                    if curr_close > last_h:
                        bos = "BULL"
                    elif curr_close < last_l:
                        choch = "BEAR"
                elif structure == "BEAR":
                    if curr_close < last_l:
                        bos = "BEAR"
                    elif curr_close > last_h:
                        choch = "BULL"

            bos_list.append(bos)
            choch_list.append(choch)
            struct_list.append(structure)
            last_hs_m1.append(last_h)
            last_ls_m1.append(last_l)

        df_full["bos"] = bos_list
        df_full["choch"] = choch_list
        df_full["structure"] = struct_list
        df_full["last_high"] = last_hs_m1
        df_full["last_low"] = last_ls_m1

        df_full["recent_low_5"] = df_full["low"].rolling(5).min()
        df_full["recent_high_5"] = df_full["high"].rolling(5).max()
        df_full["recent_low_4"] = df_full["low"].rolling(4).min()
        df_full["recent_high_4"] = df_full["high"].rolling(4).max()
        df_full["vol_mean_20"] = df_full["volume"].rolling(20).mean()

        records = df_full.to_dict("records")

        symbol_rows_collected = 0
        active_fvgs = []
        active_ifvgs = []
        active_obs = []

        for i in range(2, len(records) - 50):
            if symbol_rows_collected >= max_rows_per_symbol:
                logger.info(f"✅ {symbol} reached {max_rows_per_symbol} row cap.")
                break

            c1 = records[i - 2]
            c2 = records[i - 1]
            curr = records[i]

            curr_low, curr_high = curr["low"], curr["high"]

            # 1. Manage FVGs & IFVGs
            for f in active_fvgs[:]:
                if f["type"] == "BULL" and curr_low < f["low"]:
                    active_ifvgs.append({"type": "BEAR", "high": f["high"], "low": f["low"]})
                    active_fvgs.remove(f)
                elif f["type"] == "BEAR" and curr_high > f["high"]:
                    active_ifvgs.append({"type": "BULL", "high": f["high"], "low": f["low"]})
                    active_fvgs.remove(f)

            active_ifvgs = [
                i_f
                for i_f in active_ifvgs
                if not (i_f["type"] == "BULL" and curr_low < i_f["low"])
                and not (i_f["type"] == "BEAR" and curr_high > i_f["high"])
            ]

            # Invalidate OBs
            active_obs = [
                o
                for o in active_obs
                if not (o["type"] == "BULL" and curr_low < o["low"])
                and not (o["type"] == "BEAR" and curr_high > o["high"])
            ]

            # Detect new FVG
            if c1["high"] < curr["low"] and c2["close"] > c2["open"]:
                active_fvgs.append({"type": "BULL", "high": curr["low"], "low": c1["high"]})
            elif c1["low"] > curr["high"] and c2["close"] < c2["open"]:
                active_fvgs.append({"type": "BEAR", "high": c1["low"], "low": curr["high"]})

            # Detect new OB and Tier
            is_pivot = c1.get("pivot_high", False) or c1.get("pivot_low", False)
            ob_tier = "MAJOR" if is_pivot else "INTERNAL"

            if c2["close"] > c2["open"] and c1["close"] < c1["open"] and c2["close"] > c1["high"]:
                active_obs.append({"type": "BULL", "high": c1["high"], "low": c1["low"], "tier": ob_tier})
            elif c2["close"] < c2["open"] and c1["close"] > c1["open"] and c2["close"] < c1["low"]:
                active_obs.append({"type": "BEAR", "high": c1["high"], "low": c1["low"], "tier": ob_tier})

            # Skip signal processing during warmup
            if i < 200:
                continue

            # Unified Analysis Router
            structure_info = {
                "bos": curr["bos"],
                "choch": curr["choch"],
                "structure": curr["structure"],
                "last_high": curr["last_high"],
                "last_low": curr["last_low"],
            }
            htf_trend = curr["htf_trend"]
            daily_levels = {"pdh": curr.get("pdh"), "pdl": curr.get("pdl")}

            # Provide a bounded dataframe slice for 50-period sweep detection
            df_slice = df_full.iloc[max(0, i - 100) : i + 1]

            is_liquidity_swept = TechnicalAnalyzer.detect_liquidity_sweeps(curr, df_slice, structure_info, daily_levels)
            df_dummy = pd.DataFrame()

            signal = engine.strategy_analyzer.analyze_router(
                curr, df_slice, htf_trend, active_fvgs, active_ifvgs, active_obs, structure_info, is_liquidity_swept
            )

            if not signal:
                continue

            # Precise Feature Extraction
            all_pois = active_fvgs + active_ifvgs + active_obs
            dist_nearest_poi = (
                min(
                    abs(
                        min(all_pois, key=lambda x: min(abs(x["high"] - curr["close"]), abs(x["low"] - curr["close"])))[
                            "high"
                        ]
                        - curr["close"]
                    ),
                    abs(
                        min(all_pois, key=lambda x: min(abs(x["high"] - curr["close"]), abs(x["low"] - curr["close"])))[
                            "low"
                        ]
                        - curr["close"]
                    ),
                )
                / curr["close"]
                if all_pois
                else 0.0
            )

            # Determine strict HTF alignment of the simulated trade direction
            alignment = 0
            if htf_trend != "FLAT":
                alignment = (
                    1
                    if (
                        (signal["direction"] == "LONG" and htf_trend == "BULL")
                        or (signal["direction"] == "SHORT" and htf_trend == "BEAR")
                    )
                    else -1
                )

            features = {
                "is_htf_aligned": alignment,
                "is_liquidity_swept": is_liquidity_swept,
                "is_in_fvg": 1 if any(f["low"] <= curr["close"] <= f["high"] for f in active_fvgs) else 0,
                "is_in_ifvg": 1 if any(i_f["low"] <= curr["close"] <= i_f["high"] for i_f in active_ifvgs) else 0,
                "is_in_orderblock": 1 if any(o["low"] <= curr["close"] <= o["high"] for o in active_obs) else 0,
                "structural_break": 1 if structure_info["bos"] else (2 if structure_info["choch"] else 0),
                "session_volume_spike": 1 if curr["volume"] > (curr["vol_mean_20"] * 1.5) else 0,
                "distance_to_poi": dist_nearest_poi,
            }

            # Forward Simulation
            future_window = records[i : i + 200]  # Look ahead next 200 candles (~3 hours)
            entry_price = curr["close"]
            atr = curr["atr"]

            # Dynamic SL based on strategy suggestion
            sl = signal.get(
                "suggested_sl",
                entry_price - (atr * 1.5) if signal["direction"] == "LONG" else entry_price + (atr * 1.5),
            )
            tp_dist = abs(entry_price - sl) * 2.0
            tp = entry_price + tp_dist if signal["direction"] == "LONG" else entry_price - tp_dist

            won = 0
            max_favorable = 0.0

            for f_curr in future_window:
                if signal["direction"] == "LONG":
                    max_favorable = max(max_favorable, f_curr["high"] - entry_price)
                    if f_curr["low"] <= sl:
                        won = 0
                        break
                    if f_curr["high"] >= tp:
                        won = 1
                        break
                else:
                    max_favorable = max(max_favorable, entry_price - f_curr["low"])
                    if f_curr["high"] >= sl:
                        won = 0
                        break
                    if f_curr["low"] <= tp:
                        won = 1
                        break

            lot_size = 0.01

            if won == 1:
                gross_points = abs(tp - entry_price) / point
                pnl = gross_points * tick_value * lot_size
            else:
                gross_points = abs(sl - entry_price) / point
                pnl = -gross_points * tick_value * lot_size

            target_excursion = max_favorable / atr if atr > 0 else 0.0

            collector.log_training_data(symbol, features, won, pnl, target_excursion)
            symbol_rows_collected += 1
            total_rows_collected += 1

    logger.info("✅ SMC Backfill Complete. New dataset generated.")
