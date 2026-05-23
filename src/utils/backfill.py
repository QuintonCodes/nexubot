import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import List, Optional

from src.analysis.indicators import TechnicalAnalyzer
from src.data.collector import DataCollector
from src.config import (
    FALLBACK_CRYPTO,
    FALLBACK_FOREX,
    FALLBACK_INDICES,
    FALLBACK_METALS,
    HIGH_VOLATILITY_IDENTIFIERS,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)


async def backfill_data(provider, engine, target_symbols: Optional[List[str]] = None):
    """
    Simulates historical price action to build a reliable ML dataset.
    Fully synchronized with live structural TP capping and RR filters.
    """

    collector = DataCollector()
    total_rows_collected = 0
    max_total_rows = 14000
    max_rows_per_symbol = 1500
    future_window_size = 150

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
        tick_value = symbol_info.get("trade_tick_value", 1.0)
        is_volatile_asset = any(x in symbol for x in HIGH_VOLATILITY_IDENTIFIERS)

        # Fetch M1 Data (Main execution TF) and H1 Data (HTF Bias)
        klines_main = await provider.fetch_klines(symbol, TIMEFRAME, 50000)
        klines_htf = await provider.fetch_klines(symbol, "1h", 1000)

        if not klines_main or len(klines_main) < 1000:
            logger.warning(f"⚠️ Not enough data for {symbol}. Skipping.")
            continue

        df_full = pd.DataFrame(klines_main).sort_values("time").reset_index(drop=True)
        df_htf = pd.DataFrame(klines_htf).sort_values("time").reset_index(drop=True)

        # 1. Pre-calculate all indicators (Now includes Lookback Windows)
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

        records = df_full.to_dict("records")
        symbol_rows_collected = 0

        for i in range(200, len(records) - future_window_size):
            if symbol_rows_collected >= max_rows_per_symbol:
                logger.info(f"✅ {symbol} reached {max_rows_per_symbol} row cap.")
                break

            curr = records[i]

            records_slice = records[max(0, i - 100) : i + 1]
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.extract_active_pois(records_slice)

            structure_info = {
                "bos": curr["bos"],
                "choch": curr["choch"],
                "structure": curr["structure"],
                "last_high": curr["last_high"],
                "last_low": curr["last_low"],
            }
            htf_trend = curr["htf_trend"]
            daily_levels = {"pdh": curr.get("pdh"), "pdl": curr.get("pdl")}

            is_liquidity_swept = TechnicalAnalyzer.detect_liquidity_sweeps(curr, structure_info, daily_levels)

            signal = engine.strategy_analyzer.analyze_router(
                curr, active_fvgs, active_ifvgs, active_obs, structure_info, is_liquidity_swept
            )

            if not signal:
                continue

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

            entry_price = curr["close"]
            atr = curr["atr"]

            # --- SYNCHRONIZED RISK & TP CAPPING ---
            sl_multiplier = 1.3 if (is_volatile_asset or atr > (curr["close"] * 0.005)) else 1.0
            sl_dist = max(atr * sl_multiplier, point * 50)

            if "suggested_sl" in signal:
                suggested_dist = abs(signal["suggested_sl"] - entry_price)
                if (atr * 0.2) < suggested_dist < (atr * 6.0):
                    sl_dist = suggested_dist

            all_zones = active_fvgs + active_ifvgs + active_obs
            opposing_pois = []
            if signal["direction"] == "LONG":
                opposing_pois = [p["low"] for p in all_zones if p["type"] == "BEAR" and p["low"] > entry_price]
            else:
                opposing_pois = [p["high"] for p in all_zones if p["type"] == "BULL" and p["high"] < entry_price]

            nearest_opposing_poi = (
                min(opposing_pois)
                if signal["direction"] == "LONG" and opposing_pois
                else (max(opposing_pois) if signal["direction"] == "SHORT" and opposing_pois else None)
            )
            base_tp_dist = max(atr * 3.0, sl_dist * 1.2)

            if signal["direction"] == "LONG":
                sl = entry_price - sl_dist
                max_structural_tp = (
                    (nearest_opposing_poi - point * 15) if nearest_opposing_poi else (entry_price + base_tp_dist)
                )
                tp = min(entry_price + base_tp_dist, max_structural_tp)
                actual_tp_dist = tp - entry_price
            else:
                sl = entry_price + sl_dist
                max_structural_tp = (
                    (nearest_opposing_poi + point * 15) if nearest_opposing_poi else (entry_price - base_tp_dist)
                )
                tp = max(entry_price - base_tp_dist, max_structural_tp)
                actual_tp_dist = entry_price - tp

            # --- HARD RR FILTER ---
            rr = (actual_tp_dist / sl_dist) if sl_dist > 0 else 1.0
            if rr < 1.5:
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

            features = {
                "is_htf_aligned": alignment,
                "is_liquidity_swept": is_liquidity_swept,
                "is_in_fvg": 1 if any(f["low"] <= curr["close"] <= f["high"] for f in active_fvgs) else 0,
                "is_in_ifvg": 1 if any(i_f["low"] <= curr["close"] <= i_f["high"] for i_f in active_ifvgs) else 0,
                "is_in_orderblock": 1 if any(o["low"] <= curr["close"] <= o["high"] for o in active_obs) else 0,
                "structural_break": 1 if structure_info["bos"] else (2 if structure_info["choch"] else 0),
                "session_volume_spike": 1 if curr["volume"] > (curr.get("vol_sma_20", 0) * 1.5) else 0,
                "distance_to_poi": dist_nearest_poi,
            }

            # Forward Simulation (Tracking exactly next 150 candles)
            future_window = records[i + 1 : i + 1 + future_window_size]

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
