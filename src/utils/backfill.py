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
    SESSION_CONFIG,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)


async def backfill_data(provider, engine, target_symbols: Optional[List[str]] = None):
    """
    Simulates historical price action to build a reliable ML dataset.
    Fully synchronized with live structural logic (SMC) via centralized indicators.
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
        klines_htf = await provider.fetch_klines(symbol, "1h", 15000)

        if not klines_main or len(klines_main) < 1000:
            logger.warning(f"⚠️ Not enough data for {symbol}. Skipping.")
            continue

        df_full = pd.DataFrame(klines_main).sort_values("time").reset_index(drop=True)
        df_htf = pd.DataFrame(klines_htf).sort_values("time").reset_index(drop=True)

        # 1. Pre-calculate standard trailing indicators
        df_full = TechnicalAnalyzer.calculate_indicators(df_full)
        df_htf = TechnicalAnalyzer.calculate_indicators(df_htf)

        # 2. Vectorized Daily Levels
        df_full["date"] = pd.to_datetime(df_full["time"], unit="s").dt.date
        daily_highs = df_full.groupby("date")["high"].max().shift(1)
        daily_lows = df_full.groupby("date")["low"].min().shift(1)
        df_full["pdh"] = df_full["date"].map(daily_highs)
        df_full["pdl"] = df_full["date"].map(daily_lows)

        # 3. Vectorized Timeframe Sync
        # This stamps every single 1M candle with the active 1H trend mathematically.
        df_full = df_full.sort_values("time")
        df_htf = df_htf.sort_values("time")

        df_full = pd.merge_asof(
            df_full,
            df_htf[["time", "htf_trend"]].rename(columns={"htf_trend": "htf_trend_mapped"}),
            on="time",
            direction="backward",
        )
        # Fill any early NaNs before the EMAs initialize
        df_full["htf_trend_mapped"] = df_full["htf_trend_mapped"].fillna(0.0)

        records = df_full.to_dict("records")
        symbol_rows_collected = 0

        # --- SIMULATION ENGINE ---
        # Begin simulation after 200 bars to allow sufficient warmup for structure
        for i in range(200, len(records) - future_window_size):
            if symbol_rows_collected >= max_rows_per_symbol:
                logger.info(f"✅ {symbol} reached {max_rows_per_symbol} row cap.")
                break

            curr = records[i]
            curr_time = curr["time"]
            close_price = curr["close"]

            atr = float(curr.get("atr", 1.0))
            if atr <= 0:
                atr = 1.0

            # 3. Dynamic Data Slices (Prevents Lookahead Bias & Reuses Central Logic)
            df_slice = df_full.iloc[max(0, i - 200) : i + 1]
            records_slice = records[max(0, i - 100) : i + 1]

            # Read the permanently stamped HTF trend instantly
            htf_trend = curr["htf_trend_mapped"]

            structure_info = TechnicalAnalyzer.detect_structure(df_slice)
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.extract_active_pois(records_slice)

            daily_levels = {"pdh": curr.get("pdh"), "pdl": curr.get("pdl")}
            is_liquidity_swept, sweep_depth_atr = TechnicalAnalyzer.detect_liquidity_sweeps(
                curr, structure_info, daily_levels
            )

            # 5. Routing
            signal = engine.strategy_analyzer.analyze_router(
                curr, active_fvgs, active_ifvgs, active_obs, structure_info, is_liquidity_swept
            )

            if not signal:
                continue

            # --- SYNCHRONIZED RISK & TP CAPPING ---
            sl_multiplier = 1.3 if (is_volatile_asset or atr > (close_price * 0.005)) else 1.0
            sl_dist = max(atr * sl_multiplier, point * 50)

            if "suggested_sl" in signal:
                suggested_dist = abs(signal["suggested_sl"] - close_price)
                if (atr * 0.2) < suggested_dist < (atr * 6.0):
                    sl_dist = suggested_dist

            # Only use IFVGs and Major OBs as walls
            hard_blockades = active_ifvgs + [ob for ob in active_obs if ob.get("tier") == "MAJOR"]
            opposing_pois = []

            if signal["direction"] == "LONG":
                opposing_pois = [p["low"] for p in hard_blockades if p["type"] == "BEAR" and p["low"] > close_price]
            else:
                opposing_pois = [p["high"] for p in hard_blockades if p["type"] == "BULL" and p["high"] < close_price]

            nearest_opposing_poi = (
                min(opposing_pois)
                if signal["direction"] == "LONG" and opposing_pois
                else (max(opposing_pois) if signal["direction"] == "SHORT" and opposing_pois else None)
            )
            base_tp_dist = max(atr * 3.0, sl_dist * 1.5)

            if signal["direction"] == "LONG":
                sl = close_price - sl_dist
                max_structural_tp = (
                    (nearest_opposing_poi - point * 15) if nearest_opposing_poi else (close_price + base_tp_dist)
                )
                tp = min(close_price + base_tp_dist, max_structural_tp)
                actual_tp_dist = tp - close_price
            else:
                sl = close_price + sl_dist
                max_structural_tp = (
                    (nearest_opposing_poi + point * 15) if nearest_opposing_poi else (close_price - base_tp_dist)
                )
                tp = max(close_price - base_tp_dist, max_structural_tp)
                actual_tp_dist = close_price - tp

            # --- HARD RR FILTER ---
            rr = (actual_tp_dist / sl_dist) if sl_dist > 0 else 1.0
            if rr < 1.5:
                continue

            # --- PRECISE FEATURE EXTRACTION ---
            # Calculate True Contextual Alignment
            alignment_score = 0.0
            if htf_trend != 0.0:
                is_long_aligned = signal["direction"] == "LONG" and htf_trend == 1.0
                is_short_aligned = signal["direction"] == "SHORT" and htf_trend == -1.0

                if is_long_aligned or is_short_aligned:
                    alignment_score = 1.0
                else:
                    alignment_score = -1.0

            # Killzone Map
            dt_hour = pd.to_datetime(curr["time"], unit="s").hour
            active_killzone = 0.0
            if SESSION_CONFIG["ASIAN_START"] <= dt_hour < SESSION_CONFIG["ASIAN_END"]:
                active_killzone = 1.0
            elif SESSION_CONFIG["LONDON_START"] <= dt_hour < SESSION_CONFIG["LONDON_END"]:
                active_killzone = 2.0
            elif SESSION_CONFIG["NY_START"] <= dt_hour < SESSION_CONFIG["NY_END"]:
                active_killzone = 3.0

            # Distance & Mitigation Tracking
            dist_nearest_poi_atr = 0.0
            mitigation_count = 0
            all_zones = active_fvgs + active_ifvgs + active_obs
            if all_zones:
                nearest_poi = min(
                    all_zones, key=lambda x: min(abs(x["high"] - close_price), abs(x["low"] - close_price))
                )
                raw_distance = min(abs(nearest_poi["high"] - close_price), abs(nearest_poi["low"] - close_price))
                dist_nearest_poi_atr = raw_distance / atr
                mitigation_count = nearest_poi.get("mitigations", 0)

            features = {
                "is_htf_aligned": alignment_score,
                "is_liquidity_swept": float(is_liquidity_swept),
                "is_in_fvg": 1.0 if any(f["low"] <= close_price <= f["high"] for f in active_fvgs) else 0.0,
                "is_in_ifvg": 1.0 if any(i_f["low"] <= close_price <= i_f["high"] for i_f in active_ifvgs) else 0.0,
                "is_in_orderblock": 1.0 if any(o["low"] <= close_price <= o["high"] for o in active_obs) else 0.0,
                "structural_break": structure_info.get("structural_break", 0.0),
                "active_killzone": active_killzone,
                "distance_to_poi": dist_nearest_poi_atr,
                "pd_array_status": structure_info.get("pd_array", 0.5),
                "mitigation_count": float(mitigation_count),
                "sweep_depth_atr": sweep_depth_atr,
            }

            # --- FORWARD SIMULATION (Next 150 Candles) ---
            future_window = records[i + 1 : i + 1 + future_window_size]
            won = 0
            max_favorable = 0.0

            for f_curr in future_window:
                if signal["direction"] == "LONG":
                    max_favorable = max(max_favorable, f_curr["high"] - close_price)
                    if f_curr["low"] <= sl:
                        won = 0
                        break
                    if f_curr["high"] >= tp:
                        won = 1
                        break
                else:
                    max_favorable = max(max_favorable, close_price - f_curr["low"])
                    if f_curr["high"] >= sl:
                        won = 0
                        break
                    if f_curr["low"] <= tp:
                        won = 1
                        break

            lot_size = 0.01

            if won == 1:
                gross_points = abs(tp - close_price) / point
                pnl = gross_points * tick_value * lot_size
            else:
                gross_points = abs(sl - close_price) / point
                pnl = -gross_points * tick_value * lot_size

            target_excursion = max_favorable / atr if atr > 0 else 0.0

            collector.log_training_data(symbol, features, won, pnl, target_excursion)
            symbol_rows_collected += 1
            total_rows_collected += 1

    logger.info("✅ SMC Backfill Complete. New dataset generated.")
