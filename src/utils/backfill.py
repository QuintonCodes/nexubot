import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import List, Optional, Tuple, Dict

from src.analysis.indicators import TechnicalAnalyzer
from src.data.collector import DataCollector
from src.data.provider import DataProvider
from src.engine.strategies import StrategyAnalyzer
from src.config import (
    FALLBACK_CRYPTO,
    FALLBACK_FOREX,
    FALLBACK_INDICES,
    FALLBACK_METALS,
    HIGH_VOLATILITY_IDENTIFIERS,
    MIN_RR,
    SESSION_CONFIG,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)

# Constants for backfill limits
MAX_TOTAL_ROWS = 14000
MAX_ROWS_PER_SYMBOL = 1500
FUTURE_WINDOW_SIZE = 400
WARMUP_PERIOD = 200


async def backfill_data(provider: DataProvider, target_symbols: Optional[List[str]] = None):
    """
    Simulates historical price action to build a reliable ML dataset.
    Fully synchronized with live structural logic (SMC) via centralized indicators.
    """
    strategy_analyzer = StrategyAnalyzer()
    collector = DataCollector()
    total_rows_collected = 0

    symbols = target_symbols or (FALLBACK_CRYPTO + FALLBACK_FOREX + FALLBACK_INDICES + FALLBACK_METALS)
    logger.info(f"🔄 Starting SMC Backfill for {len(symbols)} symbols...")

    for symbol in symbols:
        if total_rows_collected >= MAX_TOTAL_ROWS:
            logger.info(f"🎯 Global limit of {MAX_TOTAL_ROWS} rows reached. Stopping backfill.")
            break

        logger.info(f"⏳ Backfilling {symbol}...")
        data_bundle = await _prepare_symbol_data(provider, symbol)
        if not data_bundle:
            continue

        df_full, records, point, tick_value, is_volatile_asset = data_bundle
        symbol_rows_collected = 0

        # --- SIMULATION ENGINE ---
        for i in range(WARMUP_PERIOD, len(records) - FUTURE_WINDOW_SIZE):
            if symbol_rows_collected >= MAX_ROWS_PER_SYMBOL:
                logger.info(f"✅ {symbol} reached {MAX_ROWS_PER_SYMBOL} row cap.")
                break

            curr = records[i]
            close_price = curr["close"]
            atr = max(float(curr.get("atr", 1.0)), 1.0)

            # Slicing for logic processing
            df_slice = df_full.iloc[max(0, i - WARMUP_PERIOD) : i + 1]
            records_slice = records[max(0, i - 100) : i + 1]

            # SMC Detection
            structure_info = TechnicalAnalyzer.detect_structure(df_slice)
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.extract_active_pois(records_slice)
            daily_levels = {"pdh": curr.get("pdh"), "pdl": curr.get("pdl")}
            is_liquidity_swept, sweep_depth_atr = TechnicalAnalyzer.detect_liquidity_sweeps(
                curr, structure_info, daily_levels
            )

            # Routing
            signal = strategy_analyzer.analyze_router(
                curr, active_fvgs, active_ifvgs, active_obs, structure_info, is_liquidity_swept
            )

            if not signal:
                continue

            # Trade Parameters & Filtering
            trade_params = _calculate_risk_and_tp(
                signal, close_price, atr, point, is_volatile_asset, active_ifvgs, active_obs
            )

            if not trade_params:
                continue  # Skipped due to MIN_RR filter

            sl, tp = trade_params

            # Feature Extraction
            features = _extract_smc_features(
                curr,
                signal,
                active_fvgs,
                active_ifvgs,
                active_obs,
                structure_info,
                is_liquidity_swept,
                sweep_depth_atr,
                atr,
            )

            if not features:
                continue  # Skipped due to POI exhaustion

            # Forward Simulation
            future_window = records[i + 1 : i + 1 + FUTURE_WINDOW_SIZE]
            won, pnl, target_excursion = _simulate_trade_outcome(
                future_window, signal["direction"], close_price, sl, tp, point, tick_value, atr
            )

            collector.log_training_data(symbol, features, won, pnl, target_excursion)
            symbol_rows_collected += 1
            total_rows_collected += 1

    logger.info("✅ SMC Backfill Complete. New dataset generated.")


async def _prepare_symbol_data(
    provider: DataProvider, symbol: str
) -> Optional[Tuple[pd.DataFrame, List[Dict], float, float, bool]]:
    """Fetches MT5 specs and synchronizes multi-timeframe data for a symbol."""
    mt5.symbol_select(symbol, True)

    symbol_info = await provider.get_symbol_info(symbol)
    if not symbol_info:
        logger.warning(f"⚠️ Could not fetch MT5 specs for {symbol}. Skipping.")
        return None

    point = symbol_info.get("point", 0.00001)
    tick_value = symbol_info.get("trade_tick_value", 1.0)
    is_volatile_asset = any(x in symbol for x in HIGH_VOLATILITY_IDENTIFIERS)

    klines_main = await provider.fetch_klines(symbol, TIMEFRAME, 50000)
    klines_htf = await provider.fetch_klines(symbol, "1h", 15000)

    if not klines_main or len(klines_main) < 1000:
        logger.warning(f"⚠️ Not enough data for {symbol}. Skipping.")
        return None

    df_full = pd.DataFrame(klines_main).sort_values("time").reset_index(drop=True)
    df_htf = pd.DataFrame(klines_htf).sort_values("time").reset_index(drop=True)

    df_full = TechnicalAnalyzer.calculate_indicators(df_full)
    df_htf = TechnicalAnalyzer.calculate_indicators(df_htf)

    # Vectorized Daily Levels
    df_full["date"] = pd.to_datetime(df_full["time"], unit="s").dt.date
    daily_highs = df_full.groupby("date")["high"].max().shift(1)
    daily_lows = df_full.groupby("date")["low"].min().shift(1)
    df_full["pdh"] = df_full["date"].map(daily_highs)
    df_full["pdl"] = df_full["date"].map(daily_lows)

    # Vectorized Timeframe Sync (Mapped without lookahead)
    df_full = df_full.sort_values("time")
    df_htf = df_htf.sort_values("time")

    df_full = pd.merge_asof(
        df_full,
        df_htf[["time", "htf_trend"]].rename(columns={"htf_trend": "htf_trend_mapped"}),
        on="time",
        direction="backward",
    )
    df_full["htf_trend_mapped"] = df_full["htf_trend_mapped"].fillna(0.0)

    records = df_full.to_dict("records")

    return df_full, records, point, tick_value, is_volatile_asset


def _calculate_risk_and_tp(
    signal: Dict,
    close_price: float,
    atr: float,
    point: float,
    is_volatile_asset: bool,
    active_ifvgs: List[Dict],
    active_obs: List[Dict],
) -> Optional[Tuple[float, float]]:
    """Calculates Stop Loss and Take Profit distances, returning None if MIN_RR isn't met."""
    sl_multiplier = 1.3 if (is_volatile_asset or atr > (close_price * 0.005)) else 1.0
    sl_dist = max(atr * sl_multiplier, point * 50)

    if "suggested_sl" in signal:
        suggested_dist = abs(signal["suggested_sl"] - close_price)
        if (atr * 0.2) < suggested_dist < (atr * 6.0):
            sl_dist = suggested_dist

    hard_blockades = active_ifvgs + [ob for ob in active_obs if ob.get("tier") == "MAJOR"]

    if signal["direction"] == "LONG":
        opposing_pois = [p["low"] for p in hard_blockades if p["type"] == "BEAR" and p["low"] > close_price]
        nearest_opposing_poi = min(opposing_pois) if opposing_pois else None
    else:
        opposing_pois = [p["high"] for p in hard_blockades if p["type"] == "BULL" and p["high"] < close_price]
        nearest_opposing_poi = max(opposing_pois) if opposing_pois else None

    base_tp_dist = max(atr * 4.0, sl_dist * MIN_RR)

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

    rr = (actual_tp_dist / sl_dist) if sl_dist > 0 else 1.0
    if rr < MIN_RR:
        return None

    return sl, tp


def _extract_smc_features(
    curr: Dict,
    signal: Dict,
    active_fvgs: List[Dict],
    active_ifvgs: List[Dict],
    active_obs: List[Dict],
    structure_info: Dict,
    is_liquidity_swept: bool,
    sweep_depth_atr: float,
    atr: float,
) -> Optional[Dict[str, float]]:
    """Compiles the feature set for the ML model. Returns None if the trade should be skipped."""
    close_price = curr["close"]
    htf_trend = curr["htf_trend_mapped"]

    # Alignment Score
    alignment_score = 0.0
    if htf_trend != 0.0:
        is_long_aligned = signal["direction"] == "LONG" and htf_trend == 1.0
        is_short_aligned = signal["direction"] == "SHORT" and htf_trend == -1.0
        alignment_score = 1.0 if (is_long_aligned or is_short_aligned) else -1.0

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
        nearest_poi = min(all_zones, key=lambda x: min(abs(x["high"] - close_price), abs(x["low"] - close_price)))
        raw_distance = min(abs(nearest_poi["high"] - close_price), abs(nearest_poi["low"] - close_price))
        dist_nearest_poi_atr = raw_distance / atr
        mitigation_count = nearest_poi.get("mitigations", 0)

    # Skip exhausted POI zones
    if mitigation_count > 2:
        return None

    return {
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


def _simulate_trade_outcome(
    future_window: List[Dict],
    direction: str,
    close_price: float,
    sl: float,
    tp: float,
    point: float,
    tick_value: float,
    atr: float,
) -> Tuple[int, float, float]:
    """Iterates through the future window to determine if the trade hits SL or TP."""
    won = 0
    max_favorable = 0.0
    lot_size = 0.01

    for f_curr in future_window:
        if direction == "LONG":
            max_favorable = max(max_favorable, f_curr["high"] - close_price)
            if f_curr["low"] <= sl:
                break
            if f_curr["high"] >= tp:
                won = 1
                break
        else:
            max_favorable = max(max_favorable, close_price - f_curr["low"])
            if f_curr["high"] >= sl:
                break
            if f_curr["low"] <= tp:
                won = 1
                break

    if won == 1:
        gross_points = abs(tp - close_price) / point
        pnl = gross_points * tick_value * lot_size
    else:
        gross_points = abs(sl - close_price) / point
        pnl = -gross_points * tick_value * lot_size

    target_excursion = max_favorable / atr if atr > 0 else 0.0

    return won, pnl, target_excursion
