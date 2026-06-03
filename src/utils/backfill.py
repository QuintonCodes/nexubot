import logging
import MetaTrader5 as mt5
import pandas as pd
from typing import Dict, List, Optional, Tuple

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
    MAX_ROWS,
    MIN_RR,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)

# Constants for backfill limits
MAX_ROWS_PER_SYMBOL = 2000
MAX_ROWS_PER_STRATEGY = 2000
FUTURE_WINDOW_SIZE = 50  # 50 × 5min = ~4.2 hours, matches live timeout
WARMUP_PERIOD = 200  # EMA-200 needs 200 bars to converge; don't reduce


def _calculate_risk_and_tp(
    signal: Dict,
    close_price: float,
    atr: float,
    point: float,
    is_volatile_asset: bool,
    active_ifvgs: List[Dict],
    active_obs: List[Dict],
) -> Optional[Tuple[float, float, float]]:
    """Calculates Stop Loss and Take Profit distances, returning None if MIN_RR isn't met."""
    sl_multiplier = 1.3 if (is_volatile_asset or atr > (close_price * 0.005)) else 1.0
    sl_dist = max(atr * sl_multiplier, point * 50)

    if "suggested_sl" in signal:
        suggested_dist = abs(signal["suggested_sl"] - close_price)
        if (atr * 0.2) < suggested_dist < (atr * 6.0):
            sl_dist = suggested_dist

    hard_blockades = active_ifvgs + [ob for ob in active_obs if ob.get("tier") == "MAJOR"]

    base_tp_dist = max(atr * 4.0, sl_dist * MIN_RR)

    if signal["direction"] == "LONG":
        opposing_pois = [p["low"] for p in hard_blockades if p["type"] == "BEAR" and p["low"] > close_price]
        nearest_opposing_poi = min(opposing_pois) if opposing_pois else None

        sl = close_price - sl_dist
        max_structural_tp = (
            (nearest_opposing_poi - point * 15) if nearest_opposing_poi else (close_price + base_tp_dist)
        )
        tp = min(close_price + base_tp_dist, max_structural_tp)
        actual_tp_dist = tp - close_price
    else:
        opposing_pois = [p["high"] for p in hard_blockades if p["type"] == "BULL" and p["high"] < close_price]
        nearest_opposing_poi = max(opposing_pois) if opposing_pois else None

        sl = close_price + sl_dist
        max_structural_tp = (
            (nearest_opposing_poi + point * 15) if nearest_opposing_poi else (close_price - base_tp_dist)
        )
        tp = max(close_price - base_tp_dist, max_structural_tp)
        actual_tp_dist = close_price - tp

    rr = (actual_tp_dist / sl_dist) if sl_dist > 0 else 1.0
    if rr < MIN_RR:
        return None

    return sl, tp, rr


async def _prepare_symbol_data(
    provider: DataProvider, symbol: str
) -> Optional[Tuple[pd.DataFrame, List[Dict], float, bool]]:
    """Fetches MT5 specs and synchronizes multi-timeframe data for a symbol."""
    mt5.symbol_select(symbol, True)

    symbol_info = await provider.get_symbol_info(symbol)
    if not symbol_info:
        logger.warning(f"⚠️ Could not fetch MT5 specs for {symbol}. Skipping.")
        return None

    point = symbol_info.get("point", 0.00001)
    is_volatile_asset = any(x in symbol for x in HIGH_VOLATILITY_IDENTIFIERS)

    requested_m5 = 18000
    requested_h1 = 5000
    klines_main = await provider.fetch_klines(symbol, TIMEFRAME, requested_m5)
    klines_htf = await provider.fetch_klines(symbol, "1h", requested_h1)

    actual_m5 = len(klines_main) if klines_main else 0
    actual_h1 = len(klines_htf) if klines_htf else 0

    if actual_m5 < requested_m5:
        logger.info(f"📊 {symbol}: Requested {requested_m5} {TIMEFRAME} candles, broker capped at {actual_m5}.")

    if actual_m5 < 1000 or actual_h1 < 100:
        logger.warning(f"⚠️ Not enough data for {symbol} ({TIMEFRAME}: {actual_m5}, H1: {actual_h1}). Skipping.")
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
    return df_full, records, point, is_volatile_asset


def _simulate_trade_outcome(
    future_window: List[Dict],
    direction: str,
    close_price: float,
    sl: float,
    tp: float,
    atr: float,
) -> Tuple[int, float, float]:
    """Iterates through the future window to determine if the trade hits SL or TP."""
    won = 0
    max_favorable = 0.0
    actual_tp_dist = abs(tp - close_price)

    tp1 = close_price + (actual_tp_dist * 0.33) if direction == "LONG" else close_price - (actual_tp_dist * 0.33)
    tp2 = close_price + (actual_tp_dist * 0.66) if direction == "LONG" else close_price - (actual_tp_dist * 0.66)

    tp1_hit, tp2_hit, tp3_hit = False, False, False

    for f_curr in future_window:
        if direction == "LONG":
            max_favorable = max(max_favorable, f_curr["high"] - close_price)

            if f_curr["low"] <= sl:
                break

            if f_curr["high"] >= tp1 and not tp1_hit:
                tp1_hit = True
                sl = close_price + (atr * 0.15)
            if f_curr["high"] >= tp2 and not tp2_hit:
                tp2_hit = True
                sl = tp1
            if f_curr["high"] >= tp:
                tp3_hit = True
                break
        else:
            max_favorable = max(max_favorable, close_price - f_curr["low"])

            if f_curr["high"] >= sl:
                break

            if f_curr["low"] <= tp1 and not tp1_hit:
                tp1_hit = True
                sl = close_price - (atr * 0.15)
            if f_curr["low"] <= tp2 and not tp2_hit:
                tp2_hit = True
                sl = tp1
            if f_curr["low"] <= tp:
                tp3_hit = True
                break

    gross_dist = abs(sl - close_price)
    pnl = -gross_dist / atr if atr > 0 else 0.0

    if tp3_hit:
        won, pnl = 1, actual_tp_dist / atr
    elif tp2_hit:
        won, pnl = 1, abs(tp2 - close_price) / atr
    elif tp1_hit:
        won, pnl = 1, abs(tp1 - close_price) / atr

    target_excursion = max_favorable / atr if atr > 0 else 0.0

    return won, pnl, target_excursion


async def backfill_data(provider: DataProvider, target_symbols: Optional[List[str]] = None) -> None:
    """
    Simulates historical price action to build a reliable ML dataset.
    Fully synchronized with live structural logic (SMC) via centralized indicators.
    """
    strategy_analyzer = StrategyAnalyzer()
    collector = DataCollector()
    total_rows_collected = 0
    strategy_counts = {}

    symbols = target_symbols or (FALLBACK_CRYPTO + FALLBACK_FOREX + FALLBACK_INDICES + FALLBACK_METALS)
    logger.info(f"🔄 Starting SMC Backfill for {len(symbols)} symbols...")

    for symbol in symbols:
        if total_rows_collected >= MAX_ROWS:
            logger.info(f"🎯 Global limit of {MAX_ROWS} rows reached. Stopping backfill.")
            break

        logger.info(f"⏳ Backfilling {symbol}...")
        data_bundle = await _prepare_symbol_data(provider, symbol)
        if not data_bundle:
            continue

        df_full, records, point, is_volatile_asset = data_bundle
        symbol_rows_collected = 0

        active_fvgs, active_ifvgs, active_obs = [], [], []

        # --- SIMULATION ENGINE ---
        for i in range(2, len(records) - FUTURE_WINDOW_SIZE):
            c1, c2, curr = records[i - 2], records[i - 1], records[i]
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.update_pois(
                c1, c2, curr, active_fvgs, active_ifvgs, active_obs
            )

            # Prevent mapping until structure warms up fully
            if i < WARMUP_PERIOD:
                continue

            if symbol_rows_collected >= MAX_ROWS_PER_SYMBOL:
                logger.info(f"✅ {symbol} reached {MAX_ROWS_PER_SYMBOL} row cap.")
                break

            close_price = curr["close"]
            min_atr = point * 10
            atr = max(float(curr.get("atr", min_atr)), min_atr)

            # Slicing for logic processing
            df_slice = df_full.iloc[max(0, i - WARMUP_PERIOD) : i + 1]

            # SMC Detection
            structure_info = TechnicalAnalyzer.detect_structure(df_slice)
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

            # Evaluate strategy cap early to save computation
            strat = signal.get("strategy", "Unknown")
            if strategy_counts.get(strat, 0) >= MAX_ROWS_PER_STRATEGY:
                continue

            # Filter heavily marginal signals to avoid dataset degradation
            if signal.get("confidence", 0.0) < 60.0:
                continue

            # Trade Parameters & Filtering
            trade_params = _calculate_risk_and_tp(
                signal, close_price, atr, point, is_volatile_asset, active_ifvgs, active_obs
            )

            if not trade_params:
                continue  # Skipped due to MIN_RR filter

            sl, tp, rr = trade_params

            # Build feature dictionary using shared analyzer method
            raw_features = TechnicalAnalyzer.compile_features(
                curr=curr,
                signal_direction=signal["direction"],
                active_fvgs=active_fvgs,
                active_ifvgs=active_ifvgs,
                active_obs=active_obs,
                structure_info=structure_info,
                is_liquidity_swept=is_liquidity_swept,
                sweep_depth_atr=sweep_depth_atr,
                atr=atr,
                rr_at_entry=rr,
            )

            # Engineer features via the new Matrix
            engineered_features = collector.engineer_features(raw_features)

            if raw_features.get("mitigation_count", 0) > 2:
                continue

            # Forward Simulation
            future_window = records[i + 1 : i + 1 + FUTURE_WINDOW_SIZE]

            won, pnl, target_excursion = _simulate_trade_outcome(
                future_window, signal["direction"], close_price, sl, tp, atr
            )

            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            collector.log_training_data(symbol, strat, engineered_features, won, pnl, target_excursion)
            symbol_rows_collected += 1
            total_rows_collected += 1

    logger.info("✅ SMC Backfill Complete. New dataset generated.")
