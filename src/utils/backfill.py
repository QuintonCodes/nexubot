import asyncio
import numpy as np
import os
import pandas as pd
import sys
from typing import Dict, List, Optional

# Adjust path to root
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.data.provider import DataProvider
from src.analysis.indicators import TechnicalAnalyzer
from src.engine.strategies import StrategyAnalyzer
from src.config import FALLBACK_CRYPTO, DATA_FILE, FALLBACK_FOREX


# Memory buffer for efficient writing
BACKFILL_BUFFER: List[Dict] = []


async def backfill_data(target_symbols: Optional[List[str]] = None):
    """Main backfill routine."""
    print("🚀 Starting Optimized Backfill...")
    provider = DataProvider()

    connected = await provider.initialize()

    all_symbols = []
    if target_symbols:
        all_symbols = target_symbols
        print(f"🎯 Partial Backfill Mode: Targeting {len(all_symbols)} symbols")

    else:
        target_crypto = []
        target_forex = []

        if connected:
            print("✅ Connected to MT5. Fetching User's Market Watch...")
            dynamic_symbols = await provider.get_dynamic_symbols()

            target_crypto = dynamic_symbols.get("crypto", [])
            target_forex = dynamic_symbols.get("forex", [])

        # Fallback Logic
        if not target_crypto:
            target_crypto = FALLBACK_CRYPTO
        if not target_forex:
            target_forex = FALLBACK_FOREX

        all_symbols = target_crypto + target_forex

    print(f"📊 Processing {len(all_symbols)} symbols...")
    analyzer = StrategyAnalyzer()

    # Clear buffer
    BACKFILL_BUFFER.clear()

    # Chunking for concurrency (Batch size 5)
    chunk_size = 5
    for i in range(0, len(all_symbols), chunk_size):
        chunk = all_symbols[i : i + chunk_size]
        await asyncio.gather(*(process_symbol(sym, provider, analyzer) for sym in chunk))

    finalize_dataset()
    await provider.shutdown()
    print("🏁 Backfill Complete.")


def buffer_backfill_data(features: dict, won: bool, pnl: float, excursion: float):
    """Adds data to memory buffer instead of writing immediately."""
    BACKFILL_BUFFER.append(
        {**features, "target_win": 1 if won else 0, "target_pnl": pnl, "target_excursion": excursion}
    )


def finalize_dataset():
    """
    Cleans, deduplicates, and caps the dataset at 5,000 rows.
    """
    print("🧹 Finalizing and cleaning dataset...")

    # Create DF from new buffer
    new_df = pd.DataFrame(BACKFILL_BUFFER)
    existing_df = None
    full_df = None

    if new_df.empty:
        print("⚠️ No new data found during backfill.")
        return

    # Load existing data
    if os.path.exists(DATA_FILE):
        try:
            existing_df = pd.read_csv(DATA_FILE, on_bad_lines="skip")
            full_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Error reading existing CSV, starting fresh: {e}")
            full_df = new_df
    else:
        full_df = new_df

    # 1. Deduplicate
    # Sort by time so we keep the newest version of a specific timestamp/symbol combo if duplicates exist
    if "timestamp" in full_df.columns:
        full_df = full_df.sort_values("timestamp")

    # Drop duplicates based on Symbol + Time (assuming specific feature serves as proxy for time or row uniqueness)
    # Since features don't strictly have a timestamp column in this scope, we deduplicate by identical rows
    full_df.drop_duplicates(inplace=True)

    # 3. Dynamic Capping (Last 50000 rows)
    if len(full_df) > 50000:
        print(f"✂️ Capping dataset: Trimming {len(full_df)} rows down to 50000.")
        full_df = full_df.iloc[-50000:]

    # 4. Overwrite File
    full_df.to_csv(DATA_FILE, index=False, mode="w")
    print(f"💾 Database Updated: {len(full_df)} records saved to {DATA_FILE}")


def log_backfill_data(features: dict, won: bool, pnl: float, excursion: float):
    """Logs the backfill data to CSV."""
    new_row = {**features, "target_win": 1 if won else 0, "target_pnl": pnl, "target_excursion": excursion}
    df = pd.DataFrame([new_row])

    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False, mode="w")
    else:
        df.to_csv(DATA_FILE, index=False, mode="a", header=False)


async def process_symbol(symbol, provider: DataProvider, analyzer: StrategyAnalyzer) -> int:
    """Processes a single symbol for backfilling data with MTF logic."""
    print(f"📥 Processing {symbol}...")

    # 1. Fetch Lower Timeframe Data (15m) (Execution)
    klines_m15 = await provider.fetch_klines(symbol, "15m", 6000)
    if not klines_m15:
        return 0

    df_m15 = pd.DataFrame(klines_m15)
    df_m15 = TechnicalAnalyzer.calculate_indicators(df_m15, heavy=True)

    # 2. Fetch Higher Timeframe (Trend)
    # Use H4 for Forex, H1 for Crypto
    htf_tf = "4h" if symbol in FALLBACK_FOREX else "1h"
    klines_htf = await provider.fetch_klines(symbol, htf_tf, 3000)

    if not klines_htf:
        return 0

    df_htf = pd.DataFrame(klines_htf)
    df_htf = TechnicalAnalyzer.calculate_indicators(df_htf, heavy=False)

    df_htf["htf_ema_200"] = df_htf["ema_200"]
    df_htf["htf_slope"] = df_htf["ema_200_slope"]

    # Vectorized Trend Calculation for Merge
    conditions = [
        (df_htf["close"] > df_htf["htf_ema_200"]) & (df_htf["ema_200_slope"] > 0),
        (df_htf["close"] < df_htf["htf_ema_200"]) & (df_htf["ema_200_slope"] < 0),
    ]
    choices = [1, -1]  # 1=BULL, -1=BEAR
    df_htf["htf_trend_val"] = np.select(conditions, choices, default=0)

    # Keep only time and trend columns to merge
    df_htf_clean = df_htf[["time", "htf_trend_val"]].sort_values("time")

    # Shift HTF timestamps to prevent lookahead
    shift_seconds = 14400 if htf_tf == "4h" else 3600
    df_htf_clean["time"] = df_htf_clean["time"] + shift_seconds

    # Merge HTF Data into M15 Data
    df_m15 = df_m15.sort_values("time")
    df_merged = pd.merge_asof(df_m15, df_htf_clean, on="time", direction="backward")

    if len(df_merged) > 100:
        df_merged = df_merged.iloc[2000:].reset_index(drop=True)
    else:
        return 0

    processed = 0
    df_merged["avg_atr"] = df_merged["atr"].rolling(24).mean()

    for i in range(50, len(df_merged) - 16):
        if processed >= 1000:  # Limit samples per symbol
            break

        curr = df_merged.iloc[i]
        trend_val = curr["htf_trend_val"]
        htf_trend_str = "BULL" if trend_val == 1 else ("BEAR" if trend_val == -1 else "FLAT")

        # 1. Feature Extraction
        pivot_high = df_merged.iloc[i - 20 : i]["high"].max()
        dist_to_pivot = abs(curr["close"] - pivot_high) / curr["close"]
        range_len = curr["high"] - curr["low"]
        wick_ratio = (curr["high"] - curr["close"]) / range_len if range_len > 0 else 0
        avg_atr = curr["avg_atr"] if curr["avg_atr"] > 0 else curr["atr"]
        dist_to_vwap = (curr["close"] - curr["vwap"]) / curr["vwap"] if curr["vwap"] != 0 else 0.0

        curr_time = pd.to_datetime(curr["time"], unit="s")
        symbol_type = provider.get_symbol_type(symbol)
        day_norm_val = 0.0 if symbol_type == "CRYPTO" else curr_time.weekday() / 6.0

        rolling_acc = 0.5  # Default neutral

        # recent_range_std (high-low) over prior 20 candles (safe with i check)
        start_idx = max(0, i - 20)
        recent_body = df_merged.iloc[start_idx:i]
        recent_range_std = recent_body["high"].sub(recent_body["low"]).std() if not recent_body.empty else 0.0

        features = {
            "rsi": curr["rsi"],
            "adx": curr["adx"],
            "atr": curr["atr"],
            "atr_ratio": curr["atr"] / (avg_atr + 1e-9),
            "avg_atr_24": avg_atr,
            "ema_dist": (curr["close"] - curr["ema_50"]) / curr["close"],
            "bb_width": curr["bb_width"],
            "vol_ratio": curr["volume"] / curr["vol_sma"] if curr["vol_sma"] else 1,
            "htf_trend": trend_val,
            "dist_to_pivot": dist_to_pivot,
            "hour_norm": curr_time.hour / 24.0,
            "day_norm": day_norm_val,
            "wick_ratio": wick_ratio,
            "dist_ema200": (curr["close"] - curr["ema_200"]) / curr["close"],
            "volatility_ratio": curr["atr"] / avg_atr,
            "dist_to_vwap": dist_to_vwap,
            "rolling_acc": rolling_acc,
            "recent_range_std": recent_range_std if not pd.isna(recent_range_std) else 0.0,
        }

        # 2. Check Strategy
        signal = None

        if symbol_type == "FOREX" or symbol in FALLBACK_FOREX:
            signal = analyzer.analyze_forex(curr, df_merged.iloc[: i + 1], htf_trend_str, [])
        else:
            signal = analyzer.analyze_crypto(curr, df_merged.iloc[: i + 1], [])

        if signal:
            # 3. Simulate Outcome
            entry = curr["close"]
            if signal.get("order_type") == "LIMIT":
                entry = signal["price"]

            is_long = signal["direction"] == "LONG"
            atr = curr["atr"]

            sl_dist = atr * 1.0
            if "suggested_sl" in signal:
                sl_dist = abs(entry - signal["suggested_sl"])

            tp_dist = max(atr * 2.0, sl_dist * 1.5)

            sl = entry - sl_dist if is_long else entry + sl_dist
            tp = entry + tp_dist if is_long else entry - tp_dist

            # Ensure this matches your Live Bot duration (e.g., 4 hours = 16 candles)
            lookahead_candles = 16
            future = df_merged.iloc[i + 1 : i + 1 + lookahead_candles]

            # Simulate
            won, pnl, excursion = simulate_trade_management(entry, sl, tp, is_long, future, atr)

            buffer_backfill_data(features, won, pnl, excursion)
            processed += 1

    print(f"✅ {symbol}: {processed} signals found.")
    return processed


def simulate_trade_management(
    entry: float, sl: float, tp: float, is_long: bool, future_candles: pd.DataFrame, atr: float
) -> tuple[bool, float, float]:
    """
    Simulates the Live Bot's Trailing Stop Logic (BE -> 1R -> 2R).
    Returns: (won, pnl_r_multiple, max_excursion_atr)
    """
    current_sl = sl
    be_stage = 0  # 0=None, 1=BE, 2=Lock 1R, 3=Lock 2R

    max_favorable_dist = 0.0
    outcome_pnl = 0.0
    won = False

    entry_price = entry

    for _, row in future_candles.iterrows():
        # Approx Tick Data from Candle
        row_high = row["high"]
        row_low = row["low"]

        if is_long:
            # 1. Check Stops/TP
            if row_low <= current_sl:
                outcome_pnl = current_sl - entry_price
                won = False
                break

            if row_high >= tp:
                outcome_pnl = tp - entry_price
                won = True
                break

            # 2. Update Max Excursion
            curr_dist = row_high - entry_price
            if curr_dist > max_favorable_dist:
                max_favorable_dist = curr_dist

            # 3. Trailing Logic (Mirrors console.py)
            # Stage 3: Lock 2R
            if be_stage < 3 and max_favorable_dist > (atr * 3.0):
                new_sl = entry_price + (atr * 2.0)
                if new_sl > current_sl:
                    current_sl = new_sl
                    be_stage = 3

            # Stage 2: Lock 1R
            elif be_stage < 2 and max_favorable_dist > (atr * 2.0):
                new_sl = entry_price + (atr * 1.0)
                if new_sl > current_sl:
                    current_sl = new_sl
                    be_stage = 2

            # Stage 1: Breakeven
            elif be_stage < 1 and max_favorable_dist > (atr * 1.0):
                new_sl = entry_price + (atr * 0.1)  # Small buffer
                if new_sl > current_sl:
                    current_sl = new_sl
                    be_stage = 1

        else:  # Short
            # 1. Check Stops/TP
            if row_high >= current_sl:
                outcome_pnl = entry_price - current_sl
                won = False
                break

            if row_low <= tp:
                outcome_pnl = entry_price - tp
                won = True
                break

            # 2. Update Max Excursion
            curr_dist = entry_price - row_low
            if curr_dist > max_favorable_dist:
                max_favorable_dist = curr_dist

            # 3. Trailing Logic
            if be_stage < 3 and max_favorable_dist > (atr * 3.0):
                new_sl = entry_price - (atr * 2.0)
                if new_sl < current_sl:
                    current_sl = new_sl
                    be_stage = 3
            elif be_stage < 2 and max_favorable_dist > (atr * 2.0):
                new_sl = entry_price - (atr * 1.0)
                if new_sl < current_sl:
                    current_sl = new_sl
                    be_stage = 2
            elif be_stage < 1 and max_favorable_dist > (atr * 1.0):
                new_sl = entry_price - (atr * 0.1)
                if new_sl < current_sl:
                    current_sl = new_sl
                    be_stage = 1

    # Check for timeout (end of candles)
    if outcome_pnl == 0.0:
        # Close at last candle close
        last_close = future_candles.iloc[-1]["close"]
        if is_long:
            outcome_pnl = last_close - entry_price
        else:
            outcome_pnl = entry_price - last_close
        won = outcome_pnl > 0

    return won, outcome_pnl, (max_favorable_dist / atr if atr > 0 else 0)


if __name__ == "__main__":
    asyncio.run(backfill_data())
