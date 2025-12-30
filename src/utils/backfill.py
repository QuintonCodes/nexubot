import asyncio
import sys
import os
import pandas as pd
from typing import Dict, List

# Adjust path to root
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.data.provider import DataProvider
from src.analysis.indicators import TechnicalAnalyzer
from src.engine.strategies import StrategyAnalyzer
from src.config import ALL_SYMBOLS, CRYPTO_SYMBOLS, DATA_FILE, FOREX_SYMBOLS


# Memory buffer for efficient writing
BACKFILL_BUFFER: List[Dict] = []


async def backfill_data():
    """Main backfill routine."""
    print("🚀 Starting Optimized Backfill...")
    provider = DataProvider()
    if not await provider.initialize():
        return
    analyzer = StrategyAnalyzer()

    # Clear buffer
    BACKFILL_BUFFER.clear()

    # Chunking for concurrency (Batch size 5)
    chunk_size = 5
    for i in range(0, len(ALL_SYMBOLS), chunk_size):
        chunk = ALL_SYMBOLS[i : i + chunk_size]
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

    # Load existing data if any
    existing_df = pd.DataFrame()
    if os.path.exists(DATA_FILE):
        try:
            existing_df = pd.read_csv(DATA_FILE, on_bad_lines="skip")
        except Exception:
            pass

    # Create DF from new buffer
    new_df = pd.DataFrame(BACKFILL_BUFFER)

    if new_df.empty:
        print("⚠️ No new data buffered.")
        return

    # Merge
    full_df = pd.concat([existing_df, new_df], ignore_index=True)

    if full_df.empty:
        return

    # 1. Remove Duplicates
    # Assuming columns match, we drop exact duplicates
    full_df.drop_duplicates(inplace=True)

    # 3. Dynamic Capping (Last 50000 rows)
    if len(full_df) > 50000:
        print(f"✂️ Capping dataset: Trimming {len(full_df)} rows down to 50000.")
        full_df = full_df.iloc[-50000:]

    # 4. Overwrite File
    full_df.to_csv(DATA_FILE, index=False, mode="w")
    print(f"💾 Saved {len(full_df)} rows to {DATA_FILE}")


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
    htf_tf = "4h" if symbol in FOREX_SYMBOLS else "1h"
    klines_htf = await provider.fetch_klines(symbol, htf_tf, 3000)

    if not klines_htf:
        return 0

    df_htf = pd.DataFrame(klines_htf)
    # Calculate simple EMA trend for HTF
    df_htf["htf_ema_200"] = df_htf["close"].ewm(span=200).mean()
    df_htf["htf_trend_val"] = df_htf.apply(
        lambda x: 1 if x["close"] > x["htf_ema_200"] else (-1 if x["close"] < x["htf_ema_200"] else 0), axis=1
    )

    # Keep only time and trend columns to merge
    df_htf = df_htf[["time", "htf_trend_val"]].sort_values("time")

    # 3. Merge HTF Data into M15 Data
    # 'merge_asof' finds the last known H4 candle for each M15 candle
    df_m15 = df_m15.sort_values("time")
    df_merged = pd.merge_asof(df_m15, df_htf, on="time", direction="backward")  # Look for the closest PAST H4 candle

    if len(df_merged) > 500:
        df_merged = df_merged.iloc[500:].reset_index(drop=True)
    else:
        return 0

    processed = 0
    df_merged["avg_atr"] = df_merged["atr"].rolling(24).mean()

    for i in range(50, len(df_merged) - 20):
        if processed >= 1000:  # Limit samples per symbol
            break

        curr = df_merged.iloc[i]

        # Determine HTF Trend string for Strategy Analyzer
        trend_val = curr["htf_trend_val"]
        htf_trend_str = "BULL" if trend_val == 1 else ("BEAR" if trend_val == -1 else "FLAT")

        # 1. Feature Extraction
        pivot_high = df_merged.iloc[i - 20 : i]["high"].max()
        dist_to_pivot = abs(curr["close"] - pivot_high) / curr["close"]
        range_len = curr["high"] - curr["low"]
        wick_ratio = (curr["high"] - curr["close"]) / range_len if range_len > 0 else 0
        avg_atr = curr["avg_atr"] if curr["avg_atr"] > 0 else curr["atr"]

        dist_to_vwap = (curr["close"] - curr["vwap"]) / curr["vwap"] if curr["vwap"] != 0 else 0.0

        # --- Time Normalization Fix ---
        curr_time = pd.to_datetime(curr["time"], unit="s")
        # Crypto runs 24/7 so day of week is noise (set to 0.0), Forex respects weekday
        day_norm_val = 0.0 if symbol in CRYPTO_SYMBOLS else curr_time.weekday() / 6.0

        features = {
            "rsi": curr["rsi"],
            "adx": curr["adx"],
            "atr": curr["atr"],
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
        }

        # 2. Check Strategy
        signal = None

        if symbol in FOREX_SYMBOLS:
            signal = analyzer.analyze_forex(curr, df_merged.iloc[: i + 1], htf_trend_str, [])
        else:
            signal = analyzer.analyze_crypto(curr, df_merged.iloc[: i + 1], [])

        if signal:
            # 3. Simulate Outcome (Max Excursion)
            entry = curr["close"]
            is_long = signal["direction"] == "LONG"
            atr = curr["atr"]
            sl_dist = atr * 1.5

            # Ensure this matches your Live Bot duration (e.g., 4 hours = 16 candles)
            lookahead_candles = 16
            future = df_merged.iloc[i + 1 : i + 1 + lookahead_candles]

            won = False
            max_atr_gain = 0.0

            for _, row in future.iterrows():
                if is_long:
                    if row["low"] < (entry - sl_dist):  # Hit SL
                        break
                    gain = row["high"] - entry
                    max_atr_gain = max(max_atr_gain, gain / atr)
                    if gain > (atr * 1.5):
                        won = True
                else:
                    if row["high"] > (entry + sl_dist):  # Hit SL
                        break
                    gain = entry - row["low"]
                    max_atr_gain = max(max_atr_gain, gain / atr)
                    if gain > (atr * 1.5):
                        won = True

            buffer_backfill_data(features, won, 0.0, max_atr_gain)
            processed += 1

    print(f"✅ {symbol}: {processed} signals found.")
    return processed


if __name__ == "__main__":
    asyncio.run(backfill_data())
