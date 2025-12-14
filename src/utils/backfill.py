import asyncio
import sys
import os
import pandas as pd

# Adjust path to root
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.data.provider import DataProvider
from src.analysis.indicators import TechnicalAnalyzer
from src.engine.strategies import StrategyAnalyzer
from src.config import ALL_SYMBOLS, DATA_FILE


def log_backfill_data(features: dict, won: bool, pnl: float, excursion: float):
    """Logs the backfill data to CSV."""
    df = pd.DataFrame([{**features, "target_win": 1 if won else 0, "target_pnl": pnl, "target_excursion": excursion}])
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False, mode="w")
    else:
        df.to_csv(DATA_FILE, index=False, mode="a", header=False)


async def process_symbol(symbol, provider: DataProvider, analyzer: StrategyAnalyzer) -> int:
    """Processes a single symbol for backfilling data."""
    print(f"📥 Processing {symbol}...")
    klines = await provider.fetch_klines(symbol, "15m", 5000)

    if not klines:
        return 0

    df = pd.DataFrame(klines)
    df = TechnicalAnalyzer.calculate_indicators(df, heavy=True)

    processed = 0
    # Calculate Avg ATR for Volatility Ratio
    df["avg_atr"] = df["atr"].rolling(24).mean()

    for i in range(200, len(df) - 20):
        curr = df.iloc[i]

        # 1. Feature Extraction (Replicating ai_engine)
        # Pivot Calc
        window_slice = df.iloc[i - 20 : i]
        pivot_high = window_slice["high"].max()
        dist_to_pivot = abs(curr["close"] - pivot_high) / curr["close"]
        range_len = curr["high"] - curr["low"]
        wick_ratio = (curr["high"] - curr["close"]) / range_len if range_len > 0 else 0
        avg_atr = curr["avg_atr"] if curr["avg_atr"] > 0 else curr["atr"]

        features = {
            "rsi": curr["rsi"],
            "adx": curr["adx"],
            "atr": curr["atr"],
            "ema_dist": (curr["close"] - curr["ema_50"]) / curr["close"],
            "bb_width": curr["bb_width"],
            "vol_ratio": curr["volume"] / curr["vol_sma"] if curr["vol_sma"] else 1,
            "htf_trend": 0,
            "dist_to_pivot": dist_to_pivot,
            "hour_norm": 0.5,
            "day_norm": 0.5,
            "wick_ratio": wick_ratio,
            "dist_ema200": (curr["close"] - curr["ema_200"]) / curr["close"],
            "volatility_ratio": curr["atr"] / avg_atr,
        }

        # 2. Check Strategy
        # We assume FLAT htf_trend for backfill simplicity, or calculate it
        signal = None

        # Quick check to speed up (Strategies require pattern lists, passing empty)
        if "USD" in symbol or "JPY" in symbol:
            signal = analyzer.analyze_forex(curr, df.iloc[: i + 1], "FLAT", [])
        else:
            signal = analyzer.analyze_crypto(curr, df.iloc[: i + 1], [])

        if signal:
            # 3. Simulate Outcome (Max Excursion)
            entry = curr["close"]
            is_long = signal["direction"] == "LONG"
            atr = curr["atr"]
            sl_dist = atr * 1.5

            # Look forward 40 candles
            future = df.iloc[i + 1 : i + 41]
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

            log_backfill_data(features, won, 0.0, max_atr_gain)
            processed += 1

    print(f"✅ {symbol}: {processed} signals found.")
    return processed


async def backfill_data():
    """Main backfill routine."""
    print("🚀 Starting Optimized Backfill...")
    provider = DataProvider()
    if not await provider.initialize():
        return
    analyzer = StrategyAnalyzer()

    # Chunking for concurrency (Batch size 5)
    chunk_size = 5
    for i in range(0, len(ALL_SYMBOLS), chunk_size):
        chunk = ALL_SYMBOLS[i : i + chunk_size]
        await asyncio.gather(*(process_symbol(sym, provider, analyzer) for sym in chunk))

    await provider.shutdown()
    print("🏁 Backfill Complete.")


if __name__ == "__main__":
    asyncio.run(backfill_data())
