"""
Market Data Normalizer.
Converts raw REST/WebSocket payloads from Twelve Data into the standardized
pandas DataFrame schema required by the Nexubot SMC strategy engine.
"""

import pandas as pd


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a Twelve Data API response DataFrame to a standard schema.
    Injects a default volume of 0.0 for Forex/Metals pairs that omit it.
    """
    # Work on a copy to prevent SettingWithCopyWarnings
    df_norm = df.copy()

    # 0. Gracefully handle missing volume for Forex/Metals
    if "volume" not in df_norm.columns:
        df_norm["volume"] = 0.0

    required_cols = ["datetime", "open", "high", "low", "close", "volume"]

    # 1. Validate incoming schema
    missing = [col for col in required_cols if col not in df_norm.columns]
    if missing:
        raise ValueError(
            f"Normalization Failed: Missing required columns {missing}. " f"Received columns: {list(df_norm.columns)}"
        )

    # 2. Type cast price and volume data to float64
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce").astype("float64")

    # Drop any row where critical numeric conversion failed (e.g., malformed WS ticks)
    df_norm = df_norm.dropna(subset=numeric_cols)

    # 3. Timezone handling: convert Twelve Data's string to UTC datetime64[ns, UTC]
    df_norm["timestamp"] = pd.to_datetime(df_norm["datetime"], utc=True)

    # 4. Standardize column names and order
    df_norm = df_norm[["timestamp", "open", "high", "low", "close", "volume"]]

    # 5. Sort chronologically (oldest to newest)
    df_norm = df_norm.sort_values("timestamp", ascending=True)

    # 6. Deduplicate (keep the latest update if multiple ticks share the same timestamp)
    df_norm = df_norm.drop_duplicates(subset=["timestamp"], keep="last")

    # Reset index for clean sequential row access by the strategy engine
    df_norm = df_norm.reset_index(drop=True)

    return df_norm
