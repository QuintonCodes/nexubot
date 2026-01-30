import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.5.0"

# ---------------------------------------------------------
# MT5 TERMINAL SETTINGS
# ---------------------------------------------------------
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\Metatrader 5\terminal64.exe")

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------
# MARKET SELECTION (Used if MT5 Market Watch is empty)
# ---------------------------------------------------------
FALLBACK_CRYPTO: list[str] = [
    "BTCUSDm",
    "ETHUSDm",
    "BNBUSDm",
    "XRPUSDm",
    "SOLUSDm",
]
FALLBACK_FOREX: list[str] = ["GBPJPYm", "USDJPYm", "EURUSDm", "AUDUSDm", "XAUUSDm"]

# High Volatility assets (Gold, Indices, Volatile Crypto)
# Used to gate trades based on the "High Volatility" toggle
HIGH_VOLATILITY_IDENTIFIERS = ["XAU", "XAG", "BTC", "ETH", "NAS", "US30", "GER30", "JPY"]

# ---------------------------------------------------------
# SESSION & TIME FILTERS (SAST)
# ---------------------------------------------------------
SESSION_CONFIG = {
    "ASIAN_START": 0,  # 00:00
    "ASIAN_END": 7,  # 07:00 (Mean Reversion Only)
    "PRE_LONDON_START": 7,
    "PRE_LONDON_END": 9,  # 07:00-09:00 (No Trading / Trap Zone)
    "LONDON_START": 9,  # 09:00 (Trend Start)
    "LONDON_END": 12,  # 12:00
    "NY_START": 15,  # 15:00
    "NY_END": 19,  # 19:00
    "NO_VOLATILITY_HOUR": 13,  # 13:00 SAST (Block Volatility Breakouts)
}

# ---------------------------------------------------------
# STRATEGY SETTINGS
# ---------------------------------------------------------
TIMEFRAME = "M15"
CANDLE_LIMIT = 500
DEFAULT_MIN_CONFIDENCE = 75.0
CHOP_THRESHOLD_TREND = 38.2
CHOP_THRESHOLD_RANGE = 61.8

# ---------------------------------------------------------
# RISK MANAGEMENT (ZAR ACCOUNT)
# ---------------------------------------------------------
DEFAULT_BALANCE_ZAR = 500.0
DEFAULT_RISK_PCT = 2.0
DEFAULT_MAX_LOT = 0.1

# Scanner Timing
SCAN_INTERVAL_CRYPTO = 30
SCAN_INTERVAL_FOREX = 30
GLOBAL_SIGNAL_COOLDOWN = 60  # 1 Min between any signal
PAIR_SIGNAL_COOLDOWN = 900  # 15 Mins per specific pair
LOSS_COOLDOWN_DURATION = 1800  # 30 Mins cooldown after a loss
MAX_SIGNALS_PER_SCAN = 2

# ---------------------------------------------------------
# NEURAL NETWORK
# ---------------------------------------------------------

ENTRY_MODEL_FILE = "nexubot_entry.keras"
EXIT_MODEL_FILE = "nexubot_exit.keras"
SCALER_FILE = "nexubot_scaler.pkl"
DATA_FILE = "training_data.csv"

FEATURE_COLS = [
    "rsi",
    "adx",
    "atr",
    "atr_ratio",
    "avg_atr_24",
    "ema_dist",
    "bb_width",
    "vol_ratio",
    "htf_trend",
    "dist_to_pivot",
    "hour_norm",
    "day_norm",
    "wick_ratio",
    "dist_ema200",
    "volatility_ratio",
    "dist_to_vwap",
    "rolling_acc",
    "recent_range_std",
]


# ---------------------------------------------------------
# DYNAMIC RISK UTILS
# ---------------------------------------------------------
def get_account_risk_caps(balance: float) -> float:
    """
    Returns the maximum allowable risk percentage based on account size.
    Smaller accounts get more breathing room for high-probability setups.
    Larger accounts get tighter safety caps.
    """
    if balance < 2000:
        return 5.0
    elif balance < 10000:
        return 4.0
    elif balance < 100000:
        return 3.0
    else:
        return 2.0
