import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.0.0"

# ---------------------------------------------------------
# MT5 TERMINAL SETTINGS
# ---------------------------------------------------------
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\Metatrader 5\terminal64.exe")

# ---------------------------------------------------------
# DATABASE & FALLBACKS
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
FALLBACK_CRYPTO: list[str] = [
    "BTCUSDm",
    "ETHUSDm",
    "BTCJPYm",
]
FALLBACK_FOREX: list[str] = ["EURUSDm", "GBPUSDm", "USDJPYm", "USDCADm" "AUDUSDm"]
FALLBACK_METALS: list[str] = ["XAUUSDm", "XAGUSDm"]
FALLBACK_INDICES = ["US30m", "NAS100m", "GER30m", "UK100m"]
HIGH_VOLATILITY_IDENTIFIERS = ["XAU", "XAG", "BTC", "ETH", "NAS", "US30", "GER30", "JPY"]

# ---------------------------------------------------------
# SESSION & TIME FILTERS (SAST)
# ---------------------------------------------------------
SESSION_CONFIG = {
    "ASIAN_START": 0,
    "ASIAN_END": 7,
    "PRE_LONDON_START": 7,
    "PRE_LONDON_END": 9,
    "LONDON_START": 9,
    "LONDON_END": 12,
    "NY_START": 15,
    "NY_END": 19,
}

# ---------------------------------------------------------
# STRATEGY & RISK SETTINGS
# ---------------------------------------------------------
TIMEFRAME = "M15"
CANDLE_LIMIT = 500
DEFAULT_MIN_CONFIDENCE = 70.0
DEFAULT_RISK_PCT = 2.0
DEFAULT_MAX_LOT = 0.1

SCAN_INTERVAL_CRYPTO = 30
SCAN_INTERVAL_FOREX = 30
SCAN_INTERVAL_INDICES = 30
SCAN_INTERVAL_METALS = 20
GLOBAL_SIGNAL_COOLDOWN = 60
PAIR_SIGNAL_COOLDOWN = 900
LOSS_COOLDOWN_DURATION = 1800
MAX_SIGNALS_PER_SCAN = 3

# ---------------------------------------------------------
# NEURAL NETWORK
# ---------------------------------------------------------
ENTRY_MODEL_FILE = "nexubot_entry.keras"
EXIT_MODEL_FILE = "nexubot_exit.keras"
SCALER_FILE = "nexubot_scaler.pkl"

FEATURE_COLS = [
    "dist_to_vwap",
    "mtf_trend_alignment",
    "hour_norm",
    "volatility_ratio",
    "dist_to_nearest_fvg",
    "is_in_breaker",
    "htf_adx_strength",
    "poi_status",
]


# ---------------------------------------------------------
# DYNAMIC RISK UTILS
# ---------------------------------------------------------
def get_account_risk_caps(balance: float, currency: str = "USD") -> float:
    """
    Returns the maximum allowable risk percentage based on account size and currency.
    """
    if currency == "ZAR":
        if balance < 2000:
            return 5.0
        elif balance < 10000:
            return 4.0
        elif balance < 100000:
            return 3.0
        else:
            return 2.0
    else:  # USD logic
        if balance < 100:
            return 5.0
        elif balance < 500:
            return 4.0
        elif balance < 5000:
            return 3.0
        else:
            return 2.0
