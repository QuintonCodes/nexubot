import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.1.0"

# ---------------------------------------------------------
# MT5 TERMINAL SETTINGS
# ---------------------------------------------------------
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\Metatrader 5\terminal64.exe")

#  ---------------------------------------------------------
# DATABASE & FALLBACKS
#  ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

FALLBACK_CRYPTO: list[str] = ["BTCUSDm", "ETHUSDm"]
FALLBACK_FOREX: list[str] = ["EURUSDm"]
FALLBACK_METALS: list[str] = ["XAUUSDm", "XAGUSDm"]
FALLBACK_INDICES: list[str] = ["US30m"]

HIGH_VOLATILITY_IDENTIFIERS = ["XAU", "XAG", "BTC", "ETH", "US30"]

#  ---------------------------------------------------------
# SESSION & TIME FILTERS (SAST - South African Standard Time)
#  ---------------------------------------------------------
SESSION_CONFIG = {
    "ASIAN_START": 2,  # 02:00 SAST (JPY, AUD, NZD Volatility)
    "ASIAN_END": 11,  # 11:00 SAST
    "LONDON_START": 9,  # 09:00 SAST (EUR, GBP Volatility)
    "LONDON_END": 18,  # 18:00 SAST
    "NY_START": 14,  # 14:00 SAST (USD, XAU, US30, NAS Volatility)
    "NY_END": 23,  # 23:00 SAST
}

# ---------------------------------------------------------
# STRATEGY & RISK SETTINGS
# ---------------------------------------------------------
TIMEFRAME = "M1"
CANDLE_LIMIT = 500
DEFAULT_MIN_CONFIDENCE = 65.0
DEFAULT_RISK_PCT = 2.0
DEFAULT_MAX_LOT = 5.0
MIN_RR = 2.0

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
EXIT_SCALER_FILE = "nexubot_exit_scaler.pkl"
CALIBRATOR_FILE = "nexubot_calibrator.pkl"
TRAINING_FILE = "training_data.csv"

FEATURE_COLS = [
    "is_htf_aligned",
    "is_liquidity_swept",
    "is_in_fvg",
    "is_in_ifvg",
    "is_in_orderblock",
    "structural_break",
    "active_killzone",
    "distance_to_poi",
    "pd_array_status",
    "mitigation_count",
    "sweep_depth_atr",
]

MAX_ROWS = 14000


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
