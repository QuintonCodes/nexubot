import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.5.0"


# ---------------------------------------------------------
# CONFIG MANAGER
# ---------------------------------------------------------
class ConfigManager:
    @staticmethod
    def get_config_path():
        local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.config"))
        return os.path.join(local_app_data, "Nexubot", "settings.json")

    @classmethod
    def load_settings(cls):
        path = cls.get_config_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @classmethod
    def save_settings(cls, settings):
        path = cls.get_config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(settings, f, indent=4)


# ---------------------------------------------------------
# MT5 TERMINAL SETTINGS
# ---------------------------------------------------------
_local_settings = ConfigManager.load_settings()
MT5_LOGIN = int(_local_settings.get("MT5_LOGIN", os.getenv("MT5_LOGIN", "0")))
MT5_PASSWORD = _local_settings.get("MT5_PASSWORD", os.getenv("MT5_PASSWORD", ""))
MT5_SERVER = _local_settings.get("MT5_SERVER", os.getenv("MT5_SERVER", ""))
MT5_PATH = _local_settings.get("MT5_PATH", os.getenv("MT5_PATH", r"C:\Program Files\Metatrader 5\terminal64.exe"))

# ---------------------------------------------------------
# DATABASE & FALLBACKS
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
FALLBACK_CRYPTO: list[str] = [
    "BTCUSDm",
    "ETHUSDm",
    "BNBUSDm",
    "XRPUSDm",
    "SOLUSDm",
]
FALLBACK_FOREX: list[str] = ["GBPJPYm", "USDJPYm", "EURUSDm", "AUDUSDm", "XAUUSDm"]
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
    "NO_VOLATILITY_HOUR": 13,
}

# ---------------------------------------------------------
# STRATEGY & RISK SETTINGS
# ---------------------------------------------------------
TIMEFRAME = "M15"
CANDLE_LIMIT = 500
DEFAULT_MIN_CONFIDENCE = 75.0
DEFAULT_BALANCE_ZAR = 500.0
DEFAULT_RISK_PCT = 2.0
DEFAULT_MAX_LOT = 0.1

SCAN_INTERVAL_CRYPTO = 30
SCAN_INTERVAL_FOREX = 30
GLOBAL_SIGNAL_COOLDOWN = 60
PAIR_SIGNAL_COOLDOWN = 900
LOSS_COOLDOWN_DURATION = 1800
MAX_SIGNALS_PER_SCAN = 2

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
