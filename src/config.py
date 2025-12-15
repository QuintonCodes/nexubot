import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.2.1"

# ---------------------------------------------------------
# MT5 TERMINAL SETTINGS
# ---------------------------------------------------------
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\HFM Metatrader 5\terminal64.exe")

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------
CRYPTO_SYMBOLS: list[str] = [
    "#BTCUSD",
    "#ETHUSD",
    "#BNBUSD",
    "#XRPUSD",
    "#TRXUSD",
]

# slow scan (credit conservation)
FOREX_SYMBOLS: list[str] = ["GBPJPY", "USDJPY", "EURUSD", "AUDUSD", "XAUUSD"]

# Combined for internal use
ALL_SYMBOLS = CRYPTO_SYMBOLS + FOREX_SYMBOLS
HIGH_RISK_SYMBOLS: list[str] = ["XRPUSD", "TRXUSD", "XAU/USD", "GBPJPY"]

# ---------------------------------------------------------
# TRADING SESSIONS (South Africa / GMT +2)
# ---------------------------------------------------------
SESSION_CONFIG = {
    "FOREX_START": 9,  # 09:00 SAST (London Open is 10:00 SAST in winter, 09:00 summer)
    "FOREX_END": 22,  # 22:00 SAST (NY Close area)
    "CRYPTO": "24/7",
}

# ---------------------------------------------------------
# STRATEGY SETTINGS
# ---------------------------------------------------------
TIMEFRAME = "M15"
CANDLE_LIMIT = 500
MIN_CONFIDENCE = 70.0

# ---------------------------------------------------------
# RISK MANAGEMENT (ZAR ACCOUNT)
# ---------------------------------------------------------
DEFAULT_BALANCE_ZAR = 450.0
RISK_PER_TRADE_PCT = 2.0
MAX_LOT_SIZE = 1

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
LEGACY_ENTRY = "nexubot_entry.h5"
LEGACY_EXIT = "nexubot_exit.h5"

FEATURE_COLS = [
    "rsi",
    "adx",
    "atr",
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
]
