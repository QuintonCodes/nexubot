import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.3.1"

# ---------------------------------------------------------
# DATABASE & API
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
BINANCE_API_URL = "https://api.binance.com/api/v3"

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------
TOP_SYMBOLS_COUNT = 15
# Fallback list if dynamic scan fails or is disabled
STATIC_SYMBOLS: list[str] = [
  # --- HFM Crypto Majors ---
  'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'TRXUSDT', 'LTCUSDT',

  # --- HFM Gold (Proxy) ---
  'PAXGUSDT',
]

# Assets requiring special risk handling (Volatility protection)
HIGH_RISK_SYMBOLS: list[str] = ['LTCUSDT', 'XRPUSDT', 'TRXUSDT']

# ---------------------------------------------------------
# CONTRACT SIZES (CRITICAL FOR ACCURATE RISK)
# ---------------------------------------------------------
CONTRACT_SIZES = {
    # Gold
    'PAXGUSDT': 100,

    # Crypto
    'BTCUSDT': 1,
    'ETHUSDT': 1,
    'BNBUSDT': 1,
    'LTCUSDT': 1,

    # "Cheap" coins
    'XRPUSDT': 1000,
    'TRXUSDT': 1000,

    'DEFAULT': 1
}

# ---------------------------------------------------------
# SIGNAL FREQUENCY (THROTTLING)
# ---------------------------------------------------------
TIMEFRAME = '15m'
CANDLE_LIMIT = 200
SCAN_INTERVAL = 60 # Seconds
GLOBAL_SIGNAL_COOLDOWN = 60 # 1 Min
PAIR_SIGNAL_COOLDOWN = 600 # 10 Mins
LOSS_COOLDOWN_DURATION = 1800 # 30 Mins
MAX_SIGNALS_PER_SCAN = 2

# ---------------------------------------------------------
# RISK MANAGEMENT (ZAR CENTRIC)
# ---------------------------------------------------------
DEFAULT_BALANCE_ZAR = 1000000.0
USD_ZAR_RATE = 18.20

RISK_PER_TRADE = 0.5
LEVERAGE = 1000

# HFM Minimum Lot Sizes
LOT_MIN = 0.01
LOT_MAX = 1.00

# Spread Simulation
SPREAD_COST_PCT = 0.60

# ---------------------------------------------------------
# AI CONFIG
# ---------------------------------------------------------
MIN_CONFIDENCE = 70.0
ADX_TREND_THRESHOLD = 25
BB_SQUEEZE_THRESHOLD = 0.10
VWAP_DEV_THRESHOLD = 2.0
MODEL_FILE = "nexubot_models.pkl"