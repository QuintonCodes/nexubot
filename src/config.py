import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.4"

# ---------------------------------------------------------
# API KEYS & URLS
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
BINANCE_API_URL = "https://api.binance.com/api/v3"

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------
# fast scan
CRYPTO_SYMBOLS: list[str] = [
  'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'TRXUSDT',
]

# slow scan (credit conservation)
FOREX_SYMBOLS: list[str] = [
    'GBP/JPY', 'USD/JPY', 'EUR/USD', 'AUD/USD', 'XAU/USD'
]

# Combined for internal use
ALL_SYMBOLS = CRYPTO_SYMBOLS + FOREX_SYMBOLS
HIGH_RISK_SYMBOLS: list[str] = ['XRPUSDT', 'TRXUSDT', 'XAU/USD', 'GBP/JPY']

# ---------------------------------------------------------
# BROKER SPECIFICATIONS (HFM / MT5)
# ---------------------------------------------------------
BROKER_SPECS = {
    # --- CRYPTO ---
    'BTCUSDT': {
        'contract_size': 1, 'digits': 3, 'min_vol': 0.01, 'max_vol': 20.0,
        'stop_level': 165000, 'commission_per_lot': 0.50
    },
    'ETHUSDT': {
        'contract_size': 1, 'digits': 3, 'min_vol': 1.0, 'max_vol': 20.0,
        'stop_level': 10000, 'commission_per_lot': 0.50
    },
    'BNBUSDT': {
        'contract_size': 1, 'digits': 3, 'min_vol': 1.0, 'max_vol': 20.0,
        'stop_level': 500, 'commission_per_lot': 0.50
    },
    'XRPUSDT': {
        'contract_size': 1000, 'digits': 5, 'min_vol': 1.0, 'max_vol': 20.0,
        'stop_level': 4000, 'commission_per_lot': 0.50
    },
    'TRXUSDT': {
        'contract_size': 1000, 'digits': 5, 'min_vol': 1.0, 'max_vol': 20.0,
        'stop_level': 900, 'commission_per_lot': 0.50
    },
    # --- FOREX ---
    'GBP/JPY': {
        'contract_size': 100000, 'digits': 3, 'min_vol': 0.01, 'max_vol': 60.0,
        'stop_level': 0, 'commission_per_lot': 0.0
    },
    'USD/JPY': {
        'contract_size': 100000, 'digits': 3, 'min_vol': 0.01, 'max_vol': 60.0,
        'stop_level': 0, 'commission_per_lot': 0.0
    },
    'EUR/USD': {
        'contract_size': 100000, 'digits': 5, 'min_vol': 0.01, 'max_vol': 60.0,
        'stop_level': 0, 'commission_per_lot': 0.0
    },
    'AUD/USD': {
        'contract_size': 100000, 'digits': 5, 'min_vol': 0.01, 'max_vol': 60.0,
        'stop_level': 40, 'commission_per_lot': 0.0
    },
    'XAU/USD': {
        'contract_size': 100, 'digits': 2, 'min_vol': 0.01, 'max_vol': 60.0,
        'stop_level': 0, 'commission_per_lot': 0.0
    }
}

# ---------------------------------------------------------
# SPREAD ESTIMATES (Points)
# ---------------------------------------------------------
SPREAD_POINTS = {
    'BTCUSDT': 3000,
    'ETHUSDT': 2550,
    'XRPUSDT': 1500,
    'TRXUSDT': 50,
    'GBP/JPY': 28,
    'EUR/USD': 12,
    'XAU/USD': 35,
    'DEFAULT': 20
}

# ---------------------------------------------------------
# STRATEGY SETTINGS
# ---------------------------------------------------------
TIMEFRAME = '15m'
CANDLE_LIMIT = 200
MIN_CONFIDENCE = 70.0
MODEL_FILE = "nexubot_models.pkl"

# ---------------------------------------------------------
# TIMING (Seoonds)
# ---------------------------------------------------------
SCAN_INTERVAL_CRYPTO = 60
SCAN_INTERVAL_FOREX = 300 # 5 Mins to save API credits
GLOBAL_SIGNAL_COOLDOWN = 60 # 1 Min between any signal
PAIR_SIGNAL_COOLDOWN = 900 # 15 Mins per specific pair
LOSS_COOLDOWN_DURATION = 1800 # 30 Mins cooldown after a loss
MAX_SIGNALS_PER_SCAN = 2

# ---------------------------------------------------------
# RISK MANAGEMENT (ZAR)
# ---------------------------------------------------------
DEFAULT_BALANCE_ZAR = 450.0
FALLBACK_USD_ZAR = 18.00

RISK_PER_TRADE_PCT = 2.0
SMALL_ACCOUNT_THRESHOLD_ZAR = 5000.0
MAX_SURVIVAL_CAP_PCT = 15.0

# LIMITS
LEVERAGE = 1000