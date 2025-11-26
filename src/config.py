import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# APP INFO
# ---------------------------------------------------------
APP_NAME = "NEXUBOT"
VERSION = "v1.2.0"

# ---------------------------------------------------------
# DATABASE (NEON / POSTGRES)
# ---------------------------------------------------------
# Ensure your .env has DATABASE_URL=postgresql://...
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------
# API & NETWORK
# ---------------------------------------------------------
BINANCE_API_URL = "https://api.binance.com/api/v3"

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------
USE_DYNAMIC_SYMBOLS = False # Auto-find best pairs?
TOP_SYMBOLS_COUNT = 15 # Number of pairs to scan

# Fallback list if dynamic scan fails or is disabled
STATIC_SYMBOLS: list[str] = [
  'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
  'PEPEUSDT', 'SHIBUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT'
]

# Assets requiring special risk handling (Volatility protection)
HIGH_RISK_SYMBOLS: list[str] = ['PEPEUSDT', 'SHIBUSDT', 'DOGEUSDT', 'BONKUSDT', 'WIFUSDT']

# ---------------------------------------------------------
# SIGNAL FREQUENCY (THROTTLING)
# ---------------------------------------------------------
TIMEFRAME = '5m' # Trading timeframe
CANDLE_LIMIT = 200 # Data points needed for EMA200
SCAN_INTERVAL = 60 # Seconds between scan loops

# Spam Protection
GLOBAL_SIGNAL_COOLDOWN = 300 # 5 Mins: Wait time between ANY signal
PAIR_SIGNAL_COOLDOWN = 600 # 10 Mins: Wait time for SAME pair
MAX_SIGNALS_PER_SCAN = 2 # Max signals to show at once (Prevent overwhelm)

# ---------------------------------------------------------
# RISK MANAGEMENT (ZAR CENTRIC)
# ---------------------------------------------------------
DEFAULT_BALANCE_ZAR = 200.0 # Base account size for testing
USD_ZAR_RATE = 18.50 # Exchange rate for internal conversions
RISK_PER_TRADE = 3.0 # Percentage of account to risk per trade
LEVERAGE = 500 # Broker leverage (HFM)

# Broker Spread Simulation
# 0.10% spread roughly mimics the gap between Bid/Ask on Standard Accounts
SPREAD_COST_PCT = 0.10

# HFM Minimum Lot Sizes
MIN_LOT_SIZE = 0.01

# ---------------------------------------------------------
# AI & THRESHOLDS
# ---------------------------------------------------------
MIN_CONFIDENCE = 80.0 # Minimum AI Score to trigger trade
MIN_VOLUME_USDT = 1000000.0 # Minimum 24h Liquidity ($1M USD)
MODEL_FILE = "nexubot_models.pkl"