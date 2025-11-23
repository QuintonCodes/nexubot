from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
BINANCE_API_URL = "https://api.binance.com/api/v3"

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------
USE_DYNAMIC_SYMBOLS = True
TOP_SYMBOLS_COUNT = 15
STATIC_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

# ---------------------------------------------------------
# TRADING TIMEFRAME & LOGIC
# ---------------------------------------------------------
TIMEFRAME = '5m'
CANDLE_LIMIT = 200
SCAN_INTERVAL = 60

# ---------------------------------------------------------
# RISK MANAGEMENT (MT5 / CFD)
# ---------------------------------------------------------
ACCOUNT_BALANCE = 1000.0       # Your Trading Balance ($)
RISK_PER_TRADE = 2.0           # Risk percentage per trade (2% is standard)
LEVERAGE = 10                  # Leverage used in MT5
STD_LOT_SIZE = 100000          # Standard Lot unit (Forex=100k, Crypto varies, usually 1 coin or contract)

# ---------------------------------------------------------
# AI & THRESHOLDS
# ---------------------------------------------------------
MIN_CONFIDENCE = 80            # Minimum score to trigger
MIN_VOLUME_USDT = 5000000      # Higher filter to ensure valid technicals
DB_NAME = "nexubot_data.db"
MODEL_FILE = "nexubot_models.pkl"