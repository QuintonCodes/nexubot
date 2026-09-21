# 🚀 Nexubot — Cloud-Native SMC Trading Signal Engine

**Nexubot** is a cloud-native, asynchronous Telegram signal trading bot built specifically for **XAU/USD (Gold)** on the **M5 timeframe**. It implements algorithmic Smart Money Concepts (SMC)—including Market Structure Shifts (BOS, CHoCH, MSS, CISD), Order Blocks, Breaker Blocks, Fair Value Gaps (FVG), ICT Optimal Trade Entries (OTE), Premium/Discount arrays, Inducement (IDM), and Session Killzones—operating 24/7 in a headless Linux container environment.

## 1. Architectural Highlights

- **Headless Cloud Deployment:** Completely removes the Windows COM bridge and local MetaTrader 5 (MT5) terminal dependencies. Runs on Linux containers (Docker / Railway) without sleep cycles with graceful `SIGTERM` handlers.
- **Dedicated Single-Asset Engine:** Locked to Gold (`XAU/USD`) with M5 execution, minimizing external API calls and maximizing signal resolution.
- **Streaming & Aggregated Data Layer:** Uses Twelve Data REST API for initial historical bootstrapping and real-time WebSockets (`wss://ws.twelvedata.com`) for live tick ingestion and M5 candle-close detection. Protected natively by an async API Rate Limiter.
- **Async Persistence Layer:** Direct PostgreSQL integration via `asyncpg` with a pooled connection to Neon PostgreSQL, utilizing `ON CONFLICT DO NOTHING` constraints to efficiently persist Order Block zones, Liquidity Pools, structure events, and signal history across container restarts.
- **Dynamic Risk Management:** Calculates Stop Losses dynamically using Average True Range (ATR) buffers, abandoning static pip measurements to adapt to real-time market volatility.
- **Fully Asynchronous Bot Interface:** Built on `aiogram v3` with HTML formatting, channel broadcast dispatching, and role-based command routing (Public vs. Admin).
- **Background Scheduling:** Non-blocking multi-timeframe scans driven by `APScheduler` (4H Macro Bias refresh every 4 hours, 1H Order Block scan hourly, 15M Confirmation sweeps).

## 2. Technology Stack

| Layer                   | Technology        | Purpose                                            |
| :---------------------- | :---------------- | :------------------------------------------------- |
| **Language**            | Python 3.13       | Core runtime environment                           |
| **Hosting**             | Railway           | Headless Linux container hosting (24/7 uptime)     |
| **Market Data**         | Twelve Data       | REST history + WebSocket live tick streaming       |
| **Database**            | Neon PostgreSQL   | Hosted serverless PostgreSQL with `asyncpg`        |
| **Telegram Framework**  | aiogram v3        | Asynchronous Telegram bot framework                |
| **Task Scheduler**      | APScheduler       | Non-blocking periodic multi-timeframe scans        |
| **Data Processing**     | pandas & numpy    | Vectorized OHLCV candle normalization and SMC math |
| **Config & Validation** | pydantic-settings | Strict type checking and `.env` validation         |
| **Logging**             | structlog         | Structured JSON logging for cloud log aggregation  |
| **Testing**             | pytest-asyncio    | Deterministic synthetic market data testing        |

---

## 3. Project Directory Structure

```bash
nexubot/
│
├── .env             # Local environment secrets (ignored by git)
├── .env.example     # Environment configuration template
├── .gitignore       # Git exclusions
├── Dockerfile       # Production container specification (python:3.13-slim)
├── railway.toml     # Railway deployment instructions
├── requirements.txt # Production Python dependencies (Strictly pinned)
├── README.md        # Project documentation
│
├── main.py # Application bootstrap, graceful shutdown, and asyncio orchestrator
│
├── config/
│   └── settings.py # Pydantic Settings singleton with strict validation
│
├── data/
│   ├── twelve_data_client.py # Twelve Data REST + WebSocket client (Rate-limited)
│   ├── candle_store.py       # In-memory rolling candle buffer (deque + asyncio.Lock)
│   └── normalizer.py         # Standardizes incoming OHLCV schemas to UTC DataFrames
│
├── strategies/
│   ├── models.py     # Dataclasses for Swings, Zones, Events, FVGs, and Signals
│   ├── sessions.py   # Session Killzones (Asia, London, NY) and NDO/NWO trackers
│   ├── structure.py  # ZigZag swings, BOS, CHoCH, MSS, and CISD logic
│   ├── smc.py        # OBs, OTE, Liquidity Sweeps, FVGs, Inducements, PD Arrays
│   └── confluence.py # Multi-timeframe engine (4H Bias → 1H Zone → 15M Confirm → 5M Entry)
│
├── db/
│   ├── migrations/
│   │   └── 002_smc_tables.sql # SMC tables (order_blocks, structure_events, signals)
│   ├── database.py            # asyncpg connection pool singleton
│   └── repositories.py        # Consolidated persistence classes (OBs, Events, Pools, Signals)
│
├── bot/
│   ├── dispatcher.py           # Bot initialization and channel broadcast functions
│   ├── handlers/
│   │   ├── commands.py         # Public commands (/start, /status, /bias, /session, /signals)
│   │   └── admin.py            # Admin-only commands (/zones, /scan)
│   └── formatters/
│       └── signal_formatter.py # HTML message templates for enhanced Telegram signals
│
├── scheduler/
│   └── jobs.py # APScheduler job definitions (4H, 1H, and 15M cycles)
│
├── utils/
│   ├── logger.py        # structlog JSON logging configuration
│   ├── rate_limiter.py  # Thread-safe Twelve Data REST limiter (avoids 429 errors)
│   └── math_helpers.py  # ATR SL logic, dynamic RRR, range checks, Fibs, and pip converters
│
└── tests/
    ├── conftest.py        # Deterministic synthetic market data fixtures (BOS, CHoCH, Sweep, FVG)
    ├── test_data_layer.py # Normalizer, time parsers, and CandleStore unit tests
    ├── test_structure.py  # Swings, BOS, CHoCH, MSS, CISD validation tests
    ├── test_smc.py        # FVG, Breaker, Sweep, OTE, Session, and Math helper tests
    └── test_confluence.py # Integration testing for the async multi-timeframe engine
```

## 4. SMC Trading Strategy Architecture

Nexubot operates on an advanced 4-step multi-timeframe confluence waterfall:

```
[Step 1: 4H Macro Bias]
└── detect_swings() -> detect_mss() -> detect_choch() -> detect_bos()
└── Establishes overarching directional bias (BULLISH / BEARISH).
└── Tracks New Day Open (NDO) and New Week Open (NWO) levels.

[Step 2: 1H Zone Identification]
└── Runs find_order_blocks() with dynamic displacement filters & FVG boosts.
└── Persists active zones to Neon PostgreSQL (ON CONFLICT IGNORE).
└── Continuously tracks Breaker Blocks (reclaimed mitigated OBs).

[Step 3: 15M Confirmation Layer]
└── Periodically aligns mid-timeframe structure shifts with 4H intent.

[Step 3: 5M Execution Trigger]
└── WebSocket tick-to-candle boundary detection triggers exact M5 rollover analysis.
└── Premium/Discount filter gates sub-optimal entries.
└── Confirms exact intersection (is_within_range) inside 1H OB, Breaker, or OTE Zone.
└── Checks for Institutional Sweeps (EQH / EQL) or Inducement (IDM) exhaustion.
└── Evaluates minimum confluence score (e.g., >= 70).
└── Calculates Dynamic ATR Stop Loss and strictly computes RRR.
└── Dispatches fully annotated HTML alert to Telegram.
```

## 5. Local Setup & Installation

#### Prerequisites

- Python 3.13+
- Neon PostgreSQL account
- Twelve Data API Key
- Telegram Bot Token (from `@BotFather`)

#### 1. Clone & Environment Setup

```bash
git clone https://github.com/QuintonCodes/nexubot.git
cd nexubot

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Environment Configuration

Copy the example environment file and populate your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
# Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ADMIN_ID=123456789
SIGNAL_CHANNEL_ID=-1001234567890

# Twelve Data
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
MAX_DAILY_API_CALLS=800

# Neon PostgreSQL
DATABASE_URL=postgresql://neondb_owner:password@ep-sample-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require

# Asset Scope & Timeframes
SYMBOLS=XAU/USD
HTF_TIMEFRAMES=4h,1h
LTF_TIMEFRAMES=15min,5min
ENTRY_TIMEFRAME=5min
CANDLE_BUFFER_SIZE=500

# Risk & Strategy Calibration
RISK_PERCENT=1.0
MIN_CONFLUENCE_SCORE=70
SIGNAL_COOLDOWN_HOURS=4
XAUUSD_PIP_TOLERANCE=40.0
SHADOW_MODE=false
```

#### 3. Database Schema Migration

Open the SQL Editor in your Neon Dashboard and execute the DDL script to generate the tracking tables required by the engine:

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS order_blocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('bullish', 'bearish')),
    ob_high DECIMAL(18, 5) NOT NULL,
    ob_low DECIMAL(18, 5) NOT NULL,
    ob_50 DECIMAL(18, 5) NOT NULL,
    strength_score DECIMAL(3, 2) NOT NULL DEFAULT 0.0,
    mitigated BOOLEAN NOT NULL DEFAULT FALSE,
    origin_timestamp TIMESTAMPTZ NOT NULL,
    mitigation_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, timeframe, direction, origin_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ob_active ON order_blocks(symbol, timeframe, mitigated)
    WHERE mitigated = FALSE;

CREATE TABLE IF NOT EXISTS structure_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('BOS', 'CHoCH', 'MSS', 'CISD')),
    direction TEXT NOT NULL CHECK (direction IN ('bullish', 'bearish')),
    price_level DECIMAL(18, 5) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_se_recent ON structure_events(symbol, timeframe, timestamp DESC);

CREATE TABLE IF NOT EXISTS liquidity_pools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    pool_type TEXT NOT NULL CHECK (pool_type IN ('EQH', 'EQL')),
    price_level DECIMAL(18, 5) NOT NULL,
    swept BOOLEAN NOT NULL DEFAULT FALSE,
    origin_timestamp TIMESTAMPTZ NOT NULL,
    sweep_timestamp TIMESTAMPTZ,
    UNIQUE (symbol, timeframe, pool_type, origin_timestamp)
);

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    entry_model TEXT,
    session TEXT,
    pd_zone TEXT,
    entry_price DECIMAL(18, 5) NOT NULL,
    stop_loss DECIMAL(18, 5) NOT NULL,
    take_profit_1 DECIMAL(18, 5) NOT NULL,
    take_profit_2 DECIMAL(18, 5) NOT NULL,
    risk_reward DECIMAL(5, 2) NOT NULL,
    confluence_score INTEGER,
    confluence_factors TEXT[],
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_recent ON signals(symbol, direction, timestamp DESC);
```

## 6. Running Tests

Run the full deterministic unit test suite:

```bash
python -m pytest tests/ -v
```

Run tests by module:

```bash
# Test Data Layer (Normalizer, Rate Limiter, and In-Memory Store)
python -m pytest tests/test_data_layer.py -v

# Test Market Structure (Swings, BOS, CHoCH, MSS, CISD)
python -m pytest tests/test_structure.py -v

# Test SMC Entry Models (FVGs, PD Arrays, Sessions, Sweeps, OTE)
python -m pytest tests/test_smc.py -v

# Test Confluence Engine (Integration & Mocked Repositories)
python -m pytest tests/test_confluence.py -v
```

## 7. Telegram Commands

#### Public Commands

- `/start` — Displays the bot overview and operational status.
- `/status` — Displays system health, dynamic API Rate Limit usage, and shadow mode status.
- `/bias` — Queries Neon PostgreSQL for the latest 4H macro trend direction.
- `/session` / `/killzone` — Displays current active ICT trading session.
- `/levels` — Prints the New Day Open (NDO) and New Week Open (NWO) reference targets.
- `/signals` — Fetches and displays the last 5 dispatched trading alerts.

#### Admin Commands (Restricted to `TELEGRAM_ADMIN_ID`)

- `/zones` — Lists all unmitigated 1H Order Blocks currently tracked in the database.
- `/scan` — Forces an immediate evaluation of current market structure for entry triggers.

## 8. Production Deployment (Railway)

#### 1. Pre-Deployment Docker Verification

Build and test the container locally:

```bash
docker build -t nexubot .
docker run --env-file .env nexubot
```

#### 2. Deploy to Railway

1. Push your code to a private GitHub repository.
2. Log in to [Railway](https://railway.com/) and create a New Project.

3. Select Deploy from GitHub repo and link your `nexubot` repository.

4. In the project dashboard, navigate to the Variables tab and add all variables defined in `.env.example`.

5. Railway will detect the `Dockerfile` and `railway.toml`, build the image, and start the application.

6. Verify startup logs in the Railway deployment console:
   - `database_pool_initialized`
   - `bootstrapping_data`
   - `scheduler_started`
   - `websocket_connected`
   - `telegram_bot_polling_started`
