# 🚀 Nexubot — Cloud-Native SMC Trading Signal Engine

![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)

**Nexubot** is a cloud-native, asynchronous Telegram signal trading bot built specifically for **XAU/USD (Gold)** on the **M5 timeframe**. It implements algorithmic Smart Money Concepts (SMC)—including Market Structure Shifts (BOS, CHoCH, MSS, CISD), Order Blocks, ICT Optimal Trade Entries (OTE), and Liquidity Sweeps—operating 24/7 in a headless Linux container environment.

## 1. Architectural Highlights

- **Headless Cloud Deployment:** Completely removes the Windows COM bridge and local MetaTrader 5 (MT5) terminal dependencies. Runs on Linux containers (Docker / Railway) without sleep cycles.
- **Dedicated Single-Asset Engine:** Locked to Gold (`XAU/USD`) with M5 execution, minimizing external API calls and maximizing signal resolution.
- **Streaming & Aggregated Data Layer:** Uses Twelve Data REST API for initial historical bootstrapping and real-time WebSockets (`wss://ws.twelvedata.com`) for live tick ingestion and M5 candle-close detection.
- **Async Persistence Layer:** Direct PostgreSQL integration via `asyncpg` with a pooled connection to Neon PostgreSQL, persisting Order Block zones, structure events, and signal history across container restarts.
- **Fully Asynchronous Bot Interface:** Built on `aiogram v3` with HTML formatting, channel broadcast dispatching, and role-based command routing (Public vs. Admin).
- **Background Scheduling:** Non-blocking multi-timeframe scans driven by `APScheduler` (4H Macro Bias refresh every 4 hours, 1H Order Block scan every hour).

## 2. Technology Stack

| Layer                   | Technology        | Purpose                                            |
| :---------------------- | :---------------- | :------------------------------------------------- |
| **Language**            | Python 3.12       | Core runtime environment                           |
| **Hosting**             | Railway           | Headless Linux container hosting (24/7 uptime)     |
| **Market Data**         | Twelve Data       | REST history + WebSocket live tick streaming       |
| **Database**            | Neon PostgreSQL   | Hosted serverless PostgreSQL with `asyncpg`        |
| **Telegram Framework**  | aiogram v3        | Asynchronous Telegram bot framework                |
| **Task Scheduler**      | APScheduler       | Non-blocking periodic multi-timeframe scans        |
| **Data Processing**     | pandas & numpy    | Vectorized OHLCV candle normalization and SMC math |
| **Config & Validation** | pydantic-settings | Strict type checking and `.env` validation         |
| **Logging**             | structlog         | Structured JSON logging for cloud log aggregation  |

---

## 3. Project Directory Structure

```bash
nexubot/
│
├── .env # Local environment secrets (ignored by git)
├── .env.example # Environment configuration template
├── .gitignore # Git exclusions
├── Dockerfile # Production container specification (python:3.12-slim)
├── railway.toml # Railway deployment instructions
├── requirements.txt # Production Python dependencies
├── README.md # Project documentation
│
├── main.py # Application bootstrap and asyncio orchestrator
│
├── config/
│ ├── init.py
│ └── settings.py # Pydantic Settings singleton with strict validation
│
├── data/
│ ├── init.py
│ ├── twelve_data_client.py # Twelve Data REST + WebSocket client
│ ├── candle_store.py # In-memory rolling candle buffer (deque + asyncio.Lock)
│ └── normalizer.py # Standardizes incoming OHLCV schemas to UTC DataFrames
│
├── strategies/
│ ├── init.py
│ ├── models.py # Dataclasses for Swings, Zones, Events, and Signals
│ ├── structure.py # ZigZag swings, BOS, CHoCH, MSS, and CISD logic
│ ├── smc.py # Order Blocks, ICT OTE zones, and Liquidity Sweeps
│ └── confluence.py # Multi-timeframe waterfall (4H Bias → 1H Zone → 5M Entry)
│
├── db/
│ ├── init.py
│ ├── database.py # asyncpg connection pool singleton
│ ├── migrations/
│ │ ├── 001_existing_schema.sql # Reference schema documentation
│ │ └── 002_smc_tables.sql # SMC tables (order_blocks, structure_events, signals)
│ └── repositories/
│ ├── init.py
│ ├── order_blocks.py # Order Block persistence and mitigation tracking
│ ├── signals.py # Signal history, audit trail, and deduplication
│ └── structure_events.py # Historical BOS / CHoCH bias retrieval
│
├── bot/
│ ├── init.py
│ ├── dispatcher.py # Bot initialization and channel broadcast functions
│ ├── handlers/
│ │ ├── init.py
│ │ ├── commands.py # Public commands (/start, /status, /bias, /signals)
│ │ └── admin.py # Admin-only commands (/zones, /scan)
│ └── formatters/
│ ├── init.py
│ └── signal_formatter.py # HTML message templates for signals
│
├── scheduler/
│ ├── init.py
│ └── jobs.py # APScheduler job definitions for 4H and 1H cycles
│
├── utils/
│ ├── init.py
│ ├── logger.py # structlog JSON logging configuration
│ ├── rate_limiter.py # Twelve Data REST rate limiter (800 calls/day budget)
│ └── math_helpers.py # Gold pip converters, RRR, and Fibonacci calculations
│
└── tests/
├── init.py
├── conftest.py # Deterministic synthetic market data fixtures
├── test_data_layer.py # Normalizer and CandleStore unit tests
├── test_structure.py # Swings, BOS, and CHoCH validation tests
├── test_smc.py # Order Block, OTE, and Liquidity Sweep tests
└── test_confluence.py # Integration test for the multi-timeframe engine
```

## 4. SMC Trading Strategy Architecture

Nexubot operates on a 3-step multi-timeframe confluence waterfall:

```
[Step 1: 4H Macro Bias]
└── detect_swings() -> classify_structure() -> detect_bos()
└── Determines directional bias: BULLISH or BEARISH

[Step 2: 1H Zone Identification]
└── Runs detect_order_blocks() aligned with 4H Bias
└── Persists active zones to Neon PostgreSQL
└── Continuously tracks 50% midpoint mitigation

[Step 3: 5M Execution Trigger]
└── WebSocket tick-to-candle boundary detection on M5 rollover
└── Confirms price is inside an active, unmitigated 1H Order Block or OTE Zone
└── Checks for preceding Liquidity Sweep (EQH / EQL)
└── Evaluates signal deduplication against the cooldown window (default 4 hours)
└── Dispatches formatted HTML alert to the Telegram channel
```

## 5. Local Setup & Installation

#### Prerequisites

- Python 3.12+
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
TELEGRAM_CHANNEL_ID=-1001234567890
TELEGRAM_ADMIN_ID=123456789

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

Open the SQL Editor in your Neon Dashboard and execute the DDL script located at `db/migrations/002_smc_tables.sql`:

```sql
CREATE TABLE IF NOT EXISTS order_blocks (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('bullish', 'bearish')),
    ob_high DECIMAL(18, 5) NOT NULL,
    ob_low DECIMAL(18, 5) NOT NULL,
    ob_50 DECIMAL(18, 5) NOT NULL,
    strength_score DECIMAL(3, 2) NOT NULL DEFAULT 0.0,
    is_mitigated BOOLEAN NOT NULL DEFAULT FALSE,
    origin_timestamp TIMESTAMPTZ NOT NULL,
    mitigation_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ob_active ON order_blocks(symbol, timeframe, is_mitigated)
    WHERE is_mitigated = FALSE;

CREATE TABLE IF NOT EXISTS structure_events (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('BOS', 'CHoCH', 'MSS', 'CISD')),
    direction TEXT NOT NULL CHECK (direction IN ('bullish', 'bearish')),
    price_level DECIMAL(18, 5) NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_se_recent ON structure_events(symbol, timeframe, created_at DESC);

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    signal_type TEXT NOT NULL,
    entry_price DECIMAL(18, 5) NOT NULL,
    stop_loss DECIMAL(18, 5) NOT NULL,
    take_profit_1 DECIMAL(18, 5) NOT NULL,
    take_profit_2 DECIMAL(18, 5) NOT NULL,
    risk_reward DECIMAL(5, 2) NOT NULL,
    confluence_score INTEGER,
    confluence_factors TEXT[],
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    telegram_message_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_signals_recent ON signals(symbol, direction, sent_at DESC);
```

## 6. Running Tests

Run the full deterministic unit test suite:

```bash
python -m pytest tests/ -v
```

Run tests by module:

```bash
# Test Data Layer (Normalizer & In-Memory Store)
python -m pytest tests/test_data_layer.py -v

# Test Market Structure (Swings, BOS, CHoCH)
python -m pytest tests/test_structure.py -v

# Test SMC Entry Models (Order Blocks, OTE, Sweeps)
python -m pytest tests/test_smc.py -v

# Test Confluence Engine (Integration & Mocked Repositories)
python -m pytest tests/test_confluence.py -v
```

## 7. Telegram Commands

#### Public Commands

- `/start` — Displays the bot overview and operational status.
- `/status` — Displays system health, asset scope, active timeframes, and shadow mode status.
- `/bias` — Queries Neon PostgreSQL for the latest 4H macro trend direction.
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
