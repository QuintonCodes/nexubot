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
    is_breaker BOOLEAN NOT NULL DEFAULT FALSE,
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_structure_event UNIQUE (symbol, timeframe, event_type, timestamp)
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
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    entry_model TEXT,
    session TEXT,
    pd_zone TEXT,
    entry_price DECIMAL(18, 5) NOT NULL,
    stop_loss DECIMAL(18, 5) NOT NULL,
    take_profit_1 DECIMAL(18, 5) NOT NULL,
    take_profit_2 DECIMAL(18, 5) NOT NULL,
    take_profit_3 DECIMAL(18, 5),
    risk_reward DECIMAL(5, 2) NOT NULL,
    confluence_score INTEGER,
    confluence_factors TEXT[],
    status TEXT NOT NULL DEFAULT 'active',
    telegram_message_id BIGINT,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_recent ON signals(symbol, direction, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_dedup ON signals(symbol, timeframe, direction, timestamp DESC);