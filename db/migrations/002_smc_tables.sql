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