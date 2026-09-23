import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from db.database import get_pool
from strategies.models import LiquidityPool, OrderBlock, StructureEvent, TradeSignal

logger = logging.getLogger(__name__)


class OrderBlockRepository:
    """Manages persistence for institutional Order Blocks (OB)."""

    async def save_order_block(self, ob: OrderBlock) -> None:
        """
        Saves an order block to the database.
        Implements ON CONFLICT DO NOTHING to prevent duplicate unmitigated OBs.
        """
        query = """
            INSERT INTO order_blocks
            (symbol, timeframe, direction, ob_high, ob_low, ob_50, origin_timestamp, strength_score, mitigated)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (symbol, timeframe, direction, origin_timestamp) DO NOTHING;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    query,
                    ob.symbol,
                    ob.timeframe,
                    ob.direction,
                    ob.ob_high,
                    ob.ob_low,
                    ob.ob_50,
                    ob.origin_timestamp,
                    ob.strength_score,
                    False,
                )
            except Exception as e:
                logger.error("Failed to save order block", exc_info=e)

    async def get_active_order_blocks(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetches all unmitigated order blocks for a specific symbol and timeframe within the last 14 days."""
        query = """
            SELECT * FROM order_blocks
            WHERE symbol = $1
              AND timeframe = $2
              AND mitigated = FALSE
              AND origin_timestamp >= NOW() - INTERVAL '14 days'
            ORDER BY origin_timestamp DESC;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe)
            return [dict(row) for row in rows]

    async def get_active_breaker_blocks(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetches previously mitigated blocks that are now eligible as Breaker Blocks."""
        query = """
            SELECT * FROM order_blocks
            WHERE symbol = $1
              AND timeframe = $2
              AND mitigated = TRUE
              AND is_breaker = TRUE
              AND origin_timestamp >= NOW() - INTERVAL '14 days'
            ORDER BY origin_timestamp DESC
            LIMIT 10;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe)
            return [dict(row) for row in rows]

    async def mark_mitigated(self, ob_id: int) -> None:
        """Marks an order block as mitigated when price action pierces the zone."""
        query = """
            UPDATE order_blocks
            SET mitigated = TRUE, mitigation_timestamp = $2
            WHERE id = $1;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, ob_id, datetime.now(timezone.utc))

    async def mark_as_breaker(self, symbol: str, timeframe: str, direction: str) -> None:
        """Promotes mitigated order blocks to breaker blocks after an opposing structural shift."""
        query = """
            UPDATE order_blocks
            SET is_breaker = TRUE
            WHERE symbol = $1 AND timeframe = $2 AND direction = $3 AND mitigated = TRUE;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, symbol, timeframe, direction)


class StructureEventRepository:
    """Tracks Market Structure Shifts, BOS, and CHoCH events."""

    async def save_event(self, event: StructureEvent) -> None:
        query = """
            INSERT INTO structure_events
            (symbol, timeframe, event_type, direction, price_level, timestamp, confirmed)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (symbol, timeframe, event_type, timestamp) DO NOTHING;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                event.symbol,
                event.timeframe,
                event.event_type,
                event.direction,
                event.price_level,
                event.timestamp,
                event.confirmed,
            )

    async def get_latest_bias(self, symbol: str, timeframe: str) -> Optional[str]:
        """Retrieves the most recent structural bias."""
        query = """
            SELECT event_type, direction FROM structure_events
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, symbol, timeframe)
            return row.get("direction") if row else None


class SignalRepository:
    """Handles persistence and deduplication of trade signals."""

    async def save_signal(self, signal: TradeSignal) -> None:
        query = """
            INSERT INTO signals (
                id, symbol, timeframe, direction, entry_model, session, pd_zone,
                entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                risk_reward, confluence_score, confluence_factors, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            ON CONFLICT (id) DO NOTHING;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                signal.signal_id,
                signal.symbol,
                signal.timeframe,
                signal.direction,
                signal.entry_model,
                signal.session,
                signal.pd_zone,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit_1,
                signal.take_profit_2,
                signal.take_profit_3,
                signal.risk_reward,
                signal.confluence_score,
                signal.confluence_factors,
                signal.timestamp,
            )

    async def is_duplicate(self, symbol: str, timeframe: str, direction: str, window_minutes: int = 30) -> bool:
        """Checks if an identical signal was fired recently to avoid execution spam."""
        query = """
            SELECT COUNT(*) FROM signals
            WHERE symbol = $1
              AND timeframe = $2
              AND direction = $3
              AND timestamp >= NOW() - INTERVAL '1 minute' * $4;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(query, symbol, timeframe, direction, window_minutes)
            return count > 0

    async def get_recent_signals(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM signals
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT $2;
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, limit)
            return [dict(row) for row in rows]

    async def get_active_signals(self, symbol: str) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM signals
            WHERE symbol = $1 AND status = 'active'
            AND timestamp >= NOW() - INTERVAL '48 hours';
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol)
            return [dict(row) for row in rows]

    async def update_signal_status(self, signal_id: str, status: str) -> None:
        query = "UPDATE signals SET status = $2 WHERE id = $1;"
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, signal_id, status)

    async def update_telegram_message_id(self, signal_id: str, message_id: int) -> None:
        query = "UPDATE signals SET telegram_message_id = $2 WHERE id = $1;"
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, signal_id, message_id)


class LiquidityPoolRepository:
    """Persists EQH/EQL state for sweep detection."""

    async def save_pool(self, pool: LiquidityPool) -> None:
        query = """
            INSERT INTO liquidity_pools (symbol, timeframe, pool_type, price_level, origin_timestamp, swept)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (symbol, timeframe, pool_type, origin_timestamp) DO NOTHING;
        """
        db_pool = get_pool()
        async with db_pool.acquire() as conn:
            origin = pool.sweep_timestamp or datetime.now(timezone.utc)
            await conn.execute(query, pool.symbol, pool.timeframe, pool.pool_type, pool.price_level, origin, False)


order_blocks = OrderBlockRepository()
structure_events = StructureEventRepository()
signals = SignalRepository()
liquidity_pools = LiquidityPoolRepository()
