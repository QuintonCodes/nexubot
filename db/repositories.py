import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from db.database import get_pool
from strategies.models import OrderBlock

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
            (symbol, timeframe, direction, ob_high, ob_low, origin_timestamp, strength_score, mitigated)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
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
                    ob.origin_timestamp,
                    ob.strength_score,
                    False,
                )
            except Exception as e:
                logger.error("Failed to save order block", exc_info=e)

    async def get_active_order_blocks(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetches all unmitigated order blocks for a specific symbol and timeframe."""
        query = """
            SELECT * FROM order_blocks
            WHERE symbol = $1 AND timeframe = $2 AND mitigated = FALSE
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
            WHERE symbol = $1 AND timeframe = $2 AND mitigated = TRUE
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


class StructureEventRepository:
    """Tracks Market Structure Shifts, BOS, and CHoCH events."""

    async def save_event(self, event: Any) -> None:
        query = """
            INSERT INTO structure_events
            (symbol, timeframe, event_type, price_level, timestamp)
            VALUES ($1, $2, $3, $4, $5);
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                query, event.symbol, event.timeframe, event.event_type, event.price_level, event.timestamp
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

    async def save_signal(self, signal: Any) -> None:
        query = """
            INSERT INTO signals
            (symbol, timeframe, direction, entry_price, stop_loss, take_profit, confluence_score, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                signal.symbol,
                signal.timeframe,
                signal.direction,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit_1,
                signal.confluence_score,
                signal.timestamp,
            )

    async def is_duplicate(self, symbol: str, timeframe: str, direction: str, window_minutes: int = 15) -> bool:
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


class LiquidityPoolRepository:
    """Persists EQH/EQL state for sweep detection."""

    async def save_pool(self, pool: Any) -> None:
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
