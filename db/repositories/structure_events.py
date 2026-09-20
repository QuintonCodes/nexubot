import uuid
from typing import Optional

from db.database import get_pool
from strategies.models import StructureEvent


async def save_event(event: StructureEvent) -> None:
    pool = get_pool()
    query = """
        INSERT INTO structure_events (id, symbol, timeframe, event_type, direction, price_level, event_timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    """
    await pool.execute(
        query,
        str(uuid.uuid4()),
        event.symbol,
        event.timeframe,
        event.event_type,
        event.direction,
        event.price_level,
        event.timestamp,
    )


async def get_latest_bias(symbol: str, timeframe: str) -> Optional[str]:
    """Retrieves the direction of the last confirmed BOS or CHoCH to establish HTF Bias."""
    pool = get_pool()
    query = """
        SELECT direction FROM structure_events
        WHERE symbol = $1 AND timeframe = $2 AND event_type IN ('BOS', 'CHoCH')
        ORDER BY event_timestamp DESC LIMIT 1
    """
    row = await pool.fetchrow(query, symbol, timeframe)
    return row["direction"] if row else None
