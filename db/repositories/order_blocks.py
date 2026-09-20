from datetime import datetime
from typing import List

from db.database import get_pool
from strategies.models import OrderBlock


async def save_order_block(ob: OrderBlock) -> None:
    pool = get_pool()
    query = """
        INSERT INTO order_blocks (id, symbol, timeframe, direction, ob_high, ob_low, ob_50, strength_score, is_mitigated, origin_timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (id) DO NOTHING
    """
    await pool.execute(
        query,
        ob.id,
        ob.symbol,
        ob.timeframe,
        ob.direction,
        ob.ob_high,
        ob.ob_low,
        ob.ob_50,
        ob.strength_score,
        ob.is_mitigated,
        ob.origin_timestamp,
    )


async def get_active_order_blocks(symbol: str, timeframe: str) -> List[OrderBlock]:
    pool = get_pool()
    query = """
        SELECT * FROM order_blocks
        WHERE symbol = $1 AND timeframe = $2 AND is_mitigated = FALSE
    """
    records = await pool.fetch(query, symbol, timeframe)

    return [
        OrderBlock(
            id=str(r["id"]),
            symbol=r["symbol"],
            timeframe=r["timeframe"],
            direction=r["direction"],
            ob_high=float(r["ob_high"]),
            ob_low=float(r["ob_low"]),
            ob_50=float(r["ob_50"]),
            origin_timestamp=r["origin_timestamp"],
            is_mitigated=r["is_mitigated"],
            mitigation_timestamp=r["mitigation_timestamp"],
            strength_score=float(r["strength_score"]),
        )
        for r in records
    ]


async def mark_mitigated(ob_id: str, mitigation_timestamp: datetime) -> None:
    pool = get_pool()
    query = "UPDATE order_blocks SET is_mitigated = TRUE, mitigation_timestamp = $1 WHERE id = $2"
    await pool.execute(query, mitigation_timestamp, ob_id)
