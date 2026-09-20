from datetime import datetime, timedelta, timezone
from typing import List

from db.database import get_pool
from config.settings import settings
from strategies.models import TradeSignal
from utils.math_helpers import price_to_pips


async def save_signal(signal: TradeSignal) -> None:
    pool = get_pool()
    query = """
        INSERT INTO signals (id, symbol, direction, signal_type, entry_price, stop_loss, take_profit_1, take_profit_2, risk_reward, confluence_score, confluence_factors, sent_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    """
    await pool.execute(
        query,
        signal.signal_id,
        signal.symbol,
        signal.direction,
        signal.signal_type,
        signal.entry_price,
        signal.stop_loss,
        signal.take_profit_1,
        signal.take_profit_2,
        signal.risk_reward,
        signal.confluence_score,
        signal.confluence_factors,
        signal.timestamp,
    )


async def is_duplicate(
    symbol: str, direction: str, entry_price: float, pip_tolerance: float, cooldown_hours: int
) -> bool:
    """Checks if an identical signal was already sent recently."""
    pool = get_pool()
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)

    query = """
        SELECT entry_price FROM signals
        WHERE symbol = $1 AND direction = $2 AND sent_at > $3
    """
    records = await pool.fetch(query, symbol, direction, cutoff_time)

    for r in records:
        if price_to_pips(abs(float(r["entry_price"]) - entry_price), symbol) <= pip_tolerance:
            return True
    return False


async def get_recent_signals(symbol: str, limit: int = 5) -> List[TradeSignal]:
    """Fetches the most recent trade signals for the Telegram /signals command."""
    pool = get_pool()
    query = """
        SELECT * FROM signals
        WHERE symbol = $1
        ORDER BY sent_at DESC LIMIT $2
    """
    records = await pool.fetch(query, symbol, limit)

    return [
        TradeSignal(
            signal_id=str(r["id"]),
            symbol=r["symbol"],
            direction=r["direction"],
            signal_type=r["signal_type"],
            entry_price=float(r["entry_price"]),
            stop_loss=float(r["stop_loss"]),
            take_profit_1=float(r["take_profit_1"]),
            take_profit_2=float(r["take_profit_2"]),
            risk_reward=float(r["risk_reward"]),
            confluence_score=r["confluence_score"],
            confluence_factors=r["confluence_factors"],
            timestamp=r["sent_at"],
            timeframe=settings.ENTRY_TIMEFRAME if "settings" in globals() else "5min",
        )
        for r in records
    ]
