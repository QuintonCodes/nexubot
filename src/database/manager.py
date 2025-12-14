import logging
import json
import re
import time
from sqlalchemy import select, desc, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from typing import List

from src.config import DATABASE_URL, LOSS_COOLDOWN_DURATION

logger = logging.getLogger(__name__)


# --- ORM MODELS ---
class Base(DeclarativeBase):
    pass


class TradeResult(Base):
    __tablename__ = "trade_results"
    id: Mapped[str] = mapped_column(primary_key=True)
    timestamp: Mapped[float] = mapped_column()
    symbol: Mapped[str] = mapped_column()
    signal_type: Mapped[str] = mapped_column()
    confidence: Mapped[float] = mapped_column()
    entry_price: Mapped[float] = mapped_column()
    exit_price: Mapped[float] = mapped_column()
    result: Mapped[int] = mapped_column()  # 1=Win, 0=Loss
    pnl_zar: Mapped[float] = mapped_column()
    strategy: Mapped[str] = mapped_column()


class ActiveTrade(Base):
    __tablename__ = "active_trades"
    symbol: Mapped[str] = mapped_column(primary_key=True)
    signal_json: Mapped[str] = mapped_column()  # Stores the full signal dict as JSON
    start_time: Mapped[float] = mapped_column()


class SessionAnalytics(Base):
    __tablename__ = "session_analytics"
    session_id: Mapped[str] = mapped_column(primary_key=True)
    start_time: Mapped[float] = mapped_column()
    end_time: Mapped[float] = mapped_column()
    total_trades: Mapped[int] = mapped_column()
    win_rate: Mapped[float] = mapped_column()
    net_pnl_zar: Mapped[float] = mapped_column()


# --- MANAGER ---
class DatabaseManager:
    """
    Async Database Manager for Neon (PostgreSQL).
    """

    def __init__(self):
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL is missing in .env")

        # Fix protocol for SQLAlchemy + AsyncPG
        connection_string = re.sub(r"^postgresql:", "postgresql+asyncpg:", DATABASE_URL)
        if "?" in connection_string:
            connection_string = connection_string.split("?")[0]

        self.engine = create_async_engine(
            connection_string, echo=False, pool_pre_ping=True, connect_args={"ssl": "require"}
        )
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_database(self):
        """Creates tables if they don't exist"""
        if not self.engine:
            return
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ DB Connected")
        except Exception as e:
            logger.error(f"DB Init Failed: {e}")

    async def save_active_trade(self, symbol: str, signal: dict):
        """Saves an active trade to DB"""
        if not self.engine:
            return
        async with self.async_session() as session:
            try:
                # Upsert logic
                await session.execute(delete(ActiveTrade).where(ActiveTrade.symbol == symbol))

                trade = ActiveTrade(symbol=symbol, signal_json=json.dumps(signal), start_time=time.time())
                session.add(trade)
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to save active trade {symbol}: {e}")

    async def delete_active_trade(self, symbol: str):
        """Deletes an active trade from DB"""
        if not self.engine:
            return
        async with self.async_session() as session:
            try:
                await session.execute(delete(ActiveTrade).where(ActiveTrade.symbol == symbol))
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to delete active trade {symbol}: {e}")

    async def get_active_trades(self) -> List:
        """Returns list of (symbol, signal_dict, start_time)"""
        if not self.engine:
            return []
        async with self.async_session() as session:
            try:
                result = await session.execute(select(ActiveTrade))
                trades = result.scalars().all()
                return [(t.symbol, json.loads(t.signal_json), t.start_time) for t in trades]
            except Exception as e:
                logger.error(f"Failed to fetch active trades: {e}")
                return []

    async def log_trade(self, trade_data: dict):
        """Logs a trade asynchronously"""
        if not self.engine:
            return
        async with self.async_session() as session:
            try:
                trade = TradeResult(
                    id=trade_data["id"],
                    timestamp=time.time(),
                    symbol=trade_data["symbol"],
                    signal_type=trade_data["signal"],
                    confidence=trade_data["confidence"],
                    entry_price=trade_data["entry"],
                    exit_price=trade_data["exit"],
                    result=1 if trade_data["won"] else 0,
                    pnl_zar=trade_data["pnl"],
                    strategy=trade_data["strategy"],
                )
                session.add(trade)
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to log trade: {e}")
                await session.rollback()

    async def log_session(self, session_id: str, start_time: float, stats: dict):
        """Logs session summary on shutdown"""
        if not self.engine:
            return
        async with self.async_session() as session:
            try:
                win_rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
                analytics = SessionAnalytics(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=time.time(),
                    total_trades=stats["total"],
                    win_rate=win_rate,
                    net_pnl_zar=stats["pnl_zar"],
                )
                session.add(analytics)
                await session.commit()
                logger.info("💾 Session analytics saved to Cloud DB")
            except Exception as e:
                logger.error(f"Failed to log session: {e}")

    async def check_recent_loss(self, symbol: str) -> bool:
        """
        Returns True if the symbol had a loss recently (Cool-down check).
        Prevents overusage of failing pairs.
        """
        if not self.engine:
            return False
        async with self.async_session() as session:
            try:
                cutoff = time.time() - LOSS_COOLDOWN_DURATION
                stmt = (
                    select(TradeResult)
                    .where(
                        TradeResult.symbol == symbol,
                        TradeResult.timestamp > cutoff,
                        TradeResult.result == 0,  # 0 is Loss
                    )
                    .order_by(desc(TradeResult.timestamp))
                )

                result = await session.execute(stmt)
                return result.scalars().first() is not None
            except Exception:
                return False

    async def get_strategy_performance(self, strategy_name: str) -> float:
        """Returns the win rate (0.0 to 1.0) of a specific strategy from DB."""
        if not self.engine:
            return 0.5
        async with self.async_session() as session:
            try:
                stmt = select(TradeResult.result).where(TradeResult.strategy == strategy_name)
                result = await session.execute(stmt)
                results = result.scalars().all()

                if not results or len(results) < 5:
                    return 0.5  # Neutral if no data
                return sum(results) / len(results)
            except Exception:
                return 0.5

    async def get_pair_performance(self, symbol: str) -> float:
        """
        Returns win rate for a specific pair.
        Used to adjust confidence if a pair is historically profitable.
        """
        if not self.engine:
            return 0.5
        async with self.async_session() as session:
            stmt = select(TradeResult.result).where(TradeResult.symbol == symbol)
            result = await session.execute(stmt)
            results = result.scalars().all()
            if len(results) < 10:  # Insufficient data
                return 0.5
            return sum(results) / len(results)

    async def cleanup_db(self):
        """Removes logs older than 30 days."""
        if not self.engine:
            return
        async with self.async_session() as session:
            cutoff = time.time() - (86400 * 30)
            stmt = delete(TradeResult).where(TradeResult.timestamp < cutoff)
            await session.execute(stmt)
            await session.commit()
            logger.info("🧹 Database cleaned.")

    async def close(self):
        """Closes the database connection"""
        if self.engine:
            await self.engine.dispose()
