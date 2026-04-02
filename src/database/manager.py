import asyncio
import logging
import json
import numpy as np
import re
import time
from functools import wraps
from sqlalchemy import select, desc, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from typing import List

from src.config import DATABASE_URL, LOSS_COOLDOWN_DURATION

logger = logging.getLogger(__name__)


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(SafeEncoder, self).default(obj)


# --- RETRY DECORATOR ---
def db_retry(retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Filter out network noise
                    if "getaddrinfo" in str(e) or "connection" in str(e).lower():
                        if i < retries - 1:
                            await asyncio.sleep(delay)
                            continue
                        logger.error(f"DB Error in {func.__name__}: {e}")
                        break
                    else:
                        logger.error(f"DB Critical Error in {func.__name__}: {e}")
                        break
            # Return default empty values on failure
            if "get_total" in func.__name__ or "performance" in func.__name__:
                return 0.0
            if "get_active" in func.__name__ or "get_history" in func.__name__:
                return []
            return None

        return wrapper

    return decorator


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
    pnl: Mapped[float] = mapped_column()
    currency: Mapped[str] = mapped_column(default="USD")
    strategy: Mapped[str] = mapped_column()
    size: Mapped[float] = mapped_column(default=0.01)


class ActiveTrade(Base):
    __tablename__ = "active_trades"
    symbol: Mapped[str] = mapped_column(primary_key=True)
    signal_json: Mapped[str] = mapped_column()
    start_time: Mapped[float] = mapped_column()


class SessionAnalytics(Base):
    __tablename__ = "session_analytics"
    session_id: Mapped[str] = mapped_column(primary_key=True)
    start_time: Mapped[float] = mapped_column()
    end_time: Mapped[float] = mapped_column()
    total_trades: Mapped[int] = mapped_column()
    win_rate: Mapped[float] = mapped_column()
    net_pnl: Mapped[float] = mapped_column()
    currency: Mapped[str] = mapped_column(default="USD")


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
            connection_string,
            echo=False,
            connect_args={"ssl": "require"},
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
            pool_pre_ping=True,
        )
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

        # {symbol: (win_rate, timestamp)}
        self._performance_cache = {}

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

    async def cleanup_db(self):
        """Removes logs older than 30 days."""
        if not self.engine:
            return

        try:
            async with self.async_session() as session:
                # Clean empty sessions (0 trades)
                stmt_sessions = delete(SessionAnalytics).where(SessionAnalytics.total_trades == 0)
                await session.execute(stmt_sessions)

                await session.commit()
                logger.info("🧹 Database cleaned (Old logs & Empty sessions removed).")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def close(self):
        """Closes the database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("🔒 Database Connection Closed.")

    async def delete_active_trade(self, symbol: str):
        """Deletes an active trade from DB"""
        if not self.engine:
            return

        try:
            async with self.async_session() as session:
                await session.execute(delete(ActiveTrade).where(ActiveTrade.symbol == symbol))
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to delete active trade {symbol}: {e}")

    @db_retry()
    async def get_active_trades(self) -> List:
        """Returns list of (symbol, signal_dict, start_time)"""
        if not self.engine:
            return []

        try:
            async with self.async_session() as session:
                result = await session.execute(select(ActiveTrade))
                trades = result.scalars().all()
                return [(t.symbol, json.loads(t.signal_json), t.start_time) for t in trades]
        except Exception as e:
            logger.error(f"Failed to fetch active trades: {e}")
            return []

    async def get_pair_performance(self, symbol: str) -> float:
        """
        Returns win rate for a specific pair.
        Used to adjust confidence if a pair is historically profitable.
        """
        if not self.engine:
            return 0.5

        # 1. Check Memory Cache (Valid for 10 minutes)
        if symbol in self._performance_cache:
            val, timestamp = self._performance_cache[symbol]
            if time.time() - timestamp < 600:
                return val

        try:
            async with self.async_session() as session:
                stmt = select(TradeResult.result).where(TradeResult.symbol == symbol)
                result = await session.execute(stmt)
                results = result.scalars().all()

                win_rate = 0.5
                if len(results) >= 10:
                    win_rate = sum(results) / len(results)

                # Cache the result
                self._performance_cache[symbol] = (win_rate, time.time())
                return win_rate
        except Exception:
            return 0.5

    @db_retry()
    async def get_recent_trades(self, limit=5) -> List[TradeResult]:
        """Fetches only the last N trades for display."""
        if not self.engine:
            return []

        try:
            async with self.async_session() as session:
                stmt = select(TradeResult).order_by(TradeResult.timestamp.desc()).limit(limit)
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Recent Trades Error: {e}")
            return []

    @db_retry()
    async def get_total_historical_win_rate(self) -> float:
        """
        Calculates the win rate across ALL trades stored in the database.
        """
        if not self.engine:
            return 0.0

        try:
            async with self.async_session() as session:
                # Select only the result column (1=Win, 0=Loss)
                stmt = select(TradeResult.result)
                result = await session.execute(stmt)
                outcomes = result.scalars().all()

                total = len(outcomes)
                if total == 0:
                    return 0.0

                wins = sum(outcomes)
                return (wins / total) * 100.0
        except Exception as e:
            logger.error(f"Failed to fetch historical win rate: {e}")
            return 0.0

    async def init_database(self):
        """Creates tables if they don't exist"""
        if not self.engine:
            return

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("✅ DB Connected")
        except Exception as e:
            logger.warning(f"⚠️ DB Connection failed: {e}")

    async def log_session(self, session_id: str, start_time: float, stats: dict):
        """Logs session summary on shutdown"""
        if not self.engine:
            return

        try:
            async with self.async_session() as session:
                win_rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
                analytics = SessionAnalytics(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=time.time(),
                    total_trades=stats["total"],
                    win_rate=win_rate,
                    net_pnl=stats.get("pnl", 0.0),
                    currency=stats.get("currency", "USD"),
                )
                session.add(analytics)
                await session.commit()
                logger.info("💾 Session analytics saved to Cloud DB")
        except Exception as e:
            logger.error(f"Failed to log session: {e}")

    @db_retry()
    async def log_trade(self, trade_data: dict):
        """Logs a trade asynchronously"""
        if not self.engine:
            return

        try:
            async with self.async_session() as session:
                trade = TradeResult(
                    id=trade_data["id"],
                    timestamp=time.time(),
                    symbol=trade_data["symbol"],
                    signal_type=trade_data["signal"],
                    confidence=trade_data["confidence"],
                    entry_price=trade_data["entry"],
                    exit_price=trade_data["exit"],
                    result=1 if trade_data["won"] else 0,
                    pnl=round(trade_data["pnl"], 2),
                    currency=trade_data.get("currency", "USD"),
                    strategy=trade_data["strategy"],
                    size=trade_data.get("lot_size", 0.01),
                )
                session.add(trade)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            await session.rollback()

    async def save_active_trade(self, symbol: str, signal: dict):
        """Saves an active trade to DB"""
        if not self.engine:
            return

        try:
            json_str = json.dumps(signal, cls=SafeEncoder)

            async with self.async_session() as session:
                stmt = pg_insert(ActiveTrade).values(symbol=symbol, signal_json=json_str, start_time=time.time())

                # If symbol exists, update the signal info and time
                do_update_stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol"], set_=dict(signal_json=json_str, start_time=time.time())
                )

                await session.execute(do_update_stmt)
                await session.commit()
        except TypeError as te:
            logger.error(f"JSON Serialization Failed for {symbol}: {te}")
        except Exception as e:
            logger.error(f"Failed to save active trade {symbol}: {e}")
