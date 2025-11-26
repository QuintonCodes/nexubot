import time
import logging
import re
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import text, select
from src.config import DATABASE_URL

logger = logging.getLogger(__name__)

# --- ORM MODELS ---
class Base(DeclarativeBase):
    pass

class TradeResult(Base):
    """Stores individual trade signals"""
    __tablename__ = "trade_results"

    id: Mapped[str] = mapped_column(primary_key=True)
    timestamp: Mapped[float] = mapped_column()
    symbol: Mapped[str] = mapped_column()
    signal_type: Mapped[str] = mapped_column()
    confidence: Mapped[float] = mapped_column()
    entry_price: Mapped[float] = mapped_column()
    exit_price: Mapped[float] = mapped_column()
    result: Mapped[int] = mapped_column() # 1=Win, 0=Loss
    pnl_zar: Mapped[float] = mapped_column()
    strategy: Mapped[str] = mapped_column()

class SessionAnalytics(Base):
    """Stores summary data per session for long-term AI analysis"""
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
        connection_string = re.sub(r'^postgresql:', 'postgresql+asyncpg:', DATABASE_URL)

        # lean URL parameters: AsyncPG does not support 'sslmode' in the URL query.
        # We strip everything after '?' to avoid "unexpected keyword argument 'sslmode'"
        if "?" in connection_string:
            connection_string = connection_string.split("?")[0]

        self.engine = create_async_engine(
            connection_string,
            echo=False,
            pool_pre_ping=True,
            connect_args={
                "ssl": "require"
            }
            )
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_database(self):
        """Creates tables if they don't exist"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Test connection
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info("✅ Connected to Neon DB (PostgreSQL)")
        except Exception as e:
            logger.error(f"DB Init Failed: {e}")

    async def log_trade(self, trade_data: dict) -> None:
        """Logs a trade asynchronously"""
        async with self.async_session() as session:
            try:
                trade = TradeResult(
                    id=trade_data['id'],
                    timestamp=time.time(),
                    symbol=trade_data['symbol'],
                    signal_type=trade_data['signal'],
                    confidence=trade_data['confidence'],
                    entry_price=trade_data['entry'],
                    exit_price=trade_data['exit'],
                    result=1 if trade_data['won'] else 0,
                    pnl_zar=trade_data['pnl'],
                    strategy=trade_data['strategy']
                )
                session.add(trade)
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to log trade: {e}")
                await session.rollback()

    async def log_session(self, session_id: str, start_time: float, stats: dict):
        """Logs session summary on shutdown"""
        async with self.async_session() as session:
            try:
                win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0.0

                analytics = SessionAnalytics(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=time.time(),
                    total_trades=stats['total'],
                    win_rate=win_rate,
                    net_pnl_zar=stats['pnl_zar']
                )
                session.add(analytics)
                await session.commit()
                logger.info("💾 Session analytics saved to Cloud DB")
            except Exception as e:
                logger.error(f"Failed to log session: {e}")

    async def close(self):
        await self.engine.dispose()
