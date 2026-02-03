import asyncio
import logging
import json
import re
import time
from datetime import datetime
from functools import wraps
from sqlalchemy import and_, select, desc, delete, text, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from typing import Dict, List

from src.data.provider import DataProvider
from src.config import DATABASE_URL, FALLBACK_CRYPTO, FALLBACK_FOREX, LOSS_COOLDOWN_DURATION

logger = logging.getLogger(__name__)


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
    pnl_zar: Mapped[float] = mapped_column()
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
    net_pnl_zar: Mapped[float] = mapped_column()


class UserSettings(Base):
    __tablename__ = "user_settings"
    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    login: Mapped[str] = mapped_column(nullable=True)
    server: Mapped[str] = mapped_column(nullable=True)
    password: Mapped[str] = mapped_column(nullable=True)
    lot_size: Mapped[float] = mapped_column(default=0.1)
    risk: Mapped[float] = mapped_column(default=2.0)
    confidence: Mapped[int] = mapped_column(default=75)
    high_vol: Mapped[bool] = mapped_column(default=False)


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
                # 1. Clean old trade results (30 days)
                cutoff = time.time() - (86400 * 30)
                stmt_trades = delete(TradeResult).where(TradeResult.timestamp < cutoff)
                await session.execute(stmt_trades)

                # 2. Clean empty sessions (0 trades)
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

    async def get_alltime_trade_history(self, provider: DataProvider, filters=None) -> Dict:
        """
        Returns all time trading history with custom filter options
        """
        if not self.engine:
            return {}

        page = 1
        limit = 10
        if filters:
            page = filters.get("page", 1)
            limit = filters.get("limit", 10)

        try:
            async with self.async_session() as session:
                # 1. Base Query
                query = select(TradeResult).order_by(TradeResult.timestamp.desc())

                # 2. Apply Filters
                conditions = []
                if filters:
                    if filters.get("startDate"):
                        try:
                            start_ts = datetime.strptime(filters["startDate"], "%Y-%m-%d").timestamp()
                            conditions.append(TradeResult.timestamp >= start_ts)
                        except:
                            pass
                    if filters.get("endDate"):
                        try:
                            end_ts = datetime.strptime(filters["endDate"], "%Y-%m-%d").timestamp() + 86400
                            conditions.append(TradeResult.timestamp <= end_ts)
                        except:
                            pass
                    if filters.get("range"):
                        now = time.time()
                        if filters["range"] == "24H":
                            conditions.append(TradeResult.timestamp >= now - 86400)
                        if filters["range"] == "7D":
                            conditions.append(TradeResult.timestamp >= now - 604800)
                        if filters["range"] == "30D":
                            conditions.append(TradeResult.timestamp >= now - 2592000)

                    outcome = filters.get("outcome", "ALL")
                    if outcome == "WINS":
                        conditions.append(TradeResult.result == 1)
                    elif outcome == "LOSSES":
                        conditions.append(TradeResult.result == 0)

                    assets = filters.get("assets", [])
                    dynamic_symbols = await provider.get_dynamic_symbols()

                    crypto_symbols = dynamic_symbols.get("crypto", FALLBACK_CRYPTO)
                    forex_symbols = dynamic_symbols.get("crypto", FALLBACK_FOREX)

                    if assets and "ALL" not in assets:
                        symbol_conditions = []
                        if "CRYPTO" in assets:
                            symbol_conditions.extend(crypto_symbols)
                        if "FOREX" in assets:
                            symbol_conditions.extend(forex_symbols)
                        if symbol_conditions:
                            conditions.append(TradeResult.symbol.in_(symbol_conditions))

                if conditions:
                    query = query.where(and_(*conditions))

                # 3. Get Total Count (for Pagination)
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(count_query)
            total_records = total_result.scalar()

            # 4. Apply Pagination
            query = query.offset((page - 1) * limit).limit(limit)

            result = await session.execute(query)
            trades = result.scalars().all()

            # 5. Calculate Lifetime Stats (Optimized: Single aggregated query)
            stats_query = select(
                func.count(TradeResult.id), func.sum(TradeResult.pnl_zar), func.sum(TradeResult.result)
            )
            stats_res = await session.execute(stats_query)
            total_trades, lifetime_pnl, total_wins = stats_res.one()

            return {
                "trades": trades,
                "page": page,
                "limit": limit,
                "total_records": total_records,
                "total_trades": total_trades,
                "lifetime_pnl": lifetime_pnl,
                "total_wins": total_wins,
            }
        except Exception as e:
            logger.error(f"Failed to fetch trade history from DB: {e}")
            return {}

    async def get_dashboard_chart_data(self) -> List:
        """Returns chart data for GUI dashboard chart"""
        if not self.engine:
            return []

        try:
            async with self.async_session() as session:
                stmt = select(TradeResult).order_by(TradeResult.timestamp.asc())
                result = await session.execute(stmt)
                all_trades = result.scalars().all()

                return all_trades
        except Exception as e:
            logger.error(f"Failed to fetch chart data from DB: {e}")
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
    async def get_settings(self) -> Dict:
        """Fetches user settings from DB"""
        if not self.engine:
            return {}

        try:
            async with self.async_session() as session:
                stmt = select(UserSettings).where(UserSettings.id == 1)
                result = await session.execute(stmt)
                settings = result.scalars().first()
                if settings:
                    return {
                        "login": settings.login,
                        "server": settings.server,
                        "password": settings.password,
                        "lot_size": settings.lot_size,
                        "risk": settings.risk,
                        "confidence": settings.confidence,
                        "high_vol": settings.high_vol,
                    }
                return {}
        except Exception as e:
            logger.error(f"Failed to load settings from DB: {e}")
            return {}

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

            # --- AUTO-MIGRATION: Check for missing columns ---
            async with self.async_session() as session:
                try:
                    await session.execute(text("SELECT size FROM trade_results LIMIT 1"))
                except Exception:
                    logger.info("🔧 Migrating DB: Adding 'size' column...")
                    await session.rollback()  # Rollback the failed select
                    async with self.engine.begin() as conn:
                        await conn.execute(text("ALTER TABLE trade_results ADD COLUMN size FLOAT DEFAULT 0.01"))

            logger.info("✅ DB Connected")
            return
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
                    net_pnl_zar=stats["pnl_zar"],
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
                    pnl_zar=trade_data["pnl"],
                    strategy=trade_data["strategy"],
                    size=trade_data.get("size", 0.01),
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
            async with self.async_session() as session:
                stmt = pg_insert(ActiveTrade).values(
                    symbol=symbol, signal_json=json.dumps(signal), start_time=time.time()
                )

                # If symbol exists, update the signal info and time
                do_update_stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol"], set_=dict(signal_json=json.dumps(signal), start_time=time.time())
                )

                await session.execute(do_update_stmt)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save active trade {symbol}: {e}")

    async def save_settings(self, data: dict):
        """Upserts user settings"""
        if not self.engine:
            return

        try:
            async with self.async_session() as session:
                stmt = select(UserSettings).where(UserSettings.id == 1)
                result = await session.execute(stmt)
                obj = result.scalars().first()

                if not obj:
                    obj = UserSettings(id=1)
                    session.add(obj)

                for key in ["login", "password", "server"]:
                    if key in data:
                        setattr(obj, key, data[key])
                if "lot_size" in data:
                    obj.lot_size = float(data["lot_size"])
                if "risk" in data:
                    obj.risk = float(data["risk"])
                if "confidence" in data:
                    obj.confidence = int(data["confidence"])
                if "high_vol" in data:
                    obj.high_vol = bool(data["high_vol"])

                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
