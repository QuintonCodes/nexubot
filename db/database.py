"""
PostgreSQL Connection Pool Manager.
Maintains a singleton asyncpg pool connected to Neon Database.
"""

import asyncpg
from typing import Optional

from config.settings import settings
from utils.logger import logger

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> None:
    """Initializes the connection pool at startup."""
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=1,
            max_size=5,  # Neon free tier safe limits
            command_timeout=10.0,
            ssl="require",
        )
        logger.info("database_pool_initialized")
    except Exception as e:
        logger.error("database_connection_failed", error=str(e))
        raise


def get_pool() -> asyncpg.Pool:
    """Returns the active pool. Raises if called before initialization."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    """Drains connections on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("database_pool_closed")
