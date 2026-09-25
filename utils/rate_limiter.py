"""
API Rate Limiter for Twelve Data REST API.
Tracks daily REST requests, resets at UTC midnight, and provides safety margins.
Persists limits to PostgreSQL to maintain state across process restarts.
"""

from datetime import datetime, timezone
from typing import Dict, Any

from config.settings import settings
from utils.logger import logger
from db.database import get_pool


class RateLimiter:
    def __init__(self, max_daily_calls: int = settings.MAX_DAILY_API_CALLS, warning_threshold: int = 750):
        self.max_daily_calls = max_daily_calls
        self.warning_threshold = warning_threshold
        self._db_initialized = False

    async def _init_db(self) -> None:
        """Bootstraps the single-row rate limiter table if it doesn't exist."""
        if self._db_initialized:
            return

        pool = get_pool()
        async with pool.acquire() as conn:
            # Create a dedicated table with a CHECK constraint to enforce a single row
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INT PRIMARY KEY DEFAULT 1,
                    current_calls INTEGER NOT NULL DEFAULT 0,
                    last_reset_date DATE NOT NULL,
                    CHECK (id = 1)
                );
            """)

            # Seed the initial row if the table is completely empty
            now_date = datetime.now(timezone.utc).date()
            await conn.execute(
                """
                INSERT INTO api_usage (id, current_calls, last_reset_date)
                VALUES (1, 0, $1)
                ON CONFLICT DO NOTHING;
            """,
                now_date,
            )

        self._db_initialized = True

    async def _check_and_reset(self) -> None:
        """Resets the counter in the database if UTC midnight has passed."""
        await self._init_db()
        now_date = datetime.now(timezone.utc).date()

        pool = get_pool()
        async with pool.acquire() as conn:
            # Atomic check-and-update prevents race conditions across instances
            result = await conn.execute(
                "UPDATE api_usage SET current_calls = 0, last_reset_date = $1 WHERE id = 1 AND last_reset_date < $1;",
                now_date,
            )

            # execute() returns a command tag like 'UPDATE 1' if a row was actually modified
            if result == "UPDATE 1":
                logger.info("rate_limiter_daily_reset", new_date=str(now_date))

    async def acquire(self) -> bool:
        """
        Atomically checks if a request can be made and increments the counter in a single SQL operation.
        Raises ConnectionRefusedError if the daily budget is exhausted.
        """
        await self._check_and_reset()

        pool = get_pool()
        async with pool.acquire() as conn:
            # Atomic increment directly at the database level eliminates read-modify-write race conditions
            new_calls = await conn.fetchval(
                "UPDATE api_usage SET current_calls = current_calls + 1 WHERE id = 1 RETURNING current_calls;"
            )

            if new_calls is None:
                new_calls = 1

            # If the atomic bump exceeded the limit, roll it back to maintain accurate metrics and block the request
            if new_calls > self.max_daily_calls:
                await conn.execute("UPDATE api_usage SET current_calls = current_calls - 1 WHERE id = 1;")

                logger.error(
                    "rate_limit_exceeded",
                    current_calls=new_calls - 1,
                    max_calls=self.max_daily_calls,
                )
                raise ConnectionRefusedError(
                    f"Twelve Data daily REST API budget reached ({new_calls - 1}/{self.max_daily_calls})."
                )

            # Trigger warning strictly once upon crossing the threshold
            if new_calls == self.warning_threshold:
                logger.warning(
                    "rate_limit_approaching",
                    current_calls=new_calls,
                    remaining_calls=self.max_daily_calls - new_calls,
                )

            return True

    async def get_usage(self) -> Dict[str, Any]:
        """Returns current rate limit metrics directly from the database."""
        await self._check_and_reset()

        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT current_calls, last_reset_date FROM api_usage WHERE id = 1;")

            current_calls = row["current_calls"] if row else 0
            last_reset_date = row["last_reset_date"] if row else datetime.now(timezone.utc).date()

        return {
            "current_calls": current_calls,
            "max_daily_calls": self.max_daily_calls,
            "remaining_calls": max(0, self.max_daily_calls - current_calls),
            "last_reset_date": str(last_reset_date),
        }

    async def force_reset(self) -> None:
        """Forces a manual reset of the daily counter in the database."""
        await self._init_db()
        now_date = datetime.now(timezone.utc).date()

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE api_usage SET current_calls = 0, last_reset_date = $1 WHERE id = 1;", now_date)

        logger.info("rate_limiter_manually_reset")


# Singleton instance for centralized rate tracking
rate_limiter = RateLimiter()
