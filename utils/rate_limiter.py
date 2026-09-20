"""
API Rate Limiter for Twelve Data REST API.
Tracks daily REST requests, resets at UTC midnight, and provides safety margins.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

from config.settings import settings
from utils.logger import logger


class RateLimiter:
    def __init__(self, max_daily_calls: int = settings.MAX_DAILY_API_CALLS, warning_threshold: int = 750):
        self.max_daily_calls = max_daily_calls
        self.warning_threshold = warning_threshold
        self.current_calls = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self._lock = asyncio.Lock()

    async def _check_and_reset(self) -> None:
        """Resets the counter if UTC midnight has passed."""
        now_date = datetime.now(timezone.utc).date()
        if now_date > self.last_reset_date:
            logger.info(
                "rate_limiter_daily_reset",
                previous_calls=self.current_calls,
                new_date=str(now_date),
            )
            self.current_calls = 0
            self.last_reset_date = now_date

    async def acquire(self) -> bool:
        """
        Checks if a request can be made and increments the counter.
        Raises ConnectionRefusedError if the daily budget is exhausted.
        """
        async with self._lock:
            await self._check_and_reset()

            if self.current_calls >= self.max_daily_calls:
                logger.error(
                    "rate_limit_exceeded",
                    current_calls=self.current_calls,
                    max_calls=self.max_daily_calls,
                )
                raise ConnectionRefusedError(
                    f"Twelve Data daily REST API budget reached ({self.current_calls}/{self.max_daily_calls})."
                )

            self.current_calls += 1

            if self.current_calls >= self.warning_threshold:
                logger.warning(
                    "rate_limit_approaching",
                    current_calls=self.current_calls,
                    remaining_calls=self.max_daily_calls - self.current_calls,
                )

            return True

    async def get_usage(self) -> Dict[str, Any]:
        """Returns current rate limit metrics."""
        async with self._lock:
            await self._check_and_reset()
            return {
                "current_calls": self.current_calls,
                "max_daily_calls": self.max_daily_calls,
                "remaining_calls": max(0, self.max_daily_calls - self.current_calls),
                "last_reset_date": str(self.last_reset_date),
            }

    async def force_reset(self) -> None:
        """Forces a manual reset of the daily counter."""
        async with self._lock:
            self.current_calls = 0
            self.last_reset_date = datetime.now(timezone.utc).date()
            logger.info("rate_limiter_manually_reset")


# Singleton instance for centralized rate tracking
rate_limiter = RateLimiter()
