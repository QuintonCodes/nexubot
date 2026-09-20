"""
Application Settings and Environment Configuration.
Enforces type safety, startup validation, and specific constraints for XAUUSD cloud operation.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, List


class Settings(BaseSettings):
    # ==========================================
    # Telegram Credentials & Routing
    # ==========================================
    TELEGRAM_BOT_TOKEN: str = Field(..., description="Telegram bot token obtained from @BotFather.")
    TELEGRAM_CHANNEL_ID: int = Field(
        ..., description="Target Telegram channel ID (negative integer for supergroups/channels)."
    )
    TELEGRAM_ADMIN_ID: int = Field(
        ..., description="Telegram user ID of the primary administrator for command authorization."
    )

    # ==========================================
    # Twelve Data Configuration
    # ==========================================
    TWELVE_DATA_API_KEY: str = Field(..., description="Twelve Data API Key for REST fetches and WebSocket streaming.")
    MAX_DAILY_API_CALLS: int = Field(default=800, ge=1, description="Twelve Data free tier daily API limit.")

    # ==========================================
    # Neon PostgreSQL Persistence
    # ==========================================
    DATABASE_URL: str = Field(..., description="PostgreSQL direct connection string with asyncpg compatibility.")

    # ==========================================
    # Asset & Timeframe Scope (Locked to XAUUSD)
    # ==========================================
    SYMBOLS: str | List[str] = Field(
        default=["XAU/USD"], description="Operational asset list. Strictly constrained to Gold (XAUUSD)."
    )
    HTF_TIMEFRAMES: str | List[str] = Field(
        default=["4h", "1h"], description="Higher timeframes used for Macro bias and structure direction."
    )
    LTF_TIMEFRAMES: str | List[str] = Field(
        default=["15min", "5min"], description="Lower timeframes used for liquidity tracking and setup detection."
    )
    ENTRY_TIMEFRAME: str = Field(
        default="5min", description="Execution timeframe for the entry trigger (CISD / OTE tap)."
    )
    CANDLE_BUFFER_SIZE: int = Field(
        default=500, ge=100, le=2000, description="Number of rolling candles kept in in-memory deques per timeframe."
    )

    # ==========================================
    # Risk Management & SMC Calibration
    # ==========================================
    RISK_PERCENT: float = Field(
        default=1.0, gt=0.0, le=5.0, description="Risk percentage per setup recommended in signals."
    )
    MIN_CONFLUENCE_SCORE: int = Field(
        default=70, ge=0, le=100, description="Minimum confluence score (0-100) required to publish a Telegram signal."
    )
    SIGNAL_COOLDOWN_HOURS: int = Field(
        default=4, ge=1, description="Suppression window to prevent duplicate alerts within the same zone."
    )
    XAUUSD_PIP_TOLERANCE: float = Field(
        default=40.0,
        gt=0.0,
        description="Pip tolerance for Equal Highs (EQH) and Equal Lows (EQL) on XAUUSD (40 pips = $4.00).",
    )
    SHADOW_MODE: bool = Field(
        default=False, description="When True, evaluates signals and saves to DB without sending Telegram messages."
    )

    # ==========================================
    # Pydantic Settings Configuration
    # ==========================================
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # ==========================================
    # Validators
    # ==========================================
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validates that the database connection string is a PostgreSQL URI."""
        if not (v.startswith("postgresql://") or v.startswith("postgres://")):
            raise ValueError(
                "DATABASE_URL must be a valid PostgreSQL connection URI starting with "
                "'postgresql://' or 'postgres://'."
            )
        return v

    @field_validator("SYMBOLS", "HTF_TIMEFRAMES", "LTF_TIMEFRAMES", mode="before")
    @classmethod
    def parse_comma_separated_strings(cls, v: Any) -> List[str]:
        """Supports comma-separated environment variables from .env or platform dashboards."""
        if isinstance(v, str):
            return [
                item.strip().upper() if "USD" in item.upper() else item.strip() for item in v.split(",") if item.strip()
            ]
        if isinstance(v, list):
            return v
        raise ValueError(f"Invalid format for list configuration: {v}")

    @field_validator("SYMBOLS")
    @classmethod
    def enforce_xauusd_only(cls, v: List[str]) -> List[str]:
        """Locks the architecture to XAUUSD but formats it as XAU/USD for Twelve Data API."""
        # Standardize missing slashes so both "XAUUSD" and "XAU/USD" from .env work flawlessly
        sanitized = [sym.upper().replace("XAUUSD", "XAU/USD") for sym in v]
        if sanitized != ["XAU/USD"]:
            raise ValueError(
                f"Architecture constraint violation: Nexubot is locked to Gold. "
                f"Expected ['XAU/USD'], received {sanitized}."
            )
        return sanitized


# Global configuration singleton
settings = Settings()
