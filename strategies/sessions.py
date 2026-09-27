"""
Session Killzone and Reference Level Manager.
Tracks high-probability liquidity windows and daily/weekly opening gaps.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional


class SessionManager:
    # Based on standard UTC time mappings for ICT Killzones
    KILLZONES = {
        "Asia Killzone": (0, 4),  # 00:00 to 04:00 UTC
        "London Killzone": (6, 9),  # 06:00 to 09:00 UTC
        "NY AM Killzone": (13.5, 15),  # 13:30 to 15:00 UTC
        "NY Lunch Killzone": (16, 17),  # 16:00 to 17:00 UTC
        "NY PM Killzone": (17.5, 20),  # 17:30 to 20:00 UTC
    }

    @staticmethod
    def is_market_open(current_time: datetime) -> bool:
        """Evaluates if the current UTC time falls within active trading hours (24/5)."""
        weekday: int = current_time.weekday()
        hour: int = current_time.hour

        # Friday after 22:00 UTC
        if weekday == 4 and hour >= 22:
            return False
        # Saturday all day
        if weekday == 5:
            return False
        # Sunday before 22:00 UTC
        if weekday == 6 and hour < 22:
            return False

        return True

    @staticmethod
    def get_active_killzone(dt_utc: datetime) -> str:
        """
        Returns the active session killzone name based on the current UTC hour.
        """
        if not SessionManager.is_market_open(dt_utc):
            return "Market Closed"

        decimal_hour = dt_utc.hour + dt_utc.minute / 60

        for name, (start, end) in SessionManager.KILLZONES.items():
            if start <= decimal_hour < end:
                return name

        return "Out of Session"

    @staticmethod
    def get_current_session_range(df: pd.DataFrame, dt_utc: datetime) -> Optional[Dict[str, float]]:
        """
        Calculates the current session's high and low if within an active killzone.
        Returns None if out of session, market closed, or if no candles exist for the active window.
        """
        active = SessionManager.get_active_killzone(dt_utc)
        if active not in SessionManager.KILLZONES or df.empty or "timestamp" not in df.columns:
            return None

        start, end = SessionManager.KILLZONES[active]
        ts_series = pd.to_datetime(df["timestamp"])
        today_utc = dt_utc.date()
        decimal_hour = ts_series.dt.hour + ts_series.dt.minute / 60.0

        mask = (ts_series.dt.date == today_utc) & (decimal_hour >= start) & (decimal_hour < end)
        kz_candles = df[mask]

        if kz_candles.empty:
            return None

        return {
            "session": active,
            "high": float(kz_candles["high"].max()),
            "low": float(kz_candles["low"].min()),
            "open": float(kz_candles.iloc[0]["open"]),
            "last": float(kz_candles.iloc[-1]["close"]),
        }

    @staticmethod
    def get_daily_weekly_open(df: pd.DataFrame) -> Dict[str, float]:
        """Calculates New Day Opening (NDO) and New Week Opening (NWO) prices."""
        if df.empty or "timestamp" not in df.columns:
            return {"NDO": 0.0, "NWO": 0.0}

        temp_df = df.set_index("timestamp")

        try:
            daily_opens = temp_df["open"].resample("D").first().dropna()
            ndo = float(daily_opens.iloc[-1]) if not daily_opens.empty else 0.0

            # 'W-MON' roots the week to start on Monday for traditional forex market hours
            weekly_opens = temp_df["open"].resample("W-MON").first().dropna()
            nwo = float(weekly_opens.iloc[-1]) if not weekly_opens.empty else 0.0
        except Exception:
            ndo, nwo = 0.0, 0.0

        return {"NDO": ndo, "NWO": nwo}
