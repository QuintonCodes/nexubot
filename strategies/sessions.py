"""
Session Killzone and Reference Level Manager.
Tracks high-probability liquidity windows and daily/weekly opening gaps.
"""

import pandas as pd
from datetime import datetime
from typing import Dict


class SessionManager:
    # Based on standard UTC time mappings for ICT Killzones
    KILLZONES = {
        "Asia Killzone": (0, 4),  # 00:00 to 04:00 UTC
        "London Killzone": (6, 9),  # 06:00 to 09:00 UTC
        "NY AM Killzone": (13.5, 15),  # 13:30 to 15:00 UTC
        "NY Lunch Killzone": (16, 17),  # 16:00 to 17:00 UTC
        "NY PM / London Close": (17.5, 20),  # 17:30 to 20:00 UTC
    }

    @staticmethod
    def get_active_killzone(dt_utc: datetime) -> str:
        """
        Returns the active session killzone name based on the current UTC hour.
        Uses decimal hours (e.g. 13.5 = 13:30) to correctly handle
        sessions that start or end on the half-hour mark.
        """
        decimal_hour = dt_utc.hour + dt_utc.minute / 60

        for name, (start, end) in SessionManager.KILLZONES.items():
            if start <= decimal_hour < end:
                return name

        return "Out of Session"

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
