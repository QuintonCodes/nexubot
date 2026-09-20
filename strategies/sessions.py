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
        "Asia Range": (0, 4),
        "London Open Killzone": (7, 9),
        "London Close": (11, 12),
        "NY AM Killzone": (13, 15),
        "ICT Silver Bullet (PM)": (14, 15),
        "NY PM / London Close": (17, 20),
    }

    @staticmethod
    def get_active_killzone(dt_utc: datetime) -> str:
        """Returns the active session killzone name based on the current UTC hour."""
        for name, (start, end) in SessionManager.KILLZONES.items():
            if start <= dt_utc.hour < end:
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
