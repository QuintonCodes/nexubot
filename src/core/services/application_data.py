import asyncio
import logging
import math
import time
from datetime import datetime, timedelta

from src.utils.logger import read_recent_logs

logger = logging.getLogger(__name__)


class ApplicationData:
    def __init__(self, engine):
        self.engine = engine

    async def get_dashboard_data(self):
        """
        Aggregates data for the dashboard:
        1. Live MT5 Balance/Equity
        2. Database Stats (Win Rate, Total PnL)
        3. Chart Data (Equity Curve simulation)
        """
        safe_response = {
            "balance": 0.0,
            "equity": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "chart_labels": [],
            "chart_data": [],
            "mode": self.engine.execution_mode,
            "system_status": self.engine.system_status,
            "recent_trades": [],
            "timestamp": time.time(),
        }

        try:
            # 1. Live Account Info
            acct = await self.engine.provider.get_account_summary()
            if acct:
                safe_response["balance"] = acct.get("balance", 0.0)
                safe_response["equity"] = acct.get("equity", 0.0)

            # 2. Fetch Stats & Recent Trades (Limit 20 for chart reconstruction)
            stats, recent_trades = await asyncio.gather(
                self.engine.db.get_dashboard_stats(), self.engine.db.get_recent_trades(20)
            )

            # 3. Process Stats
            total_trades = stats["wins"] + stats["losses"]
            win_rate = (stats["wins"] / total_trades * 100) if total_trades > 0 else 0.0
            total_pnl = stats["total_pnl"]

            # 4. Smart Chart Reconstruction (Backwards from Total PnL)
            # This avoids fetching thousands of rows for the equity curve
            chart_points = []

            # Start with current Total PnL as the last point
            current_curve_val = total_pnl

            # Iterate backwards through recent trades to build the curve history
            # Sort trades by time desc (newest first)
            sorted_trades = sorted(recent_trades, key=lambda x: x.timestamp, reverse=True)

            for t in sorted_trades:
                chart_points.append(
                    {
                        "label": datetime.fromtimestamp(t.timestamp).strftime("%d %b"),
                        "value": round(current_curve_val, 2),
                    }
                )
                # Subtract this trade's PnL to get the value BEFORE this trade happened
                current_curve_val -= t.pnl_zar

            # Reverse back to chronological order for the chart (Oldest -> Newest)
            chart_points.reverse()

            chart_labels = [p["label"] for p in chart_points]
            chart_data = [p["value"] for p in chart_points]

            # 5. Process Recent Trades
            recent_trades_data = []
            for t in sorted_trades[:5]:
                recent_trades_data.append(
                    {
                        "time": datetime.fromtimestamp(t.timestamp).strftime("%H:%M:%S"),
                        "symbol": t.symbol,
                        "signal_type": t.signal_type,
                        "entry": float(t.entry_price),
                        "exit": float(t.exit_price),
                        "size": getattr(t, "size", 0.01),
                        "pnl": float(t.pnl_zar),
                        "result": int(t.result),
                    }
                )

            safe_response.update(
                {
                    "total_pnl": total_pnl,
                    "win_rate": win_rate,
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "chart_labels": chart_labels,
                    "chart_data": chart_data,
                    "recent_trades": recent_trades_data,
                }
            )

            return safe_response
        except Exception as e:
            logger.error(f"Dashboard Data Error: {e}")
            return safe_response

    async def get_signal_page_data(self):
        """Aggregates all data needed for the Signal Page."""
        safe_response = {
            "account": {"balance": 0.0, "equity": 0.0},
            "stats": {
                "lifetime_wr": 0.0,
                "active_count": 0,
                "session_pnl": 0.0,
                "session_wins": 0,
                "session_losses": 0,
                "session_total": 0,
                "time_running": "0:00:00",
            },
            "signals": [],
            "logs": [],
            "mode": self.engine.execution_mode,
        }

        try:
            acct = await self.engine.provider.get_account_summary()
            if acct:
                safe_response["account"] = acct

            lifetime_wr = await self.engine.db.get_total_historical_win_rate()
            elapsed = timedelta(seconds=int(time.time() - self.engine.session_start))

            safe_response["stats"] = {
                "lifetime_wr": lifetime_wr,
                "active_count": len(self.engine.active_signals),
                "session_pnl": self.engine.session_stats["pnl"],
                "session_wins": self.engine.session_stats["wins"],
                "session_losses": self.engine.session_stats["losses"],
                "session_total": self.engine.session_stats["total"],
                "time_running": str(elapsed),
            }
            safe_response["signals"] = self.engine.active_signals
            safe_response["logs"] = read_recent_logs(30)

            return safe_response
        except Exception as e:
            logger.error(f"Signal Data Error: {e}")
            return safe_response

    async def get_trade_history(self, filters=None):
        """Fetches filtered trade history and global lifetime stats."""
        alltime_trades = await self.engine.db.get_alltime_trade_history(self.engine.provider, filters)
        if not alltime_trades:
            alltime_trades = {}

        stats = {
            "balance": self.engine.ai_engine.user_balance_account or 0.0,
            "lifetime_wr": 0.0,
            "total_trades": 0,
            "lifetime_pnl": 0.0,
        }

        total_trades = alltime_trades.get("total_trades", 0)
        lifetime_pnl = alltime_trades.get("lifetime_pnl", 0.0)
        total_wins = alltime_trades.get("total_wins", 0)
        lifetime_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        stats["total_trades"] = total_trades or 0
        stats["lifetime_pnl"] = lifetime_pnl or 0.0
        stats["lifetime_wr"] = lifetime_wr

        table_data = []
        raw_trades = alltime_trades.get("trades", [])
        if raw_trades:
            for t in raw_trades:
                table_data.append(
                    {
                        "time": datetime.fromtimestamp(t.timestamp).strftime("%Y-%m-%d %H:%M"),
                        "symbol": t.symbol,
                        "signal_type": t.signal_type,
                        "entry": float(t.entry_price),
                        "exit": float(t.exit_price),
                        "pnl": float(t.pnl_zar),
                        "result": int(t.result),
                        "confidence": float(t.confidence),
                        "size": getattr(t, "size", 0.01),
                    }
                )

        return {
            "stats": stats,
            "history": table_data,
            "pagination": {
                "current": alltime_trades.get("page", 1),
                "total_pages": (
                    math.ceil(total_trades / alltime_trades.get("limit", 10)) if alltime_trades.get("limit") else 1
                ),
                "total_records": total_trades,
            },
        }
