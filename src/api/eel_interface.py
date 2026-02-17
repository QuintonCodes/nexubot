import asyncio
import eel
import logging
import os
import sys

from src.core.engine import NexubotEngine
from src.utils.concurrency import get_persistent_loop, safe_get_result
from src.config import VERSION

logger = logging.getLogger(__name__)

bot_instance = None


@eel.expose
def attempt_login(login_id, server, password):
    """Called from login.html when user clicks 'Initialize Connection'"""
    global bot_instance
    if not bot_instance:
        bot_instance = NexubotEngine()

    loop = get_persistent_loop()
    future = asyncio.run_coroutine_threadsafe(bot_instance.initialize_connection(login_id, server, password), loop)
    return safe_get_result(future, timeout=45.0)


@eel.expose
def close_app():
    """Cleanup when window closes"""
    sys.exit(0)


@eel.expose
def fetch_dashboard_update():
    """Called by JS interval to get live data"""
    global bot_instance
    default_data = {
        "balance": 0.0,
        "equity": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "wins": 0,
        "losses": 0,
        "recent_trades": [],
        "chart_labels": [],
        "chart_data": [],
        "mode": "SIGNAL_ONLY",
        "system_status": "IDLE",
    }
    if not bot_instance:
        return default_data

    loop = get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.data_service.get_dashboard_data(), loop)
        res = safe_get_result(future, timeout=3.0)

        final_data = res if res else default_data.copy()

        if bot_instance and bot_instance.provider:
            final_data["latency"] = bot_instance.get_latency()

        return final_data
    except Exception as e:
        logger.error(f"Dashboard Update Failed: {e}")
        return default_data


@eel.expose
def fetch_signal_updates():
    """Polled by signal.html"""
    global bot_instance
    default_data = {
        "account": {"balance": 0, "equity": 0},
        "stats": {
            "lifetime_wr": 0,
            "active_count": 0,
            "session_pnl": 0,
            "session_wins": 0,
            "session_losses": 0,
            "session_total": 0,
            "time_running": "--",
        },
        "signals": [],
        "logs": [],
        "mode": "SIGNAL_ONLY",
    }
    if not bot_instance:
        return default_data

    loop = get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.data_service.get_signal_page_data(), loop)
        res = safe_get_result(future, timeout=3.0)

        final_data = res if res else default_data.copy()

        if bot_instance and bot_instance.provider:
            final_data["latency"] = bot_instance.get_latency()

        return final_data
    except Exception as e:
        logger.error(f"Signal Update Failed: {e}")
        return default_data


@eel.expose
def fetch_trade_history(filters=None):
    """
    Fetches filtered trade history and global lifetime stats.
    """
    global bot_instance
    default_res = {
        "stats": {"balance": 0.0, "lifetime_wr": 0.0, "total_trades": 0, "lifetime_pnl": 0.0},
        "history": [],
        "pagination": {"current": 1, "total_pages": 1, "total_records": 0},
    }
    if not bot_instance:
        return default_res

    loop = get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.data_service.get_trade_history(filters), loop)
        res = safe_get_result(future, timeout=10.0)

        final_data = res if res else default_res.copy()

        if bot_instance and bot_instance.provider:
            final_data["latency"] = bot_instance.get_latency()

        return final_data
    except Exception as e:
        logger.error(f"History Fetch Error: {e}")
        return default_res


@eel.expose
def force_close(symbol):
    global bot_instance
    if bot_instance:
        loop = get_persistent_loop()
        asyncio.run_coroutine_threadsafe(bot_instance.monitor.force_close_trade(symbol), loop)


@eel.expose
def get_user_settings():
    """Returns current settings to populate the Settings Page."""
    global bot_instance
    if not bot_instance:
        bot_instance = NexubotEngine()

    loop = get_persistent_loop()

    async def _safe_init():
        await bot_instance.initialize_settings()
        bot_instance.settings["latency"] = bot_instance.get_latency()
        bot_instance.settings["neural_meta"] = {
            "model": f"Transformer-XL {VERSION}",
            "epochs": "50,000",
            "bias": "Conservative" if bot_instance.ai_engine.min_confidence > 80 else "Balanced",
        }
        return bot_instance.settings

    future = asyncio.run_coroutine_threadsafe(_safe_init(), loop)
    return safe_get_result(future, timeout=5.0) or {}


@eel.expose
def save_settings(data):
    """Saves settings and restarts the engine."""
    global bot_instance
    if not bot_instance:
        return False

    loop = get_persistent_loop()

    async def _save_and_restart():
        await bot_instance.db.save_settings(data)
        await bot_instance.restart_system()
        return True

    future = asyncio.run_coroutine_threadsafe(_save_and_restart(), loop)
    return safe_get_result(future, timeout=10.0)


@eel.expose
def set_mode(is_auto):
    """Called by the toggle switch"""
    global bot_instance
    if bot_instance:
        mode = "FULL_AUTO" if is_auto else "SIGNAL_ONLY"
        bot_instance.set_execution_mode(mode)


@eel.expose
def shutdown_bot():
    """Stops the running bot instance gracefully."""
    global bot_instance
    if bot_instance:
        loop = get_persistent_loop()
        future = asyncio.run_coroutine_threadsafe(bot_instance.stop_session(), loop)
        safe_get_result(future, timeout=5.0)

    os._exit(0)


@eel.expose
def trigger_training(symbol=None):
    """Triggered from Frontend to start backfill/training."""
    global bot_instance
    if not bot_instance:
        return False
    loop = get_persistent_loop()
    asyncio.run_coroutine_threadsafe(bot_instance.trigger_manual_training(symbol), loop)
    return True
