"""
Telegram Bot Command Handlers.
Routes user commands to the correct database queries or engine tasks.
"""

from aiogram import Router, types
from aiogram.filters import Command
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config.settings import settings
from data.candle_store import candle_store
from db.repositories import signals, structure_events
from strategies.sessions import SessionManager
from strategies.structure import classify_structure, detect_swings
from utils.rate_limiter import rate_limiter

router = Router()


@router.message(Command("start"))
async def cmd_start(message: types.Message):
    """Public: Welcome message."""
    await message.reply(
        "🤖 <b>Nexubot Cloud SMC Engine</b>\n\n"
        f"Monitoring {settings.SYMBOLS[0]} for Smart Money Concepts setups.\n"
        "Use /status to see current engine state."
    )


@router.message(Command("status"))
async def cmd_status(message: types.Message):
    """Public: System health and config status."""
    # Retrieve real-time usage metrics from the limiter singleton
    usage = await rate_limiter.get_usage()
    await message.reply(
        f"🟢 <b>Nexubot Status: ONLINE</b>\n\n"
        f"<b>Asset:</b> {settings.SYMBOLS[0]}\n"
        f"<b>Entry TF:</b> {settings.ENTRY_TIMEFRAME}\n"
        f"<b>HTF Bias TF:</b> {settings.HTF_TIMEFRAMES[0]}\n"
        f"<b>Shadow Mode:</b> {'Enabled 🔕' if settings.SHADOW_MODE else 'Disabled 🔔'}\n"
        f"📊 <b>API Budget (Twelve Data):</b>\n"
        f"<b>Calls Used:</b> {usage['current_calls']} / {usage['max_daily_calls']}\n"
        f"<b>Remaining:</b> {usage['remaining_calls']} calls"
    )


@router.message(Command("bias"))
async def cmd_bias(message: types.Message):
    """Public: Current higher timeframe trend direction."""
    symbol = settings.SYMBOLS[0]
    htf = settings.HTF_TIMEFRAMES[0]
    bias = await structure_events.get_latest_bias(symbol, htf)

    if not bias:
        df = await candle_store.get_candles(symbol, htf)
        swings = detect_swings(df)
        bias = classify_structure(df, swings)

    bias_str = bias.upper() if bias else "UNKNOWN"
    emoji = "🐂" if bias == "bullish" else "🐻" if bias == "bearish" else "⚖️"

    await message.reply(f"🧭 <b>Current HTF Bias ({htf})</b>\n\n{symbol}: {emoji} <b>{bias_str}</b>")


@router.message(Command("session"))
@router.message(Command("killzone"))
async def cmd_session(message: types.Message):
    """Exposes Session Killzone tracking."""
    now_utc = datetime.now(timezone.utc)
    active = SessionManager.get_active_killzone(now_utc)

    # Localize strictly for the Telegram output
    now_sast = now_utc.astimezone(ZoneInfo("Africa/Johannesburg"))

    msg = (
        f"🕒 <b>SMC Killzone Tracker (UTC)</b>\n\n"
        f"<b>Current Time (SAST):</b> {now_sast.strftime('%H:%M')}\n"
        f"<b>Active Zone:</b> {active if active != 'Out of Session' else 'None (Ranging expected)'}"
    )
    await message.reply(msg)


@router.message(Command("levels"))
async def cmd_levels(message: types.Message):
    """Exposes New Day/Week Opening reference targets."""
    symbol = settings.SYMBOLS[0]
    df = await candle_store.get_candles(symbol, settings.HTF_TIMEFRAMES[0])

    levels = SessionManager.get_daily_weekly_open(df)

    msg = (
        f"📍 <b>ICT Reference Levels ({symbol})</b>\n\n"
        f"<b>NDO (New Day Open):</b> {levels['NDO']:.2f}\n"
        f"<b>NWO (New Week Open):</b> {levels['NWO']:.2f}\n"
    )
    await message.reply(msg)


@router.message(Command("signals"))
async def cmd_signals(message: types.Message):
    """Public: Show the last 5 dispatched signals."""
    symbol = settings.SYMBOLS[0]
    recent = await signals.get_recent_signals(symbol, limit=5)

    if not recent:
        await message.reply("No recent signals found in the database.")
        return

    text = f"📊 <b>Last 5 Signals ({symbol}):</b>\n\n"
    for sig in recent:
        dir_emoji = "🟢 BUY" if sig["direction"] == "buy" else "🔴 SELL"
        text += f"<b>{dir_emoji}</b> @ {sig['entry_price']:,.2f} (Score: {sig['confluence_score']})\n"
        text += f"⏰ {sig['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    await message.reply(text)
