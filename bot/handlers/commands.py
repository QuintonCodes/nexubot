"""
Telegram Bot Command Handlers.
Routes user commands to the correct database queries or engine tasks.
"""

from aiogram import Router, types
from aiogram.filters import Command

from config.settings import settings
from db.repositories import signals, structure_events

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
    await message.reply(
        f"🟢 <b>Nexubot Status: ONLINE</b>\n\n"
        f"<b>Asset:</b> {settings.SYMBOLS[0]}\n"
        f"<b>Entry TF:</b> {settings.ENTRY_TIMEFRAME}\n"
        f"<b>HTF Bias TF:</b> {settings.HTF_TIMEFRAMES[0]}\n"
        f"<b>Shadow Mode:</b> {'Enabled 🔕' if settings.SHADOW_MODE else 'Disabled 🔔'}\n"
        f"<b>API Limit:</b> {settings.MAX_DAILY_API_CALLS} calls/day"
    )


@router.message(Command("bias"))
async def cmd_bias(message: types.Message):
    """Public: Current higher timeframe trend direction."""
    symbol = settings.SYMBOLS[0]
    htf = settings.HTF_TIMEFRAMES[0]
    bias = await structure_events.get_latest_bias(symbol, htf)

    bias_str = bias.upper() if bias else "UNKNOWN"
    emoji = "🐂" if bias == "bullish" else "🐻" if bias == "bearish" else "⚖️"

    await message.reply(f"🧭 <b>Current HTF Bias ({htf})</b>\n\n{symbol}: {emoji} <b>{bias_str}</b>")


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
        dir_emoji = "🟢 BUY" if sig.direction == "buy" else "🔴 SELL"
        text += f"<b>{dir_emoji}</b> @ {sig.entry_price:,.2f} (Score: {sig.confluence_score})\n"
        text += f"⏰ {sig.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    await message.reply(text)
