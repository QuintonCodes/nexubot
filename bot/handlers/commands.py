"""
Telegram Bot Command Handlers.
Routes user commands to the correct database queries or engine tasks.
"""

import pandas as pd
from aiogram import Router, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
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
        "<b>Available Commands:</b>\n"
        "/start - Welcome and initialization message\n"
        "/status - View engine health, configuration, and API budget\n"
        "/bias - Check current HTF market trend direction\n"
        "/session - View active ICT Killzones (SAST)\n"
        "/killzone - View active ICT Killzones (SAST)\n"
        "/levels - Check New Day and New Week opening reference targets\n"
        "/signals - View the last 5 dispatched trade signals\n"
        "/subscribe - Upgrade to the VIP Signals channel\n\n"
        "🛡️ <b>Admin Commands:</b>\n"
        "/zones - Show all active unmitigated Order Blocks\n"
        "/scan - Force a manual LTF signal evaluation scan"
    )


@router.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message):
    """Public: Routes users to the direct Whop checkout for VIP Signals."""
    checkout_url = "https://whop.com/checkout/plan_vUzndOAQpUt6Z"
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="💳 Upgrade to VIP (ZAR 500/mo)", url=checkout_url)]]
    )

    # Instructional copy matching the Whop customer hub UI
    await message.reply(
        "🔐 <b>Unlock Nexubot VIP Signals</b>\n\n"
        "Get direct access to our premium automated Smart Money Concepts (SMC) "
        "trading signals for XAUUSD.\n\n"
        "<b>How it works:</b>\n"
        "1. Tap the button below to pay securely via Whop (FICA compliant).\n"
        "2. After checkout, you will be redirected to your Nexubot Systems hub.\n"
        "3. Click the <b>Telegram</b> tab on the left sidebar to claim your invite link.\n\n"
        "<i>Subscription is managed automatically by the Whop Bot.</i>",
        reply_markup=keyboard,
    )


@router.message(Command("status"))
async def cmd_status(message: types.Message):
    """System health and config status (API Budgets restricted to Admins)."""
    is_admin = message.from_user.id == settings.TELEGRAM_ADMIN_ID
    usage = await rate_limiter.get_usage()

    text = (
        f"🟢 <b>Nexubot Status: ONLINE</b>\n\n"
        f"<b>Asset:</b> {settings.SYMBOLS[0]}\n"
        f"<b>Entry TF:</b> {settings.ENTRY_TIMEFRAME}\n"
        f"<b>HTF Bias TF:</b> {settings.HTF_TIMEFRAMES[0]}\n"
        f"<b>Shadow Mode:</b> {'Enabled 🔕' if settings.SHADOW_MODE else 'Disabled 🔔'}\n"
    )

    if is_admin:
        text += (
            f"\n📊 <b>API Budget (Twelve Data):</b>\n"
            f"<b>Calls Used:</b> {usage['current_calls']} / {usage['max_daily_calls']}\n"
            f"<b>Remaining:</b> {usage['remaining_calls']} calls"
        )

    await message.reply(text)


@router.message(Command("bias"))
async def cmd_bias(message: types.Message):
    """Public: Current higher timeframe trend direction."""
    symbol = settings.SYMBOLS[0]
    timeframes = [
        settings.HTF_TIMEFRAMES[0],
        settings.HTF_TIMEFRAMES[1],
        settings.LTF_TIMEFRAMES[0],
        settings.ENTRY_TIMEFRAME,
    ]

    text = f"🧭 <b>Multi-Timeframe Bias Matrix</b>\n\n<b>Asset:</b> {symbol}\n\n"

    for tf in timeframes:
        bias = await structure_events.get_latest_bias(symbol, tf)

        if not bias:
            df = await candle_store.get_candles(symbol, tf)
            if not df.empty:
                swings = detect_swings(df)
                bias = classify_structure(df, swings)
            else:
                bias = "ranging"

        bias_str = bias.upper() if bias else "RANGING"
        emoji = "🐂" if bias == "bullish" else "🐻" if bias == "bearish" else "⚖️"
        text += f"<b>{tf.upper():<5}</b>: {emoji} {bias_str}\n"

    await message.reply(text)


@router.message(Command("session"))
@router.message(Command("killzone"))
async def cmd_session(message: types.Message):
    """Exposes Session Killzone tracking."""
    now_utc = datetime.now(timezone.utc)
    active = SessionManager.get_active_killzone(now_utc)
    now_sast = now_utc.astimezone(ZoneInfo("Africa/Johannesburg"))
    symbol = settings.SYMBOLS[0]

    msg = (
        f"🕒 <b>SMC Killzone Tracker</b>\n\n"
        f"<b>Current Time (SAST):</b> {now_sast.strftime('%H:%M')}\n"
        f"<b>Current Time (UTC):</b> {now_utc.strftime('%H:%M')}\n"
        f"<b>Active Zone:</b> {active}\n"
    )

    # Strictly show Killzone Highs and Lows ONLY when currently inside an active Killzone
    if active in SessionManager.KILLZONES:
        df = await candle_store.get_candles(symbol, settings.ENTRY_TIMEFRAME)
        if df.empty or "timestamp" not in df.columns:
            df = await candle_store.get_candles(symbol, "15min")

        kz_levels = SessionManager.get_current_session_range(df, now_utc)

        if kz_levels:
            msg += (
                f"\n🎯 <b>{active} Liquidity Extremes ({symbol}):</b>\n"
                f"<b>Session High:</b> {kz_levels['high']:,.2f}\n"
                f"<b>Session Low:</b> {kz_levels['low']:,.2f}\n"
                f"<b>Session Range:</b> {abs(kz_levels['high'] - kz_levels['low']):,.2f} pts\n"
                f"<b>Current Price:</b> {kz_levels['last']:,.2f}\n"
            )
        else:
            msg += "\n🎯 <b>Session Range:</b> Accumulating initial candles...\n"
    elif active == "Market Closed":
        msg += "\n🔒 <i>Market is closed for the weekend (Friday 22:00 UTC – Sunday 22:00 UTC).</i>"
    else:
        msg += "\n💤 <i>Currently out of session. Session High/Low tracking is inactive.</i>"

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

    # Map raw database statuses to cleaner, user-friendly labels
    status_map = {
        "active": "⏳ Active",
        "tp1_hit": "✅ TP1 Hit",
        "tp2_hit": "✅ TP2 Hit",
        "tp3_hit": "🏆 TP3 Hit",
        "sl_hit": "🛑 SL Hit",
    }

    text = f"📊 <b>Recent Signals ({symbol}):</b>\n\n"
    for sig in recent:
        dir_emoji = "🟢 BUY" if sig["direction"] == "buy" else "🔴 SELL"
        raw_status = sig.get("status", "active").lower()
        status_display = status_map.get(raw_status, raw_status.upper())

        text += f"<b>{dir_emoji}</b> @ {sig['entry_price']:,.2f} | <b>{status_display}</b>\n"
        text += f"Model: {sig['entry_model']} (Score: {sig['confluence_score']})\n"
        text += f"⏰ {sig['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    await message.reply(text)
