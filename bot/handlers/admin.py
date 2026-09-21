"""
Telegram Bot Admin Handlers.
Contains restricted commands for bot management and manual strategy overrides.
"""

from aiogram import Router, types
from aiogram.filters import Command, Filter
from aiogram.types import Message

from bot.formatters.signal_formatter import format_trade_signal
from config.settings import settings
from db.repositories import order_blocks
from strategies.confluence import ConfluenceEngine


class IsAdmin(Filter):
    """Custom filter to ensure only the authorized admin can trigger these commands."""

    async def __call__(self, message: Message) -> bool:
        return message.from_user.id == settings.TELEGRAM_ADMIN_ID


# Initialize a separate router for admin commands
router = Router()

# Apply the admin filter to all handlers in this router automatically
router.message.filter(IsAdmin())


@router.message(Command("zones"))
async def cmd_zones(message: types.Message):
    """Admin: Show all currently active, unmitigated Order Blocks."""
    symbol = settings.SYMBOLS[0]
    mtf = settings.HTF_TIMEFRAMES[1]
    active_obs = await order_blocks.get_active_order_blocks(symbol, mtf)

    if not active_obs:
        await message.reply(f"No active unmitigated Order Blocks for {symbol} ({mtf}).")
        return

    text = f"🧱 <b>Active {mtf} Order Blocks ({symbol}):</b>\n\n"
    for ob in active_obs:
        dir_emoji = "🟢" if ob["direction"] == "bullish" else "🔴"
        text += f"{dir_emoji} <b>{ob["direction"].upper()}</b>\n"
        text += f"Range: {min(ob["ob_low"], ob["ob_high"]):.2f} - {max(ob["ob_low"], ob["ob_high"]):.2f}\n"
        text += f"Mitigation (50%): {ob["ob_50"]:.2f}\n\n"

    await message.reply(text)


@router.message(Command("scan"))
async def cmd_scan(message: types.Message):
    """Admin: Manually forces a signal evaluation for the current market structure."""
    symbol = settings.SYMBOLS[0]
    await message.reply(f"🔍 Initiating manual LTF scan for {symbol}...")

    engine = ConfluenceEngine(symbol)

    try:
        signal = await engine.scan_ltf_entry()
        if signal:
            await message.reply(f"⚠️ <b>Scan found a valid setup!</b>\n\n" + format_trade_signal(signal))
        else:
            await message.reply("📉 Scan complete. No active entry setups found right now.")
    except Exception as e:
        await message.reply(f"❌ Scan failed: {str(e)}")
