"""
Telegram Bot Admin Handlers.
Contains restricted commands for bot management and manual strategy overrides.
"""

from aiogram import Router, types
from aiogram.filters import Command, Filter
from aiogram.types import Message

from bot.formatters.signal_formatter import format_signal_for_admin
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
    """Admin: View active unmitigated Order Blocks across MTF/HTF."""
    symbol = settings.SYMBOLS[0]
    zones = await order_blocks.get_active_zones(symbol)

    if not zones:
        await message.reply(f"No active unmitigated Order Blocks for {symbol}.")
        return

    text = f"🛡️ <b>Active Order Blocks ({symbol})</b>\n\n"
    for z in zones[:15]:
        emoji = "🟢 Demand (Buy)" if z["direction"] == "bullish" else "🔴 Supply (Sell)"
        text += f"{emoji} | <b>{z['timeframe']}</b>\n"
        text += f"Range: {z['ob_low']:.2f} - {z['ob_high']:.2f}\n\n"

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
            await message.reply(f"⚠️ <b>Scan found a valid setup!</b>\n\n" + format_signal_for_admin(signal))
        else:
            await message.reply("📉 Scan complete. No active entry setups found right now.")
    except Exception as e:
        await message.reply(f"❌ Scan failed: {str(e)}")
