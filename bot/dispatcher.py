"""
Telegram Bot Dispatcher.
Initializes the aiogram client, registers command routers, and provides
channel broadcasting functionality.
"""

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from typing import Optional

from bot.handlers.admin import router as admin_router
from bot.handlers.commands import router as commands_router
from config.settings import settings
from db.repositories import signals
from utils.logger import logger

# Initialize Bot with HTML Parse Mode explicitly
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# Initialize Dispatcher and attach the commands
dp = Dispatcher()
dp.include_router(admin_router)  # Register admin commands first
dp.include_router(commands_router)  # Register public commands second


async def broadcast_signal(message_html: str, signal_id: Optional[str] = None) -> None:
    """
    Dispatches generated trade signals directly to the private VIP channel.
    Captures the message_id to allow real-time live updates via the DB.
    """
    if settings.SHADOW_MODE:
        logger.info("shadow_mode_active_signal_suppressed")
        return

    try:
        msg = await bot.send_message(
            chat_id=settings.SIGNAL_CHANNEL_ID, text=message_html, disable_web_page_preview=True
        )
        logger.info("signal_broadcast_success", channel=settings.SIGNAL_CHANNEL_ID, message_id=msg.message_id)

        # Save the Telegram Message ID so the status tracking loop can reply to it
        if signal_id:
            await signals.update_telegram_message_id(signal_id, msg.message_id)
    except Exception as e:
        logger.error("signal_broadcast_failed", error=str(e), channel_id=settings.SIGNAL_CHANNEL_ID)


async def start_bot():
    """Starts the Telegram polling loop."""
    logger.info("telegram_bot_polling_started")
    await dp.start_polling(bot)
