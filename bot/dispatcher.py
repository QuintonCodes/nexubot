"""
Telegram Bot Dispatcher.
Initializes the aiogram client, registers command routers, and provides
channel broadcasting functionality.
"""

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers.admin import router as admin_router
from bot.handlers.commands import router as commands_router
from config.settings import settings
from utils.logger import logger

# Initialize Bot with HTML Parse Mode explicitly
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# Initialize Dispatcher and attach the commands
dp = Dispatcher()
dp.include_router(admin_router)  # Register admin commands first
dp.include_router(commands_router)  # Register public commands second


async def broadcast_signal(message_html: str) -> None:
    """
    Sends a formatted signal to the configured Telegram channel.
    Silently bypasses sending if SHADOW_MODE is enabled in settings.
    """
    if settings.SHADOW_MODE:
        logger.info("shadow_mode_active_signal_suppressed")
        return

    try:
        await bot.send_message(chat_id=settings.TELEGRAM_CHANNEL_ID, text=message_html)
        logger.info("telegram_signal_sent", channel=settings.TELEGRAM_CHANNEL_ID)
    except Exception as e:
        logger.error("telegram_send_failed", error=str(e))


async def start_bot():
    """Starts the Telegram polling loop."""
    logger.info("telegram_bot_polling_started")
    await dp.start_polling(bot)
