import asyncio
import logging
import os
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.config import __version__

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Handles all asynchronous Telegram communication.
    Responsible for broadcasting signals, analyzing markets on demand, and session focus control.
    """

    def __init__(self, engine=None):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.engine = engine
        self.msg_queue = asyncio.Queue()

        # Initialize focused_symbols dynamically for the scanner to read
        if self.engine:
            self.engine.focused_symbols = []

        self.app = (
            Application.builder()
            .token(self.bot_token)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .get_updates_read_timeout(42.0)
            .pool_timeout(30.0)
            .build()
        )
        self.bot = Bot(token=self.bot_token)

        # Register commands
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        self.app.add_handler(CommandHandler("focus", self.cmd_focus))
        self.app.add_error_handler(self._error_handler)

    async def _drain_queue(self):
        """Background daemon processing the telegram message queue."""
        while True:
            text = await self.msg_queue.get()
            await self._safe_send(self.chat_id, text)
            self.msg_queue.task_done()

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Silently handles Telegram network timeouts to prevent console crashes."""
        logger.warning(f"Telegram Network Loop: {context.error}")

    def _get_currency_symbol(self, currency: str) -> str:
        """Helper to return the correct currency symbol based on account base."""
        return "R" if currency.upper() == "ZAR" else "$"

    async def _safe_send(self, chat_id, text, retries=3) -> None:
        """Attempts to send a message multiple times if the network drops."""
        for i in range(retries):
            try:
                await self.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                return
            except Exception as e:
                if i < retries - 1:
                    await asyncio.sleep(5)  # Wait 5 seconds before retrying
                else:
                    logger.warning(f"Failed to send Telegram messageafter {retries} attempts: {e.__class__.__name__}")

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Runs a manual, deep SMC analysis matching the exact AI Engine logic."""
        if not context.args or not self.engine:
            await update.message.reply_text("⚠️ Usage: `/analyze XAUUSDm`")
            return

        symbol = context.args[0]
        await update.message.reply_text(f"🔍 *Deep SMC Scanning {symbol}...*", parse_mode="Markdown")

        try:
            report = await self.engine.ai_engine.analyze_for_report(symbol, self.engine.provider)
            await update.message.reply_text(
                report if report else f"❌ Could not fetch data for {symbol}.", parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Analysis failed: {e}")

    async def cmd_focus(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Allows the user to select specific symbols to focus the scanner on."""
        if not self.engine:
            return

        if not context.args or context.args[0] == "ALL":
            self.engine.focused_symbols = []
            await update.message.reply_text(
                "🌐 *Focus Mode Disabled:* Scanning ALL allowed symbols.", parse_mode="Markdown"
            )
        else:
            symbols = list(context.args)
            self.engine.focused_symbols = symbols
            await update.message.reply_text(
                f"🎯 *Focus Mode Active:*\nThe engine is now exclusively hunting setups on: *{', '.join(symbols)}*",
                parse_mode="Markdown",
            )

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends a welcome message with instructions on how to use the bot's features."""
        await update.message.reply_text(
            f"🚀 *Nexubot {__version__} Online*\n"
            f"• `/analyze [SYMBOL]` to run a deep SMC scan.\n"
            f"• `/focus [SYMBOLS]` to isolate specific pairs (e.g., `/focus XAUUSDm EURUSDm`).\n"
            f"• `/focus ALL` to resume full market scanning.",
            parse_mode="Markdown",
        )

    async def initialize(self) -> bool:
        """Starts the telegram polling asynchronously."""
        if not self.bot_token:
            return False
        try:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            asyncio.create_task(self._drain_queue())
            logger.info("✅ Async Telegram Listener Started")
            return True
        except Exception as e:
            logger.error(f"⚠️ Telegram Initialization Failed: {e}")
            return False

    def send_daily_report(self, wins: int, losses: int, total: int, pnl: float, currency: str) -> None:
        """Sends a comprehensive daily performance report to Telegram with win/loss breakdown and net PnL."""
        win_rate = (wins / total * 100) if total > 0 else 0.0
        curr_sym = self._get_currency_symbol(currency)

        msg = (
            f"📊 *NEXUBOT DAILY REPORT* 📊\n\n"
            f"🔄 *Trades Taken:* {total}\n"
            f"✅ *Wins:* {wins}\n"
            f"❌ *Losses:* {losses}\n"
            f"🎯 *Win Rate:* {win_rate:.1f}%\n\n"
            f"💵 *Net PnL:* {curr_sym}{pnl:.2f}\n\n"
        )
        self.send_message(msg)

    def send_message(self, text: str) -> None:
        """Sends a message to the configured Telegram chat asynchronously."""
        if not (self.bot_token and self.chat_id):
            return
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.msg_queue.put(text))
        except RuntimeError:
            try:
                asyncio.run(self._safe_send(self.chat_id, text))
            except Exception as e:
                logger.warning(f"Failed to send sync Telegram message: {e}")

    async def send_shutdown_message(self) -> None:
        """Sends a final shutdown message with a summary of the bot's state."""
        msg = (
            f"🛑 *NEXUBOT SHUTDOWN INITIATED*\n\n"
            f"💾 Saving active trades to secure memory...\n"
            f"🧠 Updating Neural Network weights...\n"
            f"🔌 Disconnecting from Broker.\n\n"
            f"💤 _System Offline._"
        )
        self.send_message(msg)

    def send_signal_alert(self, symbol: str, signal: dict) -> None:
        """Formats and sends a detailed signal alert to Telegram."""
        header = "🟢 *BUY SIGNAL DETECTED* 🟢" if signal["signal"] == "BUY" else "🔴 *SELL SIGNAL DETECTED* 🔴"
        direction = "LONG" if signal["signal"] == "BUY" else "SHORT"
        volatility_warning = "⚠️ *VOLATILE ASSET*" if signal.get("is_high_risk", False) else ""

        # Dynamic Decimal Formatting
        digits = signal.get("digits", 5)
        fmt = f"{{:.{digits}f}}"

        entry_str = fmt.format(signal["price"])
        sl_str = fmt.format(signal["sl"])
        tp1_str = fmt.format(signal["tp1"])
        tp2_str = fmt.format(signal["tp2"])
        tp3_str = fmt.format(signal["tp3"])

        msg = (
            f"{header}\n\n"
            f"📈 *Pair:* {symbol}\n"
            f"🧭 *Direction:* {direction}\n"
            f"🧩 *Setup:* {signal.get('strategy', 'SMC Execution')}\n"
            f"{volatility_warning}\n"
            f"🧠 *AI Confidence:* {signal['confidence']:.1f}%\n\n"
            f"🎯 *Entry:* {entry_str}\n"
            f"🛑 *Stop Loss:* {sl_str}\n"
            f"💰 *TP1:* {tp1_str}\n"
            f"💰 *TP2:* {tp2_str}\n"
            f"💰 *TP3:* {tp3_str}\n\n"
            f"⚖️ *Recommended Lot:* {signal.get('lot_size', 0.01)}"
        )
        self.send_message(msg)

    async def send_startup_message(self, win_rate: float, total_trades: int) -> None:
        """Sends a startup message with the latest performance metrics."""
        msg = (
            f"🚀 *NEXUBOT {__version__} ONLINE*\n\n"
            f"🤖 *Engine Status:* Deep Learning Linked\n"
            f"📊 *Historical Win Rate:* {win_rate:.1f}%\n"
            f"🔄 *Total Trades Analyzed:* {total_trades}\n\n"
            f"📡 _Scanning live markets for high-probability setups..._"
        )
        self.send_message(msg)

    def send_trade_result(self, symbol: str, outcome: str, pips: float, won: bool, pnl: float, currency: str) -> None:
        """Sends a trade result summary to Telegram with clear win/loss indicators and performance metrics."""
        result_emoji = "🏆" if won else "💔"
        curr_sym = self._get_currency_symbol(currency)

        msg = (
            f"{result_emoji} *TRADE CLOSED: {symbol}* {result_emoji}\n\n"
            f"📝 *Outcome:* {outcome}\n"
            f"📏 *Pips:* {pips:.1f} Pips\n"
            f"💵 *Net PnL:* {curr_sym}{pnl:.2f}\n"
        )
        self.send_message(msg)
