import asyncio
import logging
import os
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.analysis.indicators import TechnicalAnalyzer
from src.config import VERSION, TIMEFRAME

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

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Silently handles Telegram network timeouts to prevent console crashes."""
        logger.warning(f"Telegram Network Loop: {context.error}")

    def _get_currency_symbol(self, currency: str) -> str:
        """Helper to return the correct currency symbol based on account base."""
        return "R" if currency.upper() == "ZAR" else "$"

    async def _safe_send(self, chat_id, text, retries=3):
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

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"🚀 *Nexubot {VERSION} Online*\n"
            f"• `/analyze [SYMBOL]` to run a deep SMC scan.\n"
            f"• `/focus [SYMBOLS]` to isolate specific pairs (e.g., `/focus XAUUSDm EURUSDm`).\n"
            f"• `/focus ALL` to resume full market scanning.",
            parse_mode="Markdown",
        )

    async def cmd_focus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Allows the user to select specific symbols to focus the scanner on."""
        if not self.engine:
            return

        if not context.args or context.args[0].upper() == "ALL":
            self.engine.focused_symbols = []
            await update.message.reply_text(
                "🌐 *Focus Mode Disabled:* Scanning ALL allowed symbols.", parse_mode="Markdown"
            )
        else:
            symbols = [s.upper() for s in context.args]
            self.engine.focused_symbols = symbols
            await update.message.reply_text(
                f"🎯 *Focus Mode Active:*\nThe engine is now exclusively hunting setups on: *{', '.join(symbols)}*",
                parse_mode="Markdown",
            )

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or not self.engine:
            await update.message.reply_text("⚠️ Usage: `/analyze XAUUSDm`")
            return

        symbol = context.args[0]
        await update.message.reply_text(f"🔍 *Deep SMC Scanning {symbol} ...*", parse_mode="Markdown")

        try:
            klines = await self.engine.provider.fetch_klines(symbol, TIMEFRAME, 500)
            if not klines:
                await update.message.reply_text(f"❌ Could not fetch data for {symbol}.")
                return

            df = self.engine.ai_engine.prepare_data(klines)
            curr = df.iloc[-1]
            htf_trend = await self.engine.ai_engine._get_htf_trend(symbol, self.engine.provider)
            structure = TechnicalAnalyzer.detect_structure(df)

            active_fvgs, active_obs = TechnicalAnalyzer.extract_active_pois(df)

            # Calculate precise distances to liquidity
            all_pois = active_fvgs + active_obs
            dist_nearest_poi = 0.0
            in_fvg = 0
            in_ob = 0

            if all_pois:
                nearest = min(
                    all_pois, key=lambda x: min(abs(x["high"] - curr["close"]), abs(x["low"] - curr["close"]))
                )
                dist_nearest_poi = (
                    min(abs(nearest["high"] - curr["close"]), abs(nearest["low"] - curr["close"])) / curr["close"]
                )

            if any(fvg["low"] <= curr["close"] <= fvg["high"] for fvg in active_fvgs):
                in_fvg = 1
            if any(ob["low"] <= curr["close"] <= ob["high"] for ob in active_obs):
                in_ob = 1

            vol_spike = 1 if curr["volume"] > df["volume"].tail(20).mean() * 1.5 else 0
            vwap_trend = "BULL" if curr["close"] > curr.get("vwap", 0) else "BEAR"

            features = {
                "is_htf_aligned": 1 if (htf_trend == structure["structure"] and htf_trend != "FLAT") else -1,
                "is_liquidity_swept": 0,  # Reserved for active sweep detection
                "is_in_fvg": in_fvg,
                "is_in_orderblock": in_ob,
                "structural_break": 1 if structure["bos"] else (2 if structure["choch"] else 0),
                "session_volume_spike": vol_spike,
                "distance_to_poi": dist_nearest_poi,
            }

            nn_result = self.engine.ai_engine.nn_brain.predict(features)
            win_prob = nn_result["prob"] * 100

            if win_prob > 80:
                ai_thought = "Highly favorable setup forming. Aligning with HTF institutional flow."
            elif win_prob > 50:
                ai_thought = "Neutral environment. Awaiting clear liquidity sweep or structural break."
            else:
                ai_thought = "Poor conditions. High probability of chop or false breakouts. Avoiding."

            msg = (
                f"🧠 *Deep SMC Analysis: {symbol}*\n\n"
                f"📊 *HTF Flow (1H):* {htf_trend}\n"
                f"🧭 *Local Structure ({TIMEFRAME}):* {structure['structure']} (Last Break: {structure.get('bos') or structure.get('choch') or 'None'})\n"
                f"💧 *Liquidity Profile:* {len(active_fvgs)} FVGs | {len(active_obs)} OBs Active\n"
                f"🌊 *VWAP State:* {vwap_trend}\n"
                f"🔥 *Volume Anomaly:* {'Detected' if vol_spike else 'Normal'}\n\n"
                f"🤖 *Neural Output:*\n"
                f"• _Trend Alignment:_ {'Aligned ✅' if features['is_htf_aligned'] == 1 else 'Counter ⚠️'}\n"
                f"• _Calculated Win Probability:_ *{win_prob:.1f}%*\n"
                f"• _AI Conclusion:_ {ai_thought}"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Analysis failed: {e}")

    async def initialize(self):
        """Starts the telegram polling asynchronously."""
        if not self.bot_token:
            return
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        logger.info("✅ Async Telegram Listener Started")

    def send_message(self, text):
        if self.bot_token and self.chat_id:
            asyncio.create_task(self._safe_send(self.chat_id, text))

    async def send_shutdown_message(self):
        msg = (
            f"🛑 *NEXUBOT SHUTDOWN INITIATED*\n\n"
            f"💾 Saving active trades to secure memory...\n"
            f"🧠 Updating Neural Network weights...\n"
            f"🔌 Disconnecting from Broker.\n\n"
            f"💤 _System Offline._"
        )
        self.send_message(msg)

    def send_signal_alert(self, symbol: str, signal: dict):
        if signal["signal"] == "BUY":
            header = "🟢 *BUY SIGNAL DETECTED* 🟢"
        else:
            header = "🔴 *SELL SIGNAL DETECTED* 🔴"
        direction = "LONG" if signal["signal"] == "BUY" else "SHORT"
        volatility_warning = "⚠️ *VOLATILE ASSET*" if signal.get("is_high_risk", False) else ""

        msg = (
            f"{header}\n\n"
            f"📈 *Pair:* {symbol}\n"
            f"🧭 *Direction:* {direction}\n"
            f"🧩 *Setup:* {signal.get('strategy', 'SMC Execution')}\n"
            f"{volatility_warning}\n"
            f"🧠 *AI Confidence:* {signal['confidence']:.1f}%\n\n"
            f"🎯 *Entry:* {signal['price']:.5f}\n"
            f"🛑 *Stop Loss:* {signal['sl']:.5f}\n"
            f"💰 *TP1:* {signal['tp1']:.5f}\n"
            f"💰 *TP2:* {signal['tp2']:.5f}\n"
            f"💰 *TP3:* {signal['tp3']:.5f}\n\n"
            f"⚖️ *Recommended Lot:* {signal.get('lot_size', 0.01)}"
        )
        self.send_message(msg)

    async def send_startup_message(self, win_rate: float, total_trades: int):
        msg = (
            f"🚀 *NEXUBOT {VERSION} ONLINE*\n\n"
            f"🤖 *Engine Status:* Deep Learning Linked\n"
            f"📊 *Historical Win Rate:* {win_rate:.1f}%\n"
            f"🔄 *Total Trades Analyzed:* {total_trades}\n\n"
            f"📡 _Scanning live markets for high-probability setups ..._"
        )
        self.send_message(msg)

    def send_trade_result(self, symbol: str, outcome: str, pips: float, won: bool, pnl: float, currency: str):
        result_emoji = "🏆" if won else "💔"
        curr_sym = self._get_currency_symbol(currency)

        msg = (
            f"{result_emoji} *TRADE CLOSED: {symbol}* {result_emoji}\n\n"
            f"📝 *Outcome:* {outcome}\n"
            f"📏 *Pips:* {pips:.1f} Pips\n"
            f"💵 *Net PnL:* {curr_sym}{pnl:.2f}\n"
        )
        self.send_message(msg)

    def send_daily_report(self, wins: int, losses: int, total: int, pnl: float, currency: str):
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
