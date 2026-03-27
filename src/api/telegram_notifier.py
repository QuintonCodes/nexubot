import asyncio
import logging
import os
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.config import VERSION

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, engine=None):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.engine = engine

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
        self.app.add_error_handler(self._error_handler)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Silently handles Telegram network timeouts to prevent console crashes."""
        logger.warning(f"Telegram Network Loop: {context.error}")

    async def _safe_send(self, chat_id, text):
        try:
            await self.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🚀 *Nexubot Online*\nUse `/analyze [SYMBOL]` to scan.", parse_mode="Markdown")

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or not self.engine:
            await update.message.reply_text("⚠️ Usage: `/analyze XAUUSDm`")
            return

        symbol = context.args[0]
        await update.message.reply_text(f"🔍 *Deep Scanning {symbol}...*", parse_mode="Markdown")

        try:
            klines = await self.engine.provider.fetch_klines(symbol, "M15", 200)
            if not klines:
                await update.message.reply_text(f"❌ Could not fetch data for {symbol}.")
                return

            df = self.engine.ai_engine.prepare_data(klines, heavy=True)
            curr = df.iloc[-1]
            htf_trend = await self.engine.ai_engine._get_htf_trend(symbol, self.engine.provider)
            structure = self.engine.ai_engine.strategy_analyzer.detect_structure(df)

            avg_atr = df["atr"].tail(24).mean()
            vol_ratio = curr["atr"] / avg_atr if avg_atr > 0 else 1.0
            dist_vwap = (curr["close"] - curr["vwap"]) / curr["vwap"] if curr["vwap"] != 0 else 0.0

            features = {
                "dist_to_vwap": dist_vwap,
                "mtf_trend_alignment": 1 if htf_trend == structure["structure"] else -1,
                "hour_norm": 0.5,
                "volatility_ratio": vol_ratio,
                "dist_to_nearest_fvg": 0.0,
                "is_in_breaker": 0.0,
                "htf_adx_strength": curr["adx"],
                "poi_status": 1.0 if structure["structure"] != "FLAT" else 0.0,
            }

            nn_result = self.engine.ai_engine.nn_brain.predict(features)
            win_prob = nn_result["prob"] * 100

            vol_msg = "⚠️ High Volatility (Expansion Phase)" if vol_ratio > 1.5 else "🌊 Stable (Consolidation Phase)"

            if win_prob > 80:
                ai_thought = "Highly favorable setup forming. Aligning with HTF institutional flow."
            elif win_prob > 50:
                ai_thought = "Neutral environment. Awaiting clear liquidity sweep or structural break."
            else:
                ai_thought = "Poor conditions. High probability of chop or false breakouts. Avoiding."

            msg = (
                f"🧠 *Deep AI Analysis: {symbol}*\n\n"
                f"📊 *HTF Flow:* {htf_trend}\n"
                f"🧭 *Local Structure:* {structure['structure']} (Last BOS: {structure.get('bos', 'None')})\n"
                f"🌊 *Market Condition:* {vol_msg}\n"
                f"📈 *ADX Momentum:* {curr['adx']:.1f}\n\n"
                f"🤖 *Neural Output:*\n"
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

    def send_signal_alert(self, symbol, signal):
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

    async def send_startup_message(self, win_rate, total_trades):
        msg = (
            f"🚀 *NEXUBOT {VERSION} ONLINE*\n\n"
            f"🤖 *Engine Status:* Deep Learning Linked\n"
            f"📊 *Historical Win Rate:* {win_rate:.1f}%\n"
            f"🔄 *Total Trades Analyzed:* {total_trades}\n\n"
            f"📡 _Scanning live markets for high-probability setups..._"
        )
        self.send_message(msg)

    def send_trade_result(self, symbol, outcome, pips, won):
        result_emoji = "🏆" if won else "💔"

        msg = (
            f"{result_emoji} *TRADE CLOSED: {symbol}* {result_emoji}\n\n"
            f"📝 *Outcome:* {outcome}\n"
            f"📏 *Captured:* {pips:.1f} Pips\n"
        )
        self.send_message(msg)

    def send_daily_report(self, wins, losses, total, pnl):
        win_rate = (wins / total * 100) if total > 0 else 0.0
        msg = (
            f"📊 *NEXUBOT DAILY REPORT* 📊\n\n"
            f"🔄 *Trades Taken:* {total}\n"
            f"✅ *Wins:* {wins}\n"
            f"❌ *Losses:* {losses}\n"
            f"🎯 *Win Rate:* {win_rate:.1f}%\n\n"
            f"💵 *Net PnL:* R{pnl:.2f}\n\n"
            f"🧠 *Engine Status:* Continuous learning active. Neural weights updated based on today's outcomes."
        )
        self.send_message(msg)
