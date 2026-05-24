import asyncio
import logging
import os
import pandas as pd
from datetime import datetime
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.analysis.indicators import TechnicalAnalyzer
from src.config import SESSION_CONFIG, TIMEFRAME, VERSION

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
        """Sends a welcome message with instructions on how to use the bot's features."""
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

        if not context.args or context.args[0] == "ALL":
            self.engine.focused_symbols = []
            await update.message.reply_text(
                "🌐 *Focus Mode Disabled:* Scanning ALL allowed symbols.", parse_mode="Markdown"
            )
        else:
            symbols = [s for s in context.args]
            self.engine.focused_symbols = symbols
            await update.message.reply_text(
                f"🎯 *Focus Mode Active:*\nThe engine is now exclusively hunting setups on: *{', '.join(symbols)}*",
                parse_mode="Markdown",
            )

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Runs a manual, deep SMC analysis matching the exact AI Engine logic."""
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

            # Data Preparation & Shared Row Variables
            df = self.engine.ai_engine.prepare_data(klines)
            curr = df.iloc[-1]
            close_price = curr["close"]
            atr = float(curr.get("atr", 1.0))
            if atr <= 0:
                atr = 1.0

            # 1. Structural & HTF Analysis
            htf_trend = await self.engine.ai_engine._get_htf_trend(symbol, self.engine.provider)
            structure = TechnicalAnalyzer.detect_structure(df)
            active_fvgs, active_ifvgs, active_obs = TechnicalAnalyzer.extract_active_pois(df)

            # 2. Daily Levels & Sweep Depth
            df_temp = df.copy()
            df_temp["date"] = pd.to_datetime(df_temp["time"], unit="s").dt.date
            prev_days = df_temp[df_temp["date"] < df_temp["date"].iloc[-1]]
            if not prev_days.empty:
                last_day = prev_days["date"].iloc[-1]
                yesterday_df = prev_days[prev_days["date"] == last_day]
                pdh, pdl = yesterday_df["high"].max(), yesterday_df["low"].min()
            else:
                pdh, pdl = None, None

            daily_levels = {"pdh": pdh, "pdl": pdl}
            is_liquidity_swept, sweep_depth_atr = TechnicalAnalyzer.detect_liquidity_sweeps(
                curr, structure, daily_levels
            )

            # 3. Killzone Mapping
            hour = datetime.now().hour
            active_killzone = 0.0
            killzone_name = "Dead Zone"
            if SESSION_CONFIG["ASIAN_START"] <= hour < SESSION_CONFIG["ASIAN_END"]:
                active_killzone = 1.0
                killzone_name = "Asian Killzone"
            elif SESSION_CONFIG["LONDON_START"] <= hour < SESSION_CONFIG["LONDON_END"]:
                active_killzone = 2.0
                killzone_name = "London Killzone"
            elif SESSION_CONFIG["NY_START"] <= hour < SESSION_CONFIG["NY_END"]:
                active_killzone = 3.0
                killzone_name = "New York Killzone"

            # 4. Nearest POI & Mitigation Tracking
            all_pois = active_fvgs + active_ifvgs + active_obs
            dist_nearest_poi_atr = 0.0
            mitigation_count = 0

            if all_pois:
                nearest_poi = min(
                    all_pois, key=lambda x: min(abs(x["high"] - close_price), abs(x["low"] - close_price))
                )
                raw_distance = min(abs(nearest_poi["high"] - close_price), abs(nearest_poi["low"] - close_price))
                dist_nearest_poi_atr = raw_distance / atr
                mitigation_count = nearest_poi.get("mitigations", 0)

            # Construct the Feature Dictionary exactly as the ML model expects
            features = {
                "is_htf_aligned": htf_trend,
                "is_liquidity_swept": float(is_liquidity_swept),
                "is_in_fvg": 1.0 if any(f["low"] <= curr["close"] <= f["high"] for f in active_fvgs) else 0.0,
                "is_in_ifvg": 1.0 if any(i_f["low"] <= curr["close"] <= i_f["high"] for i_f in active_ifvgs) else 0.0,
                "is_in_orderblock": 1.0 if any(o["low"] <= curr["close"] <= o["high"] for o in active_obs) else 0.0,
                "structural_break": structure.get("structural_break", 0.0),
                "active_killzone": active_killzone,
                "distance_to_poi": dist_nearest_poi_atr,
                "pd_array_status": structure.get("pd_array", 0.5),
                "mitigation_count": float(mitigation_count),
                "sweep_depth_atr": sweep_depth_atr,
            }

            # Formatting Display Variables for Telegram Readability
            # HTF Map
            htf_text = "BULLISH 🐂" if htf_trend == 1.0 else ("BEARISH 🐻" if htf_trend == -1.0 else "FLAT ➖")

            # PD Array Map
            pd_status = features["pd_array_status"]
            pd_text = "Equilibrium"
            if pd_status <= 0.4:
                pd_text = f"Discount 🟢 ({pd_status:.2f})"
            elif pd_status >= 0.6:
                pd_text = f"Premium 🔴 ({pd_status:.2f})"

            # Structure Break Map
            struct_break = features["structural_break"]
            break_text = "None"
            if struct_break == 1.0:
                break_text = "Bullish BOS 📈"
            elif struct_break == -1.0:
                break_text = "Bearish BOS 📉"
            elif struct_break == 2.0:
                break_text = "Bullish CHoCH 🔄"
            elif struct_break == -2.0:
                break_text = "Bearish CHoCH 🔄"

            # Sweep Map
            sweep_text = "None"
            if is_liquidity_swept == 3:
                sweep_text = f"Daily High/Low Swept 🔥 ({sweep_depth_atr:.1f} ATR)"
            elif is_liquidity_swept == 2:
                sweep_text = f"Major Swing Swept ⚡ ({sweep_depth_atr:.1f} ATR)"
            elif is_liquidity_swept == 1:
                sweep_text = f"Internal Pivot Swept ({sweep_depth_atr:.1f} ATR)"

            vwap_trend = "BULL" if close_price > curr.get("vwap", 0) else "BEAR"

            # Run Neural Predictor
            nn_result = self.engine.ai_engine.nn_brain.predict(features)
            raw_prob = nn_result["prob"]

            # Incorporate Session Awareness & Historical DB Performance for accurate manual read
            session_info = self.engine.ai_engine._get_session_status(symbol)
            base_conf = 60.0

            trend_bonus = 0
            if (htf_trend == 1.0 and structure["structure"] == "BULL") or (
                htf_trend == -1.0 and structure["structure"] == "BEAR"
            ):
                trend_bonus = 15

            nn_factor = (raw_prob - 0.5) * 40

            hist_win_rate = await self.engine.db.get_pair_performance(symbol) if self.engine.db else 0.5
            history_factor = -10 if hist_win_rate < 0.4 else (10 if hist_win_rate > 0.6 else 0)

            final_conf = (base_conf + trend_bonus + history_factor + nn_factor) * session_info["multiplier"]
            win_prob = max(0.0, min(99.0, final_conf))

            # Interpretation Logic
            if win_prob > 80:
                ai_thought = (
                    "Highly favorable setup forming. Aligning with HTF institutional flow and Killzone momentum."
                )
            elif win_prob > 60:
                ai_thought = "Viable environment. Awaiting clear liquidity sweep or decisive structural confirmation."
            else:
                ai_thought = "Poor conditions. High probability of chop or false breakouts. Avoiding."

            msg = (
                f"🧠 *Deep SMC Analysis: {symbol}*\n\n"
                f"🌍 *Active Killzone:* {killzone_name} (Vol Map: {session_info['multiplier']}x)\n"
                f"📊 *HTF Flow (1H):* {htf_text}\n"
                f"🧭 *Local Flow ({TIMEFRAME}):* {structure['structure']} | Break: {break_text}\n"
                f"📏 *PD Array:* Price in {pd_text}\n"
                f"💧 *Liquidity Profile:* {len(active_fvgs)} FVGs | {len(active_ifvgs)} IFVGs | {len(active_obs)} OBs\n"
                f"🎯 *Nearest POI Status:* {dist_nearest_poi_atr:.1f} ATR Away | Touched {mitigation_count}x\n"
                f"🧹 *Sweep Status:* {sweep_text}\n"
                f"🌊 *VWAP State:* {vwap_trend}\n\n"
                f"🤖 *Neural Output:*\n"
                f"• _Trend Alignment:_ {'Aligned ✅' if trend_bonus > 0 else 'Counter ⚠️'}\n"
                f"• _Calculated AI Confidence:_ *{win_prob:.1f}%*\n"
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
        """Sends a message to the configured Telegram chat asynchronously."""
        if self.bot_token and self.chat_id:
            asyncio.create_task(self._safe_send(self.chat_id, text))

    async def send_shutdown_message(self):
        """Sends a final shutdown message with a summary of the bot's state."""
        msg = (
            f"🛑 *NEXUBOT SHUTDOWN INITIATED*\n\n"
            f"💾 Saving active trades to secure memory...\n"
            f"🧠 Updating Neural Network weights...\n"
            f"🔌 Disconnecting from Broker.\n\n"
            f"💤 _System Offline._"
        )
        self.send_message(msg)

    def send_signal_alert(self, symbol: str, signal: dict):
        """Formats and sends a detailed signal alert to Telegram."""
        if signal["signal"] == "BUY":
            header = "🟢 *BUY SIGNAL DETECTED* 🟢"
        else:
            header = "🔴 *SELL SIGNAL DETECTED* 🔴"
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

    async def send_startup_message(self, win_rate: float, total_trades: int):
        """Sends a startup message with the latest performance metrics."""
        msg = (
            f"🚀 *NEXUBOT {VERSION} ONLINE*\n\n"
            f"🤖 *Engine Status:* Deep Learning Linked\n"
            f"📊 *Historical Win Rate:* {win_rate:.1f}%\n"
            f"🔄 *Total Trades Analyzed:* {total_trades}\n\n"
            f"📡 _Scanning live markets for high-probability setups ..._"
        )
        self.send_message(msg)

    def send_trade_result(self, symbol: str, outcome: str, pips: float, won: bool, pnl: float, currency: str):
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

    def send_daily_report(self, wins: int, losses: int, total: int, pnl: float, currency: str):
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
