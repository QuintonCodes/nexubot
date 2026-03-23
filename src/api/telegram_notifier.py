import logging
import os
import requests
import time

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text, retries=3):
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram credentials missing.")
            return

        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}

        for attempt in range(retries):
            try:
                response = requests.post(self.base_url, json=payload, timeout=10)
                response.raise_for_status()
                return  # Success, break out of loop
            except requests.exceptions.Timeout as e:
                logger.warning("⚠️ Telegram send attempt {attempt + 1} failed: {e}")
                time.sleep(2)  # Wait 2 seconds before retrying

        logger.error("❌ Failed to send Telegram message after max retries.")

    def send_signal_alert(self, symbol, signal):
        emoji = "🟢" if signal["signal"] == "BUY" else "🔴"
        direction = "LONG" if signal["signal"] == "BUY" else "SHORT"

        msg = (
            f"{emoji} *NEXUBOT SIGNAL DETECTED* {emoji}\n\n"
            f"📈 *Pair:* {symbol}\n"
            f"🧭 *Direction:* {direction}\n"
            f"🧠 *AI Confidence:* {signal['confidence']:.1f}%\n\n"
            f"🎯 *Entry:* {signal['price']:.5f}\n"
            f"🛑 *Stop Loss:* {signal['sl']:.5f}\n"
            f"💰 *Take Profit:* {signal['tp']:.5f}\n\n"
            f"⚙️ *Strategy:* {signal.get('strategy', 'AI Core')}\n"
            f"⚖️ *Recommended Lot:* {signal.get('lot_size', 0.01)}"
        )
        self.send_message(msg)

    def send_trade_result(self, symbol, outcome, pnl, won):
        result_emoji = "🏆" if won else "💔"
        pnl_text = f"+R{pnl:.2f}" if won else f"-R{abs(pnl):.2f}"

        msg = (
            f"{result_emoji} *TRADE CLOSED: {symbol}* {result_emoji}\n\n"
            f"📝 *Outcome:* {outcome}\n"
            f"💵 *Profit/Loss:* {pnl_text}\n"
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
