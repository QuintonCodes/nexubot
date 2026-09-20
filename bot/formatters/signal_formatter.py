"""
Telegram Signal Message Formatter.
Converts TradeSignal dataclasses into strictly formatted HTML messages.
"""

from config.settings import settings
from strategies.models import TradeSignal


def format_trade_signal(signal: TradeSignal) -> str:
    """Formats a TradeSignal into the official Nexubot Telegram HTML layout."""

    # 1. Header & Emojis
    emoji = "🟢 BUY SIGNAL" if signal.direction == "buy" else "🔴 SELL SIGNAL"

    # 2. Confluence Factors (bullet points)
    factors_text = "\n".join([f"  ✅ {f}" for f in signal.confluence_factors])

    # 3. Time formatting
    time_str = signal.timestamp.strftime("%Y-%m-%d %H:%M UTC")

    # 4. Construct the HTML string
    msg = f"""╔══════════════════════════════╗
║  {emoji} — {signal.symbol}
╚══════════════════════════════╝

📐 <b>Setup:</b> {signal.signal_type}
⏱ <b>Timeframe:</b> {signal.timeframe}
⭐ <b>Confluence Score:</b> {signal.confluence_score}/100

──────────────────────────────
📈 <b>ENTRY:</b>  {signal.entry_price:,.2f}
🛑 <b>STOP LOSS:</b>  {signal.stop_loss:,.2f}
🎯 <b>TP1:</b>  {signal.take_profit_1:,.2f}  (+1:1.5)
🏆 <b>TP2:</b>  {signal.take_profit_2:,.2f}  (+1:{signal.risk_reward:.1f})
──────────────────────────────

📊 <b>Confluence Factors:</b>
{factors_text}

⏰ {time_str}

<i>Risk {settings.RISK_PERCENT}% of account. Manage your position.</i>"""

    return msg
