"""
Telegram Signal Message Formatter.
Converts TradeSignal dataclasses into strictly formatted HTML messages.
Handles separate formats for public subscribers and internal admin analytics.
"""

from strategies.models import TradeSignal


def format_signal_for_channel(signal: TradeSignal) -> str:
    """Clean public format — only actionable trade information for subscribers."""
    direction_emoji = "🟢 BUY" if signal.direction == "buy" else "🔴 SELL"

    # Calculate RRR dynamically for accurate display per level
    sl_dist = abs(signal.entry_price - signal.stop_loss)
    rr_tp1 = abs(signal.take_profit_1 - signal.entry_price) / sl_dist if sl_dist > 0 else 0
    rr_tp2 = abs(signal.take_profit_2 - signal.entry_price) / sl_dist if sl_dist > 0 else 0
    rr_tp3 = abs(signal.take_profit_3 - signal.entry_price) / sl_dist if sl_dist > 0 else 0

    return f"""╔══════════════════════════════╗
║  {direction_emoji} — {signal.symbol}
╚══════════════════════════════╝

📈 <b>ENTRY:</b>    {signal.entry_price:,.2f}
🛑 <b>STOP LOSS:</b> {signal.stop_loss:,.2f}

🎯 <b>TP1:</b>  {signal.take_profit_1:,.2f}   <i>(1:{rr_tp1:.1f}R)</i>
💰 <b>TP2:</b>  {signal.take_profit_2:,.2f}   <i>(1:{rr_tp2:.1f}R)</i>
🏆 <b>TP3:</b>  {signal.take_profit_3:,.2f}   <i>(1:{rr_tp3:.1f}R)</i>

⚠️ <i>Risk accordingly to your risk management · Manage your position.</i>"""


def format_signal_for_admin(signal: TradeSignal) -> str:
    """Full verbose format for the admin /scan command detailing the internal logic."""
    emoji = "🟢 BUY SIGNAL" if signal.direction == "buy" else "🔴 SELL SIGNAL"
    factors_text = "\n".join([f"  ✅ {f}" for f in signal.confluence_factors])
    time_str = signal.timestamp.strftime("%Y-%m-%d %H:%M UTC")

    sl_dist = abs(signal.entry_price - signal.stop_loss)
    rr_tp1 = abs(signal.take_profit_1 - signal.entry_price) / sl_dist if sl_dist > 0 else 0
    rr_tp2 = abs(signal.take_profit_2 - signal.entry_price) / sl_dist if sl_dist > 0 else 0
    rr_tp3 = abs(signal.take_profit_3 - signal.entry_price) / sl_dist if sl_dist > 0 else 0

    msg = f"""╔══════════════════════════════╗
║  {emoji} — {signal.symbol}
╚══════════════════════════════╝

📐 <b>Setup:</b> {signal.signal_type}
📍 <b>Array Zone:</b> {signal.pd_zone}
🕐 <b>Session:</b> {signal.session}
⭐ <b>Confluence Score:</b> {signal.confluence_score}/100

──────────────────────────────
📈 <b>ENTRY:</b>  {signal.entry_price:,.2f}
🛑 <b>STOP LOSS:</b>  {signal.stop_loss:,.2f}
🎯 <b>TP1:</b>  {signal.take_profit_1:,.2f}  (+1:{rr_tp1:.1f}R)
💰 <b>TP2:</b>  {signal.take_profit_2:,.2f}  (+1:{rr_tp2:.1f}R)
🏆 <b>TP3:</b>  {signal.take_profit_3:,.2f}  (+1:{rr_tp3:.1f}R)
──────────────────────────────

📊 <b>Confluence Factors:</b>
{factors_text}

⏰ {time_str}"""

    return msg
