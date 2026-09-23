from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING

from bot.dispatcher import bot, broadcast_signal
from bot.formatters.signal_formatter import format_signal_for_channel
from config.settings import settings
from data.candle_store import candle_store
from data.normalizer import normalize_ohlcv
from db.repositories import signals
from strategies.confluence import ConfluenceEngine
from utils.logger import logger

# Evaluated during static type checking, bypassed during runtime module initialization
if TYPE_CHECKING:
    from data.twelve_data_client import TwelveDataClient


def _calc_pips(sig: dict, tp_level: int) -> float:
    # Safely convert price differential to estimated PIP values
    multiplier = 10 if "XAU" in sig["symbol"] else (100 if "JPY" in sig["symbol"] else 10000)
    return abs(sig["entry_price"] - sig[f"take_profit_{tp_level}"]) * multiplier


async def monitor_active_signals(symbol: str, current_price: float) -> None:
    """Monitors live market ticks against active trade setups to manage trade lifecycle."""
    active = await signals.get_active_signals(symbol)
    for sig in active:
        update_msg = None
        new_status = None

        if sig["direction"] == "buy":
            if current_price >= sig["take_profit_3"]:
                new_status, update_msg = "tp3_hit", f"🏆 TP3 Hit! +{_calc_pips(sig, 3):.0f} pips 🎯"
            elif current_price >= sig["take_profit_2"]:
                new_status, update_msg = "tp2_hit", f"✅ TP2 Hit! +{_calc_pips(sig, 2):.0f} pips 💰"
            elif current_price >= sig["take_profit_1"]:
                new_status, update_msg = "tp1_hit", f"✅ TP1 Hit! +{_calc_pips(sig, 1):.0f} pips 📈"
            elif current_price <= sig["stop_loss"]:
                new_status, update_msg = "sl_hit", "❌ Setup Invalidated — SL Hit 🛑"
        else:
            if current_price <= sig["take_profit_3"]:
                new_status, update_msg = "tp3_hit", f"🏆 TP3 Hit! +{_calc_pips(sig, 3):.0f} pips 🎯"
            elif current_price <= sig["take_profit_2"]:
                new_status, update_msg = "tp2_hit", f"✅ TP2 Hit! +{_calc_pips(sig, 2):.0f} pips 💰"
            elif current_price <= sig["take_profit_1"]:
                new_status, update_msg = "tp1_hit", f"✅ TP1 Hit! +{_calc_pips(sig, 1):.0f} pips 📈"
            elif current_price >= sig["stop_loss"]:
                new_status, update_msg = "sl_hit", "❌ Setup Invalidated — SL Hit 🛑"

        if new_status and new_status != sig["status"]:
            await signals.update_signal_status(sig["id"], new_status)
            if sig.get("telegram_message_id"):
                await bot.send_message(
                    chat_id=settings.SIGNAL_CHANNEL_ID,
                    text=f"📊 <b>{sig['symbol']} Update</b>\n\n{update_msg}",
                    reply_to_message_id=sig["telegram_message_id"],
                    parse_mode="HTML",
                )


async def bootstrap_historical_data(client: TwelveDataClient, symbol: str):
    """Fetches initial data for all timeframes on startup."""
    timeframes = settings.HTF_TIMEFRAMES + [settings.LTF_TIMEFRAMES]

    for tf in timeframes:
        logger.info("bootstrapping_data", symbol=symbol, timeframe=tf)
        raw_df = await client.get_historical_ohlcv(symbol, tf, outputsize=500)
        norm_df = normalize_ohlcv(raw_df)
        await candle_store.initialize(symbol, tf, norm_df)

    engine = ConfluenceEngine(symbol)
    await engine.scan_htf()
    await engine.scan_mtf()
    await engine.scan_mtf_confirmation()


async def on_candle_close(symbol: str, timeframe: str, candle: pd.Series) -> None:
    """Callback fired by WebSocket tick aggregator precisely on 5-minute rollovers."""
    await candle_store.add_candle(symbol, timeframe, candle)

    engine = ConfluenceEngine(symbol)
    signal = await engine.scan_ltf_entry()

    if signal:
        logger.info("signal_confirmed_broadcasting", signal_id=signal.signal_id)
        msg_html = format_signal_for_channel(signal)
        await broadcast_signal(msg_html, signal.signal_id)
