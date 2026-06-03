import asyncio
import logging
import time
from typing import List

from src.config import (
    CANDLE_LIMIT,
    FALLBACK_CRYPTO,
    FALLBACK_FOREX,
    FALLBACK_INDICES,
    FALLBACK_METALS,
    MAX_SIGNALS_PER_SCAN,
    SCAN_INTERVAL_CRYPTO,
    SCAN_INTERVAL_FOREX,
    SCAN_INTERVAL_INDICES,
    SCAN_INTERVAL_METALS,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)


class MarketScanner:
    """Continuously scans the market for new trading signals and manages active trades."""

    def __init__(self, engine):
        self.engine = engine
        self.cached_dfs = {}
        self._active_signals_lock = asyncio.Lock()
        self._currently_scanning = set()

    async def _process_batch(self, symbols: list) -> None:
        """Processes a batch of symbols to find trading signals."""
        if self.engine.system_status != "IDLE":
            return

        signals_found = 0

        for sym in symbols:
            if not self.engine.is_running or signals_found >= MAX_SIGNALS_PER_SCAN:
                break

            # 1. State Validation Lock
            async with self._active_signals_lock:
                if sym in self._currently_scanning:
                    continue

                active_syms = [s.get("symbol") for s in self.engine.active_signals]

                if sym == "ETHUSDm" and any("BTC" in s for s in active_syms):
                    continue
                if sym == "BTCUSDm" and any("ETH" in s for s in active_syms):
                    continue

                if sym in active_syms:
                    continue

                # Mark as scanning to block concurrent overlapping batch intervals
                self._currently_scanning.add(sym)

            try:
                klines = await self.engine.provider.fetch_klines(sym, TIMEFRAME, CANDLE_LIMIT)
                if klines:
                    df = self.engine.ai_engine.prepare_data(klines)
                    if df is not None and not df.empty:
                        self.cached_dfs[sym] = df.tail(2)

                    signal = await self.engine.ai_engine.analyze_market(sym, klines, self.engine.provider)
                    if signal:
                        signal.setdefault("symbol", sym)
                        signal["detected_at"] = time.time()
                        signals_found += 1

                        async with self._active_signals_lock:
                            self.engine.active_signals.append(signal)

                        # 1. Alert via Telegram
                        if hasattr(self.engine, "notifier"):
                            self.engine.notifier.send_signal_alert(sym, signal)

                        # 2. Start monitoring the trade for TP/SL
                        self.engine.monitored_tasks[sym] = asyncio.create_task(
                            self.engine.monitor.verify_trade_realtime(sym, signal)
                        )
            finally:
                # Always release the scanning lock, even if MT5 fails or throws an error
                async with self._active_signals_lock:
                    if sym in self._currently_scanning:
                        self._currently_scanning.remove(sym)

    async def _refresh_market_watch_symbols(self) -> None:
        """Fetches the latest list of symbols from the provider and updates the engine's active lists."""
        data = await self.engine.provider.get_dynamic_symbols()
        self.engine.active_crypto_list = data.get("crypto", [])
        self.engine.active_forex_list = data.get("forex", [])
        self.engine.active_indices_list = data.get("indices", [])
        self.engine.active_metals_list = data.get("metals", [])

        if not any(
            [
                self.engine.active_crypto_list,
                self.engine.active_forex_list,
                self.engine.active_indices_list,
                self.engine.active_metals_list,
            ]
        ):
            self.engine.active_crypto_list = list(FALLBACK_CRYPTO)
            self.engine.active_forex_list = list(FALLBACK_FOREX)
            self.engine.active_indices_list = list(FALLBACK_INDICES)
            self.engine.active_metals_list = list(FALLBACK_METALS)

    async def _sort_pairs(self, symbols: list) -> List:
        """Sorts symbols by volatility using the AI engine's ranking function."""
        ranked = self.engine.ai_engine.rank_symbols_by_volatility(symbols, self.cached_dfs)
        result = ranked + [s for s in symbols if s not in ranked]
        return result

    async def scanner_loop(self) -> None:
        """Main loop for continuously scanning the market for new signals and managing active trades."""
        logger.info("🚀 Scanner Loop Initiated...")

        last_crypto, last_forex, last_indices, last_metals, last_sort_time = 0, 0, 0, 0, 0
        await self._refresh_market_watch_symbols()

        while self.engine.is_running:
            try:
                # If training, pause scanning
                if self.engine.system_status != "IDLE":
                    await asyncio.sleep(2)
                    continue

                # Fail-safe: Check if MT5 is actually alive
                if not self.engine.provider.connected:
                    logger.warning("⚠️ MT5 Disconnected. Pausing scanner...")
                    await asyncio.sleep(5)
                    continue

                # 1. Update Balance context
                acct = await self.engine.provider.get_account_summary()
                if acct:
                    self.engine.ai_engine.set_context(acct["balance"], self.engine.db, acct.get("currency", "USD"))

                now = time.time()

                # 2. Re-rank pairs by volatility every 15 minutes
                if now - last_sort_time > 900:
                    logger.info("📊 Re-ranking pairs by volatility...")

                    # Prevent spamming Telegram during dead market hours
                    session_status = self.engine.ai_engine.get_session_status()
                    if session_status.get("allow_trade", True):
                        self.engine.notifier.send_message(
                            "📊 *Market Scanner Update*\nRe-ranking pairs by volatility and hunting for high-probability setups..."
                        )

                    await self._refresh_market_watch_symbols()

                    # Apply sorts directly to the engine lists so _process_batch utilizes the ranking
                    self.engine.active_crypto_list = await self._sort_pairs(self.engine.active_crypto_list)
                    self.engine.active_forex_list = await self._sort_pairs(self.engine.active_forex_list)
                    self.engine.active_indices_list = await self._sort_pairs(self.engine.active_indices_list)
                    self.engine.active_metals_list = await self._sort_pairs(self.engine.active_metals_list)

                    last_sort_time = now

                # 3. Batch Process (Parallel Scanning)
                tasks = []

                if now - last_crypto > SCAN_INTERVAL_CRYPTO:
                    tasks.append(self._process_batch(self.engine.active_crypto_list[:5]))
                    last_crypto = now

                if now - last_forex > SCAN_INTERVAL_FOREX:
                    tasks.append(self._process_batch(self.engine.active_forex_list[:10]))
                    last_forex = now

                if now - last_indices > SCAN_INTERVAL_INDICES:
                    tasks.append(self._process_batch(self.engine.active_indices_list[:5]))
                    last_indices = now

                if now - last_metals > SCAN_INTERVAL_METALS:
                    tasks.append(self._process_batch(self.engine.active_metals_list[:5]))
                    last_metals = now

                if tasks:
                    await asyncio.gather(*tasks)

                await asyncio.sleep(1)

            except RuntimeError as e:
                if "shutdown" in str(e) or "closed" in str(e):
                    logger.info("🛑 Scanner loop stopping due to shutdown.")
                    break
                logger.error(f"Scanner Runtime Error: {e}")
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(5)
