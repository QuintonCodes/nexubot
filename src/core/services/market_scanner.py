import asyncio
import logging
import time

from src.config import (
    CANDLE_LIMIT,
    FALLBACK_CRYPTO,
    FALLBACK_FOREX,
    MAX_SIGNALS_PER_SCAN,
    SCAN_INTERVAL_CRYPTO,
    SCAN_INTERVAL_FOREX,
    TIMEFRAME,
)

logger = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self, engine):
        self.engine = engine

    async def _process_batch(self, symbols: list):
        """Processes a list of symbols concurrently."""
        if self.engine.system_status != "IDLE":
            return

        signals_found = 0

        for sym in symbols:
            if not self.engine.is_running:
                break

            # Limit signals per batch
            if signals_found >= MAX_SIGNALS_PER_SCAN:
                break

            # Skip if already active in UI
            if any(s.get("symbol") == sym for s in self.engine.active_signals):
                continue

            klines = await self.engine.provider.fetch_klines(sym, TIMEFRAME, CANDLE_LIMIT)
            if klines:
                signal = await self.engine.ai_engine.analyze_market(sym, klines, self.engine.provider)
                if signal:
                    signal.setdefault("symbol", sym)
                    signal["detected_at"] = time.time()
                    signal["status"] = "PENDING"  # Initial status

                    is_shadow = signal.get("is_shadow", False)

                    if not is_shadow:
                        signals_found += 1
                        self.engine.active_signals.insert(0, signal)

                        if self.engine.execution_mode == "FULL_AUTO":
                            placed = await self.engine.provider.execute_trade_on_mt5(signal)
                            if placed:
                                signal["status"] = "PLACED"
                        self.engine.monitored_tasks[sym] = asyncio.create_task(
                            self.engine.monitor.verify_trade_realtime(sym, signal)
                        )
                    else:
                        asyncio.create_task(self.engine.monitor.verify_trade_realtime(sym, signal))

    async def _refresh_market_watch_symbols(self):
        """Fetches current Market Watch from Provider."""
        data = await self.engine.provider.get_dynamic_symbols()
        self.engine.active_crypto_list = data.get("crypto", [])
        self.engine.active_forex_list = data.get("forex", [])

        if not self.engine.active_crypto_list and not self.engine.active_forex_list:
            self.engine.active_crypto_list = list(FALLBACK_CRYPTO)
            self.engine.active_forex_list = list(FALLBACK_FOREX)

    async def _sort_pairs(self, symbols: list) -> list:
        """Fetches 100 candles for all pairs to rank them by volatility (Ported from console.py)."""
        data_map = {}
        for sym in symbols:
            # Quick fetch
            k = await self.engine.provider.fetch_klines(sym, TIMEFRAME, 100)
            if k:
                df = self.engine.ai_engine.prepare_data(k, heavy=False)
                if df is not None:
                    data_map[sym] = df

        ranked = self.engine.ai_engine.rank_symbols_by_volatility(symbols, data_map)
        result = ranked + [s for s in symbols if s not in ranked]
        return result

    async def scanner_loop(self):
        """Background loop to scan markets."""
        logger.info("🚀 Scanner Loop Initiated...")

        active_crypto = []
        active_forex = []
        last_crypto = 0
        last_forex = 0
        last_sort_time = 0

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
                    self.engine.ai_engine.set_context(acct["balance"], self.engine.db)

                now = time.time()

                # 2. Sort Pairs every 15 mins (Logic from console.py)
                if now - last_sort_time > 900:
                    logger.info("📊 Re-ranking pairs by volatility...")
                    await self._refresh_market_watch_symbols()

                    active_crypto = await self._sort_pairs(active_crypto)
                    active_forex = await self._sort_pairs(active_forex)
                    last_sort_time = now

                # 3. Batch Process (Parallel Scanning)
                tasks = []

                if now - last_crypto > SCAN_INTERVAL_CRYPTO:
                    tasks.append(self._process_batch(self.engine.active_crypto_list[:5]))
                    last_crypto = now

                if now - last_forex > SCAN_INTERVAL_FOREX:
                    tasks.append(self._process_batch(self.engine.active_forex_list[:10]))
                    last_forex = now

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
