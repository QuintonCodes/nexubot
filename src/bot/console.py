import asyncio
import logging
import math
import sys
import time
from datetime import datetime

from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ai_engine import AITradingEngine
from src.config import (
    ALL_SYMBOLS,
    APP_NAME,
    CANDLE_LIMIT,
    CRYPTO_SYMBOLS,
    DEFAULT_BALANCE_ZAR,
    FOREX_SYMBOLS,
    GLOBAL_SIGNAL_COOLDOWN,
    MAX_SIGNALS_PER_SCAN,
    SCAN_INTERVAL_CRYPTO,
    SCAN_INTERVAL_FOREX,
    TIMEFRAME,
    VERSION,
)

logger = logging.getLogger(__name__)

# Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
MAGENTA = "\033[95m"
GRAY = "\033[90m"


class NexubotConsole:
    """
    Terminal Interface for Nexubot.
    """

    def __init__(self):
        self.ai_engine = AITradingEngine()
        self.db = DatabaseManager()
        self.last_global_signal_time = 0
        self.provider = DataProvider()
        self.running = False
        self.session_id = f"SESSION_{int(time.time())}"
        self.stats = {
            "wins": 0,
            "loss": 0,
            "total": 0,
            "pnl_zar": 0.0,
            "start": datetime.now(),
        }
        self.symbol_digits = {}
        self.tasks = set()

    async def _cache_digits(self):
        """Caches the decimal digits for all symbols."""
        for sym in ALL_SYMBOLS:
            info = await self.provider.get_symbol_info(sym)
            if info:
                self.symbol_digits[sym] = info.get("digits", 5)

    def _calculate_offline_result(self, symbol: str, signal: dict, start_time: float, klines):
        """Synchronous CPU-bound calculation logic for offline verification."""
        if not klines:
            return None, 0.0, False

        sl = signal["sl"]
        tp = signal["tp"]
        entry = signal["price"]
        is_long = signal["direction"] == "LONG"
        order_type = signal.get("order_type", "MARKET")
        trade_duration = 14400

        outcome = None
        pnl = 0.0
        filled_offline = order_type == "MARKET"

        for k in klines:
            if k["time"] < start_time:
                continue

            if (k["time"] - start_time) > trade_duration:
                outcome = "TIMEOUT (Offline)"
                break

            if not filled_offline:
                if is_long:
                    if k["low"] <= entry:
                        filled_offline = True
                else:
                    if k["high"] >= entry:
                        filled_offline = True
                if filled_offline:
                    continue

            if filled_offline:
                if is_long:
                    if k["low"] <= sl:
                        outcome = "LOSS (SL Offline)"
                        pnl = -signal["risk_zar"]
                        break
                    if k["high"] >= tp:
                        outcome = "WIN (TP Offline)"
                        pnl = signal["profit_zar"]
                        break
                else:
                    if k["high"] >= sl:
                        outcome = "LOSS (SL Offline)"
                        pnl = -signal["risk_zar"]
                        break
                    if k["low"] <= tp:
                        outcome = "WIN (TP Offline)"
                        pnl = signal["profit_zar"]
                        break

        return outcome, pnl, filled_offline

    async def _verify_offline_outcome(self, symbol: str, signal: dict, start_time: float) -> bool:
        """
        Checks if a trade that existed while offline has finished.
        """
        print(f"{MAGENTA}Checking offline history for {symbol}...{RESET}")

        now = time.time()
        elapsed = now - start_time
        candles_needed = math.ceil(elapsed / 900) + 5
        candles_needed = min(candles_needed, 1000)

        klines = await self.provider.fetch_klines(symbol, TIMEFRAME, candles_needed)

        if not klines:
            # If we can't get data, we assume we can't verify, so we delete if old
            if elapsed > 14400:
                await self.db.delete_active_trade(symbol)
                return True
            return False  # Try to resume if data missing but time remains?

        outcome, pnl, filled_offline = await asyncio.to_thread(
            self._calculate_offline_result, symbol, signal, start_time, klines
        )

        # Case 1: Trade finished (TP or SL hit)
        if outcome and "TIMEOUT" not in outcome:
            won = pnl > 0
            color = GREEN if pnl > 0 else RED
            print(f"{BOLD}🔔 Offline Result ({symbol}): {color}{outcome}{RESET} | PnL: {color}R{pnl:.2f}{RESET}")

            await self.db.log_trade(
                {
                    "id": f"{symbol}_OFFLINE_{int(time.time())}",
                    "symbol": symbol,
                    "signal": signal["signal"],
                    "confidence": signal["confidence"],
                    "entry": signal["price"],
                    "exit": signal["tp"] if won else signal["sl"],
                    "won": won,
                    "pnl": pnl,
                    "strategy": signal["strategy"] + " (Offline)",
                }
            )

            # Update Stats
            if won:
                self.stats["wins"] += 1
            else:
                self.stats["loss"] += 1
            self.stats["total"] += 1
            self.stats["pnl_zar"] += pnl

            await self.db.delete_active_trade(symbol)
            return True

        # Case 2: Trade Timed Out (Duration exceeded)
        if outcome == "TIMEOUT (Offline)" or elapsed > 14400:
            # If we never filled, it's a Cancel
            if not filled_offline:
                print(f"{GRAY}🚫 Offline Result ({symbol}): CANCELLED (Never Filled){RESET}")
                await self.db.delete_active_trade(symbol)
                return True

            # If filled, calculate floating PnL based on last known price (latest candle close)
            last_close = klines[-1]["close"]
            tick_value = signal.get("tick_value", 0.0)
            lot_size = signal["lot_size"]
            point = signal.get("point", 0.00001)

            entry = signal["price"]
            is_long = signal["direction"] == "LONG"
            if is_long:
                points_diff = (last_close - entry) / point
            else:
                points_diff = (entry - last_close) / point

            final_pnl = points_diff * tick_value * lot_size
            outcome_str = "WIN (Timeout)" if final_pnl > 0 else "LOSS (Timeout)"
            color = GREEN if final_pnl > 0 else RED

            print(
                f"{BOLD}🔔 Offline Result ({symbol}): {color}{outcome_str}{RESET} | PnL: {color}R{final_pnl:.2f}{RESET}"
            )

            # Log as a closed trade
            await self.db.log_trade(
                {
                    "id": f"{symbol}_TIMEOUT_{int(time.time())}",
                    "symbol": symbol,
                    "signal": signal["signal"],
                    "confidence": signal["confidence"],
                    "entry": entry,
                    "exit": last_close,
                    "won": final_pnl > 0,
                    "pnl": final_pnl,
                    "strategy": signal["strategy"] + " (Timeout)",
                }
            )

            self.stats["pnl_zar"] += final_pnl
            self.stats["total"] += 1

            await self.db.delete_active_trade(symbol)
            return True

        # Case 3: Trade is still active
        return False

    def display_signal(self, symbol: str, signal: dict):
        """Displays a formatted trading signal in the console."""
        sys.stdout.write(f"\r{' ' *50}\r")

        direction = signal["direction"]
        color = GREEN if direction == "LONG" else RED
        icon = "📈" if direction == "LONG" else "📉"
        risk_badge = f"{MAGENTA}[HIGH VOLATILITY]{RESET}" if signal.get("is_high_risk") else ""

        entry_str = self.smart_round(symbol, signal["price"])
        sl_str = self.smart_round(symbol, signal["sl"])
        tp_str = self.smart_round(symbol, signal["tp"])

        # Formatted Output
        print(f"{WHITE}" + "=" * 60)
        print(f"{BOLD}{color}🚨 {signal['signal']} {icon} SIGNAL DETECTED {risk_badge}{RESET}")
        print(f"{WHITE}" + "=" * 60)

        # Section: Asset Info
        print(f"{BOLD}Asset:{RESET}        {symbol}")
        print(f"{BOLD}Strategy:{RESET}     {signal['strategy']}")
        print(f"{BOLD}Confidence:{RESET}   {signal['confidence']:.1f}%")
        print(f"{WHITE}" + "-" * 60)

        # Section: Execution
        print(f"{BOLD}ENTRY:{RESET}        {entry_str}")
        print(f"{RED}STOP LOSS:{RESET}    {sl_str}")
        print(f"{GREEN}TAKE PROFIT:{RESET}  {tp_str}")
        print(f"{WHITE}" + "-" * 60)

        # Section: Sizing
        print(f"{BOLD}LOT SIZE:{RESET}     {YELLOW}{signal['lot_size']:.2f} Lots{RESET}")
        print(f"{WHITE}" + "-" * 60)

        print(f"{BOLD}MAX RISK:{RESET}     {RED}-R{signal['risk_zar']:.2f}  (If SL Hit){RESET}")
        print(f"{BOLD}EST PROFIT:{RESET}   {GREEN}+R{signal['profit_zar']:.2f}  (If TP Hit){RESET}")
        print(f"{WHITE}" + "-" * 60)

        print(f"{BOLD}ACTION:{RESET} Open {signal['lot_size']:.2f} Lots {signal['signal']} on {symbol}{RESET}")
        print(f"{WHITE}" + "=" * 60 + "\n")

    def print_dashboard(self):
        """Displays session statistics."""
        elapsed = datetime.now() - self.stats["start"]
        win_rate = (self.stats["wins"] / self.stats["total"] * 100) if self.stats["total"] > 0 else 0
        current_pnl = self.stats["pnl_zar"]
        pnl_color = GREEN if current_pnl >= 0 else RED

        dash = f"""
            {CYAN}╔════════════════════ SESSION DASHBOARD ════════════════════╗{RESET}
            {CYAN}║{RESET} Time Running: {str(elapsed).split('.')[0]}   |   Total Signals: {self.stats['total']}
            {CYAN}║{RESET} Win Rate: {BOLD}{win_rate:.1f}%{RESET}          |   Wins: {GREEN}{self.stats['wins']}{RESET} / Loss: {RED}{self.stats['loss']}{RESET}
            {CYAN}║{RESET} Session PnL: {pnl_color}R{current_pnl:.2f}{RESET} (Est){RESET}
            {CYAN}╚═══════════════════════════════════════════════════════════╝{RESET}
        """
        sys.stdout.write(f"\r{' ' *80}\r")
        print(dash)

    async def process_batch(self, symbols: list[str]):
        """Processes a batch of symbols for signals."""
        signals_found = 0
        for sym in symbols:
            # Check Rate Limits
            if time.time() - self.last_global_signal_time < GLOBAL_SIGNAL_COOLDOWN:
                break
            if signals_found >= MAX_SIGNALS_PER_SCAN:
                break

            # Fetch
            klines = await self.provider.fetch_klines(sym, TIMEFRAME, CANDLE_LIMIT)
            if not klines:
                # NEW: WARN IF DATA IS MISSING (SILENCE DETECTOR)
                print(f"{YELLOW}⚠️ No data for {sym} (Check MT5 Connection){RESET}")
                continue

            # Analyze
            signal = await self.ai_engine.analyze_market(sym, klines, self.provider)
            if signal:
                is_shadow = signal.get("is_shadow", False)

                if not is_shadow:
                    self.display_signal(sym, signal)
                    signals_found += 1
                    self.last_global_signal_time = time.time()
                else:
                    # Silent log for shadow trades
                    print(f"{GRAY}👻 Shadow signal tracking started for {sym}...{RESET}")

                # Start verification task for BOTH real and shadow trades
                task = asyncio.create_task(self.verify_trade_realtime(sym, signal))
                self.tasks.add(task)

    async def start(self):
        """Initializes the bot and starts the scan loop."""
        self.running = True
        print(f"\n{BOLD}{CYAN}🚀 {APP_NAME} {VERSION}{RESET}")

        # Init DB
        await self.db.init_database()
        print(f"{YELLOW}🧹 Cleaning old database records...{RESET}")
        await self.db.cleanup_db()

        # Fetch and display Total Historical Win Rate
        total_wr = await self.db.get_total_historical_win_rate()
        print(f"{BOLD}{MAGENTA}🏆 Lifetime Win Rate: {total_wr:.1f}%{RESET}")

        # Init MT5 via Provider
        print(f"✅ {WHITE}Account Set: {GREEN}R{DEFAULT_BALANCE_ZAR:.2f}{RESET}")
        self.ai_engine.set_context(DEFAULT_BALANCE_ZAR, self.db)

        # Init MT5 via Provider
        print(f"{YELLOW}⏳ Initializing MT5 (Timeout 60s)... Please wait.{RESET}")
        if not await self.provider.initialize():
            print(f"{RED}Failed to initialize MT5 Provider. Check logs.{RESET}")
            return

        await self._cache_digits()

        # Restore active trades
        print(f"{YELLOW}🔄 Checking for interrupted trades...{RESET}")
        active_trades = await self.db.get_active_trades()

        if active_trades:
            print(f"{CYAN}Found {len(active_trades)} active trades. Resuming monitoring...{RESET}")
            for symbol, signal, start_time in active_trades:
                self.ai_engine.register_active_trade(symbol)
                is_finished = await self._verify_offline_outcome(symbol, signal, start_time)

                if not is_finished:
                    elapsed = time.time() - start_time
                    remaining = 14400 - elapsed  # 4 Hours

                    if remaining > 0:
                        print(f"Resuming {symbol} (Time remaining: {remaining/60:.0f}m)")
                        task = asyncio.create_task(
                            self.verify_trade_realtime(symbol, signal, resume_start_time=start_time)
                        )
                        self.tasks.add(task)
                    else:
                        # Expired while offline
                        await self.db.delete_active_trade(symbol)

        print(f"{YELLOW}📊 Ranking pairs by volatility...{RESET}")
        await self.scan_loop()

    async def scan_loop(self):
        """Main infinite loop."""
        print(f"{GREEN}✅ Scanner Active. Monitoring {len(ALL_SYMBOLS)} pairs...{RESET}\n")

        last_crypto = 0
        last_forex = 0
        last_sort_time = 0
        last_heartbeat = time.time()

        active_crypto = list(CRYPTO_SYMBOLS)
        active_forex = list(FOREX_SYMBOLS)

        try:
            while self.running:
                # Circuit breaker
                if self.stats["pnl_zar"] < -(DEFAULT_BALANCE_ZAR * 0.10):
                    print(f"{RED}🛑 CIRCUIT BREAKER: Session Drawdown > 10%. Stopping Bot.{RESET}")
                    self.running = False
                    return

                now = time.time()

                # Prints every 15 minutes if no other signals have appeared
                if now - last_heartbeat > 900:
                    print(
                        f"{GRAY}💓 [{datetime.now().strftime('%H:%M')}] Scanner is alive. Monitoring markets...{RESET}"
                    )
                    last_heartbeat = now

                if now - last_sort_time > 900:
                    active_crypto = await self.sort_pairs(active_crypto)
                    active_forex = await self.sort_pairs(active_forex)
                    last_sort_time = now

                tasks = []
                # Crypto Scan
                if now - last_crypto > SCAN_INTERVAL_CRYPTO:
                    tasks.append(self.process_batch(active_crypto))
                    last_crypto = now

                # Forex Scan
                if now - last_forex > SCAN_INTERVAL_FOREX:
                    tasks.append(self.process_batch(active_forex))
                    last_forex = now

                if tasks:
                    await asyncio.gather(*tasks)

                # Cleanup
                self.tasks = {t for t in self.tasks if not t.done()}
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def sort_pairs(self, symbols: list[str]) -> list[str]:
        """Fetches 100 candles for all pairs to rank them."""
        data_map = {}
        for sym in symbols:
            # Quick fetch
            k = await self.provider.fetch_klines(sym, TIMEFRAME, 100)
            if k:
                df = self.ai_engine._prepare_data(k, heavy=False)
                if df is not None:
                    data_map[sym] = df

        ranked = self.ai_engine.rank_symbols_by_volatility(symbols, data_map)
        result = ranked + [s for s in symbols if s not in ranked]
        return result

    def smart_round(self, symbol: str, value: float) -> str:
        """Rounds value based on symbol's decimal digits."""
        digits = self.symbol_digits.get(symbol, 5)
        return f"{value:.{digits}f}"

    async def stop(self):
        """Gracefully stops the bot and cleans up."""
        self.running = False
        print(f"\n{YELLOW}🛑 Stopping Nexubot...{RESET}")

        # Cancel all pending monitoring tasks
        for task in self.tasks:
            task.cancel()

        # Wait briefly for tasks to clean up
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        await self.db.log_session(self.session_id, self.stats["start"].timestamp(), self.stats)
        await self.db.close()
        await self.provider.shutdown()

    async def verify_trade_realtime(self, symbol: str, signal: dict, resume_start_time=None):
        """
        Monitors price and Logs Data for ML.
        Uses True Tick Value for accurate PnL calculation.
        """
        is_shadow = signal.get("is_shadow", False)
        if not is_shadow:
            print(f"{CYAN}👀 Monitoring trade {symbol} for outcome...{RESET}")
            await self.db.save_active_trade(symbol, signal)

        entry = signal["price"]
        sl = signal["sl"]
        tp = signal["tp"]
        is_long = signal["direction"] == "LONG"
        atr = signal.get("atr", 1.0)
        lot_size = signal["lot_size"]
        tick_value = signal.get("tick_value", 0.0)
        point = signal.get("point", 0.00001)

        # Order Management
        order_type = signal.get("order_type", "MARKET")
        is_filled = order_type == "MARKET"

        # Trailer State
        be_stage = 0  # 0=None, 1=BE, 2=Lock 1R, 3=Lock 2R
        duration = 14400  # 4 hours
        start_time = resume_start_time if resume_start_time else time.time()
        interval = 1  # Check every second

        outcome = "TIMEOUT"
        final_pnl = 0.0
        max_favorable_dist = 0.0

        try:
            while (time.time() - start_time) < duration:
                # Check if shutting down
                if not self.running:
                    return

                # Get Tick for Spread Logic
                tick = await self.provider.get_current_tick(symbol)
                if not tick:
                    await asyncio.sleep(interval)
                    continue

                current_bid = tick.bid
                current_ask = tick.ask
                # Spread is crucial for fill simulation
                spread = current_ask - current_bid

                # --- 1. GHOST ORDER LOGIC (WAIT FOR FILL) ---
                if not is_filled:
                    if is_long:
                        if current_ask <= (entry - spread):  # Ask price hits entry
                            is_filled = True
                            if not is_shadow:
                                print(f"{YELLOW}⚡ {symbol} Ghost Limit BUY Filled at {entry}{RESET}")
                    else:
                        if current_bid >= (entry + spread):  # Bid price hits entry
                            is_filled = True
                            if not is_shadow:
                                print(f"{YELLOW}⚡ {symbol} Ghost Limit SELL Filled at {entry}{RESET}")

                    # Runaway cancellation
                    dist_away = (current_ask - entry) if is_long else (entry - current_bid)
                    if dist_away > (atr * 2):
                        outcome = "CANCELLED (Runaway)"
                        break

                    await asyncio.sleep(interval)
                    continue

                # --- 2. TRADE MONITORING (FILLED) ---
                if is_long:
                    # Favorable direction is Up (Bid)
                    dist = current_bid - entry
                    if dist > max_favorable_dist:
                        max_favorable_dist = dist

                    # SL Logic (Sell at Bid)
                    if current_bid <= sl:
                        outcome = "LOSS (SL Hit)"
                        points_lost = (sl - entry) / point
                        final_pnl = points_lost * tick_value * lot_size
                        break
                    # TP Logic (Sell at Bid)
                    elif current_bid >= tp:
                        outcome = "WIN (TP Hit)"
                        points_won = (tp - entry) / point
                        final_pnl = points_won * tick_value * lot_size
                        break

                    # --- MULTI-STAGE TRAILING (Only for real trades) ---
                    if not is_shadow:
                        if be_stage < 3 and max_favorable_dist > (atr * 3.0):
                            new_sl = entry + (atr * 2.0)
                            if new_sl > sl:
                                sl = new_sl
                                be_stage = 3
                                print(f"{CYAN}🛡️ {symbol} Locked 2R Profit{RESET}")

                        # Stage 2: Lock 1R if > 2R
                        elif be_stage < 2 and max_favorable_dist > (atr * 2.0):
                            new_sl = entry + (atr * 1.0)
                            if new_sl > sl:
                                sl = new_sl
                                be_stage = 2
                                print(f"{CYAN}🛡️ {symbol} Locked 1R Profit{RESET}")

                        # Stage 1: Breakeven if > 1R
                        elif be_stage < 1 and max_favorable_dist > (atr * 1.0):
                            new_sl = entry + (20 * point)  # Slight buffer
                            if new_sl > sl:
                                sl = new_sl
                                be_stage = 1
                                print(f"{CYAN}🛡️ {symbol} SL Moved to Breakeven{RESET}")
                else:  # Short
                    # Favorable direction is Down (Ask)
                    dist = entry - current_ask
                    if dist > max_favorable_dist:
                        max_favorable_dist = dist

                    # SL Logic (Buy at Ask)
                    if current_ask >= sl:
                        outcome = "LOSS (SL Hit)"
                        points_lost = (entry - sl) / point
                        final_pnl = points_lost * tick_value * lot_size
                        break
                    # TP Logic (Buy at Ask)
                    elif current_ask <= tp:
                        outcome = "WIN (TP Hit)"
                        points_won = (entry - tp) / point
                        final_pnl = points_won * tick_value * lot_size
                        break

                    # --- MULTI-STAGE TRAILING ---
                    if not is_shadow:
                        if be_stage < 3 and max_favorable_dist > (atr * 3.0):
                            new_sl = entry - (atr * 2.0)
                            if new_sl < sl:
                                sl = new_sl
                                be_stage = 3
                                print(f"{CYAN}🛡️ {symbol} Locked 2R Profit{RESET}")

                        elif be_stage < 2 and max_favorable_dist > (atr * 2.0):
                            new_sl = entry - (atr * 1.0)
                            if new_sl < sl:
                                sl = new_sl
                                be_stage = 2
                                print(f"{CYAN}🛡️ {symbol} Locked 1R Profit{RESET}")

                        elif be_stage < 1 and max_favorable_dist > (atr * 1.0):
                            new_sl = entry - (20 * point)
                            if new_sl < sl:
                                sl = new_sl
                                be_stage = 1
                                print(f"{CYAN}🛡️ {symbol} SL Moved to Breakeven{RESET}")

                await asyncio.sleep(interval)

            # Cleanup Persistence
            if not is_shadow:
                await self.db.delete_active_trade(symbol)

            # Outcome calculation
            if outcome == "TIMEOUT":
                tick = await self.provider.get_current_tick(symbol)
                if tick:
                    if is_long:
                        points_diff = (tick.bid - entry) / point
                        final_pnl = points_diff * tick_value * lot_size
                    else:
                        points_diff = (entry - tick.ask) / point
                        final_pnl = points_diff * tick_value * lot_size

                outcome = "WIN (Floating)" if final_pnl > 0 else "LOSS (Floating)"
            elif outcome == "CANCELLED (Runaway)":
                final_pnl = 0.0

            # Log Result
            won = final_pnl > 0
            excursion_atr = max_favorable_dist / atr if atr > 0 else 0.0

            if not is_shadow:
                self.stats["pnl_zar"] += final_pnl
                if is_filled:
                    if won:
                        self.stats["wins"] += 1
                    else:
                        self.stats["loss"] += 1
                    self.stats["total"] += 1

                color = GREEN if won else RED
                print(f"\n{BOLD}🔔 Result ({symbol}): {color}{outcome}{RESET} | PnL: {color}R{final_pnl:.2f}{RESET}\n")
                self.print_dashboard()

                if is_filled:
                    await self.db.log_trade(
                        {
                            "id": f"{symbol}_{int(time.time())}",
                            "symbol": symbol,
                            "signal": signal["signal"],
                            "confidence": signal["confidence"],
                            "entry": entry,
                            "exit": tp if "TP" in outcome else sl if "SL" in outcome else entry,
                            "won": won,
                            "pnl": final_pnl,
                            "strategy": signal["strategy"],
                        }
                    )
            else:
                color = GREEN if won else RED
                print(f"{GRAY}👻 Shadow Result ({symbol}): {color}{outcome}{RESET} (Virtual)")

            if is_filled:
                self.ai_engine.record_trade_outcome(symbol, won, final_pnl, excursion_atr, is_shadow)

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Error verifying {symbol}: {e}")
