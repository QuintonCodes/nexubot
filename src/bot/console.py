import asyncio
import aiohttp
import time
import logging
import sys
import socket
from datetime import datetime

from src.database.manager import DatabaseManager
from src.engine.ai_engine import AITradingEngine
from src.config import (
    BINANCE_API_URL, USE_DYNAMIC_SYMBOLS, STATIC_SYMBOLS,
    SCAN_INTERVAL, CANDLE_LIMIT
)
from src.utils.market_data import fetch_top_volatile_symbols

logger = logging.getLogger(__name__)

# Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
WHITE = "\033[97m"

class NexubotConsole:
    def __init__(self):
        self.db = DatabaseManager()
        self.ai_engine = AITradingEngine(self.db)
        self.session = None
        self.running = False
        self.active_symbols = STATIC_SYMBOLS
        self.stats = {'wins': 0, 'loss': 0, 'total': 0}

    async def start(self):
        self.running = True

        resolver = aiohttp.AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])

        connector = aiohttp.TCPConnector(
            family=socket.AF_INET,
            ssl=False,
            limit=100,
            force_close=True,
            resolver=resolver
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True
        )

        print(f"\n{BOLD}{CYAN}🚀 NEXUBOT PROFESSIONAL (MT5 EDITION){RESET}")
        print(f"{YELLOW}Strategies: Trend Flow | Volatility Breakout | Reversal Scalp{RESET}")
        print(f"{YELLOW}Risk Mgmt: Active (Dynamic Lots & SL/TP){RESET}")

        if USE_DYNAMIC_SYMBOLS:
            print("🌊 Fetching top volatile markets from Binance...")
            dynamic_symbols = await fetch_top_volatile_symbols(self.session)
            if dynamic_symbols:
                self.active_symbols = dynamic_symbols
                print(f"✅ Monitoring: {', '.join(self.active_symbols)}\n")
            else:
                self.clean_print(f"{RED}⚠️ Scan failed. Reverting to static list.{RESET}")
                self.active_symbols = STATIC_SYMBOLS

        await self.scan_loop()

    def clean_print(self, message):
        """Helper to clear the scanning line and print a message"""
        # \r moves cursor to start, ' ' * 80 wipes line, \r moves back
        sys.stdout.write(f"\r{' ' * 80}\r")
        print(message)

    async def scan_loop(self):
        """Continuous scanning loop"""
        while self.running:
            try:
                start_time = time.time()

                scan_msg = f"\r{CYAN}🔄 Scanning {len(self.active_symbols)} markets... {datetime.now().strftime('%H:%M:%S')}{RESET}\n"
                sys.stdout.write(scan_msg)
                sys.stdout.flush()

                for symbol in self.active_symbols:
                    await self.analyze_pair(symbol)
                    await asyncio.sleep(0.1) # Rate limit protect

                elapsed = time.time() - start_time
                sleep_time = max(1, SCAN_INTERVAL - elapsed)

                # Update symbols every 30 minutes if dynamic
                if USE_DYNAMIC_SYMBOLS and time.time() % 1800 < SCAN_INTERVAL:
                    new_symbols = await fetch_top_volatile_symbols(self.session)
                    if new_symbols: self.active_symbols = new_symbols

                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.clean_print(f"{RED}Loop Error: {e}{RESET}")
                await asyncio.sleep(5)

    async def analyze_pair(self, symbol):
        try:
            url = f"{BINANCE_API_URL}/klines"
            params = {'symbol': symbol, 'interval': '5m', 'limit': CANDLE_LIMIT}

            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not isinstance(data, list):
                return

            signal = self.ai_engine.analyze_market(symbol, data)

            if signal:
                self.display_signal(symbol, signal)

        except Exception as e:
            if "getaddrinfo" not in str(e) and "Timeout" not in str(e):
                self.clean_print(f"{RED}Analysis failed for {symbol}: {e}{RESET}")

    def display_signal(self, symbol, signal):
        # Clear scanning line first
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()

        # Visual Alert
        color = GREEN if signal['signal'] == 'BUY' else RED
        type_txt = "LONG (BUY)" if signal['signal'] == 'BUY' else "SHORT (SELL)"

        print(f"\n\n{WHITE}" + "="*50)
        print(f"{BOLD}{color}🚨 {type_txt} SIGNAL DETECTED{RESET}")
        print(f"{WHITE}" + "="*50)

        print(f"{BOLD}Asset:{RESET}      {symbol}")
        print(f"{BOLD}Strategy:{RESET}   {signal['strategy']}")
        print(f"{BOLD}Confidence:{RESET} {signal['confidence']:.2f}%")
        print(f"{WHITE}" + "-"*50)

        # Price Levels
        print(f"{BOLD}ENTRY:{RESET}      {signal['price']}")
        print(f"{RED}STOP LOSS:{RESET}  {signal['sl']}")
        print(f"{GREEN}TAKE PROFIT:{RESET} {signal['tp']}")
        print(f"{WHITE}" + "-"*50)

        # Risk Management
        print(f"{BOLD}LOT SIZE:{RESET}   {signal['lot_size']} units")
        print(f"{BOLD}RISK:{RESET}       ${signal['risk_amount']:.2f}")
        print(f"{BOLD}EST PROFIT:{RESET} ${signal['est_profit']:.2f}")

        print(f"{WHITE}" + "="*50 + "\n")

        # Launch verification task
        asyncio.create_task(self.verify_trade(symbol, signal))

    async def verify_trade(self, symbol, signal):
        """Simple verification simulation (waiting for TP or SL hit)"""
        await asyncio.sleep(300) # Wait 5 mins for result (Simulation)
        try:
            url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol}"
            async with self.session.get(url) as resp:
                data = await resp.json()
                exit_price = float(data['price'])

            # Determine Win/Loss
            won = False
            if signal['signal'] == 'BUY':
                if exit_price > signal['price']: won = True
            else:
                if exit_price < signal['price']: won = True

            self.stats['total'] += 1
            if won: self.stats['wins'] += 1
            else: self.stats['loss'] += 1

            # Update AI Brain (Pickle)
            self.ai_engine.learn(symbol, signal['strategy'], won)

            # Update Database History (SQLite)
            self.save_trade_to_db(symbol, signal, exit_price, won)

            # Report to Console
            outcome = f"{GREEN}WIN ✅{RESET}" if won else f"{RED}LOSS ❌{RESET}"
            self.clean_print(f"{BOLD}Trade Update ({symbol}):{RESET} {outcome} (Entry: {signal['price']} -> Exit: {exit_price})")

        except Exception:
            pass

    def save_trade_to_db(self, symbol, signal, exit_price, won):
        """Saves the completed trade details to SQLite"""
        try:
            trade_id = f"{symbol}_{int(time.time())}"
            # Calculate PnL Pips (approximate)
            diff = exit_price - signal['price']
            pips = (diff / signal['price']) * 10000 if signal['price'] > 0 else 0

            self.db.log_trade(
                trade_id=trade_id,
                symbol=symbol,
                signal_type=signal['signal'],
                confidence=signal['confidence'],
                entry=signal['price'],
                exit_p=exit_price,
                won=won,
                pips=pips,
                strategy=signal['strategy']
            )
        except Exception as e:
            logger.error(f"DB Save Error: {e}")

    async def stop(self):
        self.running = False
        if self.session: await self.session.close()
        self.ai_engine.save_models()