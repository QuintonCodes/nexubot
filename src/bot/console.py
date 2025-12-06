import asyncio
import aiohttp
import time
import logging
import sys
import socket
from datetime import datetime

from src.database.manager import DatabaseManager
from src.engine.ai_engine import AITradingEngine
from src.data.provider import DataProvider
from src.config import (
    ALL_SYMBOLS, CRYPTO_SYMBOLS, FOREX_SYMBOLS,
    SCAN_INTERVAL_CRYPTO, SCAN_INTERVAL_FOREX,
    GLOBAL_SIGNAL_COOLDOWN, MAX_SIGNALS_PER_SCAN,
    DEFAULT_BALANCE_ZAR, SMALL_ACCOUNT_THRESHOLD_ZAR,
    APP_NAME, VERSION, TIMEFRAME, CANDLE_LIMIT,
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

class NexubotConsole:
    """
    Terminal Interface for Nexubot.
    """
    def __init__(self):
        self.db = DatabaseManager()
        self.ai_engine = AITradingEngine()
        self.provider = None
        self.session = None
        self.running = False
        self.last_scan = {'crypto': 0, 'forex': 0}
        self.last_global_signal_time = 0
        self.session_id = f"SESSION_{int(time.time())}"
        self.stats = {
            'wins': 0, 'loss': 0, 'total': 0,
            'pnl_zar': 0.0, 'start': datetime.now()
        }

    async def start(self):
        """Initializes the bot and starts the scan loop."""
        self.running = True
        print(f"\n{BOLD}{CYAN}🚀 {APP_NAME} {VERSION}{RESET}")

        await self.db.init_database()
        self.ai_engine.set_context(0, 18.00, self.db)

        try:
            print(f"{WHITE}Enter Account Balance (ZAR) [Default: R{DEFAULT_BALANCE_ZAR}]: {RESET}", end="")
            user_input = input()
            zar_balance = float(user_input) if user_input.strip() else DEFAULT_BALANCE_ZAR
        except ValueError:
            zar_balance = DEFAULT_BALANCE_ZAR

        self.ai_engine.user_balance_zar = zar_balance
        print(f"✅ Account Set: {GREEN}R{zar_balance:.2f}{RESET}")

        if zar_balance < SMALL_ACCOUNT_THRESHOLD_ZAR:
            print(f"{YELLOW}⚠️ Small Account Logic Enabled (Survival Mode){RESET}")

        nameservers = ["1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4"]

        # --- Network Setup (Google DNS) ---
        resolver = aiohttp.AsyncResolver(nameservers=nameservers)
        connector = aiohttp.TCPConnector(
            family=socket.AF_INET,
            ssl=False,
            limit=100,
            resolver=resolver,
            keepalive_timeout=30,
        )
        self.session = aiohttp.ClientSession(connector=connector)
        self.provider = DataProvider(self.session)

        await self.scan_loop()

    def print_dashboard(self):
        """Displays session statistics."""
        elapsed = datetime.now() - self.stats['start']
        win_rate = (self.stats['wins']/self.stats['total']*100) if self.stats['total'] > 0 else 0

        current_pnl = self.stats['pnl_zar']
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

    async def scan_loop(self):
        """Main infinite loop."""
        print(f"{GREEN}✅ Scanner Started. Monitoring {len(ALL_SYMBOLS)} pairs...{RESET}\n")

        while self.running:
            now = time.time()
            rate = await self.provider.get_usd_zar()
            self.ai_engine.usd_zar_rate = rate

            tasks = []

            # 1. Crypto Scan
            if now - self.last_scan['crypto'] > SCAN_INTERVAL_CRYPTO:
                tasks.append(self.process_batch(CRYPTO_SYMBOLS, "Crypto"))
                self.last_scan['crypto'] = now

            # 2. Forex Scan
            if now - self.last_scan['forex'] > SCAN_INTERVAL_FOREX:
                tasks.append(self.process_batch(FOREX_SYMBOLS, "Forex"))
                self.last_scan['forex'] = now

            if tasks:
                await asyncio.gather(*tasks)

            # Dashboard Update
            if int(time.time()) % 300 == 0: self.print_dashboard()
            await asyncio.sleep(1)

    async def process_batch(self, symbols, label):
        sys.stdout.write(f"\r{CYAN}⚡ Scanning {label}... {datetime.now().strftime('%H:%M:%S')}{RESET}\n")
        sys.stdout.flush()

        signals_found = 0
        for sym in symbols:
            # Check Rate Limits
            if time.time() - self.last_global_signal_time < GLOBAL_SIGNAL_COOLDOWN: break
            if signals_found >= MAX_SIGNALS_PER_SCAN: break

            # Fetch
            klines = await self.provider.fetch_klines(sym, TIMEFRAME, CANDLE_LIMIT)
            if not klines: continue

            # Analyze
            signal = await self.ai_engine.analyze_market(sym, klines)
            if signal:
                self.display_signal(sym, signal)
                signals_found += 1
                self.last_global_signal_time = time.time()

            await asyncio.sleep(0.5)

    def display_signal(self, symbol: str, signal: dict):
        sys.stdout.write(f"\r{' ' *50}\r")

        direction = signal['direction']
        color = GREEN if direction == 'LONG' else RED
        icon = "📈" if direction == 'LONG' else "📉"
        risk_badge = f"{MAGENTA}[HIGH VOLATILITY]{RESET}" if signal.get('is_high_risk') else ""

        # Formatted Output
        print(f"{WHITE}" + "="*60)
        print(f"{BOLD}{color}🚨 {signal['signal']} {icon} SIGNAL DETECTED {risk_badge}{RESET}")
        print(f"{WHITE}" + "="*60)

        # Section: Asset Info
        print(f"{BOLD}Asset:{RESET}        {symbol}")
        print(f"{BOLD}Strategy:{RESET}     {signal['strategy']}")
        print(f"{BOLD}Confidence:{RESET}   {signal['confidence']:.1f}%")
        print(f"{WHITE}" + "-"*60)

        # Section: Execution
        print(f"{BOLD}ENTRY:{RESET}        {signal['price']}")
        print(f"{RED}STOP LOSS:{RESET}    {signal['sl']}")
        print(f"{GREEN}TAKE PROFIT:{RESET}  {signal['tp']}")
        print(f"{WHITE}" + "-"*60)

        # Section: Sizing
        print(f"{BOLD}LOT SIZE:{RESET}     {YELLOW}{signal['lot_size']:.2f} Lots{RESET}")
        print(f"{WHITE}" + "-"*60)

        print(f"{BOLD}SPREAD COST:{RESET}  {RED}-R{signal['spread_cost_zar']:.2f}  (Immediate Drawdown){RESET}")
        print(f"{BOLD}MAX RISK:{RESET}     {RED}-R{signal['risk_zar']:.2f}  (If SL Hit){RESET}")
        print(f"{BOLD}EST PROFIT:{RESET}   {GREEN}+R{signal['profit_zar']:.2f}  (If TP Hit){RESET}")
        print(f"{WHITE}" + "-"*60)

        print(f"{BOLD}ACTION:{RESET} Open {signal['lot_size']:.2f} Lots {signal['signal']} on {symbol}{RESET}")
        print(f"{WHITE}" + "="*60 + "\n")

        # Launch verification task
        asyncio.create_task(self.verify_trade(symbol, signal))

    async def verify_trade(self, symbol: str, signal: dict):
        """Monitors price to see if trade would have won/lost."""
        await asyncio.sleep(300) # Wait 5 mins

        exit_price = await self.provider.get_latest_price(symbol)
        if not exit_price: return

        is_buy = signal['direction'] == 'LONG'
        won = (exit_price > signal['price']) if is_buy else (exit_price < signal['price'])

        # PnL logic
        pnl = signal['profit_zar'] if won else -signal['risk_zar']

        # Update Stats
        self.stats['total'] += 1
        if won: self.stats['wins'] += 1
        else: self.stats['loss'] += 1
        self.stats['pnl_zar'] += pnl

        await self.db.log_trade({
            'id': f"{symbol}_{int(time.time())}",
            'symbol': symbol,
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'entry': signal['price'],
            'exit': exit_price,
            'won': won,
            'pnl': pnl,
            'strategy': signal['strategy']
        })
        self.ai_engine.learn(symbol, signal['strategy'], won, pnl)

        res = f"{GREEN}WIN ✅{RESET}" if won else f"{RED}LOSS ❌{RESET}"
        print(f"\n{BOLD}📢 Trade Result ({symbol}):{RESET} {res} (R{pnl:.2f})\n")

    async def stop(self):
        self.running = False
        await self.db.log_session(
            self.session_id,
            self.stats['start'].timestamp(),
            self.stats
        )
        if self.session: await self.session.close()
        await self.db.close()
        self.ai_engine.save_models()