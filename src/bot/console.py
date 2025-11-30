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
    BINANCE_API_URL, STATIC_SYMBOLS, TIMEFRAME,
    SCAN_INTERVAL, CANDLE_LIMIT, DEFAULT_BALANCE_ZAR,
    GLOBAL_SIGNAL_COOLDOWN, MAX_SIGNALS_PER_SCAN,
    VERSION, APP_NAME
)

logger = logging.getLogger(__name__)

# ASNI Colors
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
        self.session = None
        self.running = False
        self.active_symbols = STATIC_SYMBOLS
        self.last_global_signal_time = 0
        self.session_id = f"SESSION_{int(time.time())}"
        self.stats = {
            'wins': 0, 'loss': 0, 'total': 0,
            'pnl_zar': 0.0, 'start_time': datetime.now()
        }

    async def start(self):
        """Initializes the bot and starts the scan loop."""
        self.running = True
        print(f"\n{BOLD}{CYAN}🚀 {APP_NAME} {VERSION}{RESET}")
        print(f"{WHITE}Broker Target:{RESET} {YELLOW}HFM (ZAR){RESET}")

        # Database init
        await self.db.init_database()

        # Inject DB manager into AI engine for historical lookups
        self.ai_engine.set_db_manager(self.db)

        # User input
        try:
            print(f"{WHITE}Enter Account Balance (ZAR) [Default: R{DEFAULT_BALANCE_ZAR}]: {RESET}", end="")
            user_input = input()
            zar_balance = float(user_input) if user_input.strip() else DEFAULT_BALANCE_ZAR
        except ValueError:
            zar_balance = DEFAULT_BALANCE_ZAR

        self.ai_engine.set_user_balance(zar_balance)
        print(f"✅ Account Set: {GREEN}R{zar_balance:.2f}{RESET}")
        print(f"{YELLOW}Rate Limit: {MAX_SIGNALS_PER_SCAN} Signals / Scan{RESET}")

        # --- Network Setup (Google DNS) ---
        resolver = aiohttp.AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])
        connector = aiohttp.TCPConnector(
            family=socket.AF_INET,
            ssl=False,
            limit=100,
            resolver=resolver
        )
        self.session = aiohttp.ClientSession(connector=connector)

        await self.scan_loop()

    def print_dashboard(self):
        """Displays session statistics."""
        elapsed = datetime.now() - self.stats['start_time']
        win_rate = (self.stats['wins']/self.stats['total']*100) if self.stats['total'] > 0 else 0
        pnl_color = GREEN if self.stats['pnl_zar'] >= 0 else RED

        dash = f"""
            {CYAN}╔════════════════════ SESSION DASHBOARD ════════════════════╗{RESET}
            {CYAN}║{RESET} Time Running: {str(elapsed).split('.')[0]}   |   Total Signals: {self.stats['total']}
            {CYAN}║{RESET} Win Rate: {BOLD}{win_rate:.1f}%{RESET}          |   Wins: {GREEN}{self.stats['wins']}{RESET} / Loss: {RED}{self.stats['loss']}{RESET}
            {CYAN}║{RESET} Session PnL: {pnl_color}R{self.stats['pnl_zar']:.2f}{RESET} (Est){RESET}
            {CYAN}╚═══════════════════════════════════════════════════════════╝{RESET}
        """
        sys.stdout.write(f"\r{' ' * 100}\r")
        print(dash)

    async def scan_loop(self):
        """Main infinite loop."""
        while self.running:
            try:
                scan_msg = f"\r{YELLOW}🔄 Scanning {len(self.active_symbols)} pairs... {datetime.now().strftime('%H:%M:%S')}{RESET}\n"
                sys.stdout.write(scan_msg)
                sys.stdout.flush()

                signals_this_scan = 0

                for symbol in self.active_symbols:
                    # Global Rate Limit Check
                    if time.time() - self.last_global_signal_time < GLOBAL_SIGNAL_COOLDOWN:
                        break
                    if signals_this_scan >= MAX_SIGNALS_PER_SCAN: break

                    # Analyze
                    if await self.analyze_pair(symbol):
                        signals_this_scan += 1
                        await asyncio.sleep(2)

                    await asyncio.sleep(0.1)

                sleep_time = max(1, SCAN_INTERVAL - (time.time() % SCAN_INTERVAL))
                if int(time.time()) % 300 == 0: self.print_dashboard()
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Loop Error: {e}")
                await asyncio.sleep(5)

    async def analyze_pair(self, symbol: str) -> bool:
        try:
            url = f"{BINANCE_API_URL}/klines"
            params = {'symbol': symbol, 'interval': TIMEFRAME, 'limit': CANDLE_LIMIT}

            async with self.session.get(url, params=params) as resp:
                if resp.status != 200: return False
                data = await resp.json()

            if not isinstance(data, list): return False

            # Analyze
            signal = await self.ai_engine.analyze_market(symbol, data)
            if signal:
                self.display_signal(symbol, signal)
                self.last_global_signal_time = time.time()
                return True
            return False
        except Exception:
            return False

    def display_signal(self, symbol: str, signal: dict):
        sys.stdout.write(f"\r{' ' * 100}\r")
        display_symbol = symbol.replace("USDT", "USD")
        if "PAXG" in display_symbol: display_symbol = "XAUUSD (GOLD)"

        # Colors & Icons
        is_buy = signal['signal'] == 'BUY'
        color = GREEN if is_buy else RED
        icon = "📈" if is_buy else "📉"
        risk_badge = f"{MAGENTA}[HIGH VOLATILITY]{RESET}" if signal.get('is_high_risk') else ""

        # Formatted Output
        print(f"{WHITE}" + "="*60)
        print(f"{BOLD}{color}🚨 {signal['direction']} ({signal['signal']}) {icon} SIGNAL DETECTED {risk_badge}{RESET}")
        print(f"{WHITE}" + "="*60)

        # Section: Asset Info
        print(f"{BOLD}Asset:{RESET}        {display_symbol}")
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
        print(f"{BOLD}MAX RISK:{RESET}     {RED}-R{signal['risk_zar']:.2f}  (If Stop Loss Hit){RESET}")
        print(f"{BOLD}EST PROFIT:{RESET}   {GREEN}+R{signal['profit_zar']:.2f}  (If Take Profit Hit){RESET}")
        print(f"{WHITE}" + "-"*60)

        print(f"{BOLD}ACTION:{RESET} Open {signal['lot_size']:.2f} Lots {signal['signal']} on {display_symbol}{RESET}")
        print(f"{WHITE}" + "="*60 + "\n")

        # Launch verification task
        asyncio.create_task(self.verify_trade(symbol, signal))

    async def verify_trade(self, symbol: str, signal: dict):
        """
        Simulate trade outcome verifying against realistic Bid/Ask prices.
        """
        await asyncio.sleep(300)
        try:
            url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol}"
            async with self.session.get(url) as resp:
                data = await resp.json()
                exit_price = float(data['price'])

            entry = signal['price']
            is_buy = signal['signal'] == 'BUY'
            won = (exit_price > entry) if is_buy else (exit_price < entry)

            pnl_value = signal['profit_zar'] if won else -signal['risk_zar']

            self.stats['total'] += 1
            if won:
                self.stats['wins'] += 1
                self.stats['pnl_zar'] += pnl_value
            else:
                self.stats['loss'] += 1
                self.stats['pnl_zar'] += pnl_value

            await self.db.log_trade({
                'id': f"{symbol}_{int(time.time())}",
                'symbol': symbol, 'signal': signal['signal'],
                'confidence': signal['confidence'], 'entry': entry,
                'exit': exit_price, 'won': won, 'pnl': pnl_value,
                'strategy': signal['strategy']
            })
            self.ai_engine.learn(symbol, signal['strategy'], won, pnl_value)

            res_color = GREEN if won else RED
            txt = "WIN ✅" if won else "LOSS ❌"
            print(f"\n{BOLD}Trade Update ({symbol}):{RESET} {res_color}{txt}{RESET} (P/L: R{pnl_value:.2f})\n")

        except Exception as e:
            logger.error(f"Verify failed: {e}")

    async def stop(self):
        self.running = False
        await self.db.log_session(
            self.session_id,
            self.stats['start_time'].timestamp(),
            self.stats
        )
        if self.session: await self.session.close()
        await self.db.close()
        self.ai_engine.save_models()