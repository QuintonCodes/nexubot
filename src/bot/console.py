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
    SCAN_INTERVAL, CANDLE_LIMIT, DEFAULT_BALANCE_ZAR,
    GLOBAL_SIGNAL_COOLDOWN, MAX_SIGNALS_PER_SCAN,
    VERSION, APP_NAME, SPREAD_COST_PCT
)
from src.utils.market_data import fetch_top_volatile_symbols

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
        print(f"{WHITE}Broker Spread Sim:{RESET} {YELLOW}{SPREAD_COST_PCT}%{RESET}")
        print(f"{WHITE}Leverage:{RESET}          {YELLOW}1:1000{RESET}")

        # --- Database Init ---
        await self.db.init_database()

        # --- User Input ---
        try:
            print(f"{WHITE}Enter Account Balance (ZAR) [Default: R{DEFAULT_BALANCE_ZAR}]: {RESET}", end="")
            user_input = input()
            zar_balance = float(user_input) if user_input.strip() else DEFAULT_BALANCE_ZAR
        except ValueError:
            zar_balance = DEFAULT_BALANCE_ZAR

        self.ai_engine.set_user_balance(zar_balance)

        print(f"✅ Account Set: {GREEN}R{zar_balance:.2f}{RESET}")
        print(f"{YELLOW}Global Rate Limit: 1 Signal / {GLOBAL_SIGNAL_COOLDOWN//60} mins{RESET}")

        # --- Network Setup (Google DNS) ---
        resolver = aiohttp.AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])
        connector = aiohttp.TCPConnector(
            family=socket.AF_INET,
            ssl=False,
            limit=100,
            resolver=resolver
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True
        )

        # --- Market Fetch ---
        if USE_DYNAMIC_SYMBOLS:
            print("🌊 Fetching market data...")
            dynamic_symbols = await fetch_top_volatile_symbols(self.session)
            if dynamic_symbols:
                self.active_symbols = dynamic_symbols
                print(f"✅ Monitoring: {', '.join(self.active_symbols)}\n")

        await self.scan_loop()

    def print_dashboard(self):
        """Displays session statistics."""
        elapsed = datetime.now() - self.stats['start_time']
        win_rate = (self.stats['wins']/self.stats['total']*100) if self.stats['total'] > 0 else 0
        pnl_color = GREEN if self.stats['pnl_zar'] >= 0 else RED

        dash = f"""
            {CYAN}╔════════════════════ SESSION DASHBOARD ════════════════════╗{RESET}
            {CYAN}║{RESET} Time Running: {str(elapsed).split('.')[0]}   |   Total Signals: {self.stats['total']}
            {CYAN}║{RESET} Win Rate: {BOLD}{win_rate:.1f}%{RESET}         |   Wins: {GREEN}{self.stats['wins']}{RESET} / Loss: {RED}{self.stats['loss']}{RESET}
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

                    # Scan Limit Check (Max 2 per loop)
                    if signals_this_scan >= MAX_SIGNALS_PER_SCAN:
                        break

                    # Analyze
                    if await self.analyze_pair(symbol):
                        signals_this_scan += 1
                        await asyncio.sleep(1) # Short pause between signals

                    await asyncio.sleep(0.1)

                sleep_time = max(1, SCAN_INTERVAL - (time.time() % SCAN_INTERVAL))

                # Show dashboard periodically
                if int(time.time()) % 300 == 0:
                    self.print_dashboard()

                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Loop Error: {e}")
                await asyncio.sleep(5)

    async def analyze_pair(self, symbol: str) -> bool:
        try:
            url = f"{BINANCE_API_URL}/klines"
            params = {'symbol': symbol, 'interval': '5m', 'limit': CANDLE_LIMIT}

            async with self.session.get(url, params=params) as resp:
                if resp.status != 200: return False
                data = await resp.json()

            if not isinstance(data, list): return False

            signal = self.ai_engine.analyze_market(symbol, data)

            if signal:
                self.display_signal(symbol, signal)
                self.last_global_signal_time = time.time()
                return True

            return False
        except Exception:
            return False

    def display_signal(self, symbol: str, signal: dict):
        sys.stdout.write(f"\r{' ' * 100}\r") # Clean line

        # Visual Alert
        color = GREEN if signal['signal'] == 'BUY' else RED
        direction_text = f"{signal['direction']} ({signal['signal']})"
        icon = "📈" if signal['signal'] == 'BUY' else "📉"
        risk_badge = f"{MAGENTA}[HIGH RISK]{RESET} " if signal.get('is_high_risk') else ""

        print(f"{WHITE}" + "="*60)
        print(f"{BOLD}{color}🚨 {direction_text} {icon} SIGNAL DETECTED {risk_badge}{RESET}")
        print(f"{WHITE}" + "="*60)

        print(f"{BOLD}Asset:{RESET}       {symbol}")
        print(f"{BOLD}Strategy:{RESET}    {signal['strategy']}")
        print(f"{BOLD}Confidence:{RESET}  {signal['confidence']:.1f}%")
        print(f"{WHITE}" + "-"*60)

        # Explicitly mention pricing type to user
        price_lbl = "Ask" if signal['signal'] == 'BUY' else "Bid"
        print(f"{BOLD}ENTRY ({price_lbl}):{RESET} {signal['price']}")
        print(f"{RED}STOP LOSS:{RESET}   {signal['sl']}")
        print(f"{GREEN}TAKE PROFIT:{RESET} {signal['tp']}")
        print(f"{WHITE}" + "-"*60)

        print(f"{BOLD}LOT SIZE:{RESET}   {signal['lot_size']} units")
        print(f"{BOLD}RISK:{RESET}       R{signal['risk_zar']:.2f}")
        print(f"{BOLD}EST PROFIT:{RESET} R{signal['profit_zar']:.2f}")
        print(f"{WHITE}Instruction:{RESET} Open 1 Trade with listed Lot Size")

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

                # Binance returns 'price' which usually tracks the latest trade (close to Mid/Bid)
                # We assume raw_price is BID.
                raw_price = float(data['price'])
                bid_price = raw_price
                ask_price = raw_price * (1 + (SPREAD_COST_PCT/100))

            won = False
            entry = signal['price'] # This entry already accounts for spread

            if signal['signal'] == 'BUY':
                # Long closes at BID.
                # To win, current BID must be higher than Entry Ask.
                exit_price = bid_price
                won = exit_price > entry
            else:
                # Short closes at ASK.
                # To win, current ASK must be lower than Entry Bid.
                exit_price = ask_price
                won = exit_price < entry

            # Calculate Realized PnL for Stats
            # If won, we gain Profit. If loss, we lose Risk amount.
            pnl_value = signal['profit_zar'] if won else -signal['risk_zar']

            self.stats['total'] += 1
            if won:
                self.stats['wins'] += 1
                self.stats['pnl_zar'] += pnl_value
            else:
                self.stats['loss'] += 1
                self.stats['pnl_zar'] += pnl_value

            # AI Learning (Pass PnL value)
            await self.save_trade_to_db(symbol, signal, exit_price, won, pnl_value)
            self.ai_engine.learn(symbol, signal['strategy'], won, pnl_value)

            res_color = GREEN if won else RED
            txt = "WIN ✅" if won else "LOSS ❌"
            print(f"\n{BOLD}Trade Update ({symbol}):{RESET} {res_color}{txt}{RESET} (P/L: R{pnl_value:.2f})")

        except Exception as e:
            logger.error(f"Verify failed: {e}")

    async def save_trade_to_db(self, symbol: str, signal: dict, exit_price: float, won: bool, pnl_value):
        try:
            trade_id = f"{symbol}_{int(time.time())}_{int(signal['price'])}"
            data = {
                'id': trade_id, 'symbol': symbol,
                'signal': signal['signal'], 'confidence': signal['confidence'],
                'entry': signal['price'], 'exit': exit_price,
                'won': won, 'pnl': pnl_value, 'strategy': signal['strategy']
            }
            await self.db.log_trade(data)
        except Exception:
            pass

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