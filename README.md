# 🚀 Nexubot Professional (v1.0.0)

**Nexubot** is an institutional-grade algorithmic trading bot designed for High-Frequency Crypto Trading on MetaTrader 5 (via HFM/Deriv). It specializes in M5 scalping strategies with a native **ZAR (South African Rand)** Risk Management Core.

## 🧠 Core Features

1. Hybrid Risk Engine (ZAR/USD)

Native ZAR Input: Input your HFM balance (e.g., R200), and the bot handles the currency conversion math automatically.

Volatility Protection: Special risk multipliers for Meme Coins (PEPE, SHIB, DOGE) to prevent stop-hunts.

Dynamic Lot Sizing: Calculates exact lot sizes to maintain strict 3% risk per trade.

2. Multi-Strategy AI

The bot scans markets using 4 distinct strategies simultaneously:

Trend Pullback: Catches dips in strong trends (EMA 50/200).

Volatility Breakout: Exploits Bollinger Band squeezes.

Reversal Scalp: Identifies overbought/oversold extremes (RSI + Bands).

MACD Momentum: Rides strong momentum impulses.

3. Cloud Analytics (Neon DB)

Async Architecture: Uses SQLAlchemy + AsyncPG for non-blocking database logging.

Session Analytics: Stores Win Rate, PnL, and Strategy Performance in the cloud for long-term optimization.

## 🛠️ Installation

1. Clone the Repository

```bash
git clone [https://github.com/QuintonCodes/nexubot.git](https://github.com/QuintonCodes/nexubot.git)
cd nexubot
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Configure Environment
   Create a .env file in the root directory:
   DATABASE_URL="postgresql://neondb_owner:xxxxx@ep-patient-mouse.aws.neon.tech/neondb?sslmode=require"

4. Run the Bot

```bash
python main.py
```

## 📊 Signal Dashboard

The console provides a live view of market scanning and performance:

╔════════════════════ SESSION DASHBOARD ════════════════════╗
║ Time Running: 0:24:09 | Total Signals: 3
║ Win Rate: 100.0% | Wins: 3 / Loss: 0
║ Session PnL: R24.00 (Est)
╚═══════════════════════════════════════════════════════════╝

## ⚠️ Risk Warning

Trading CFDs involves significant risk. This software is provided for educational purposes. Ensure you understand leverage before trading live funds.

Copyright © 2025 Nexubot Systems.
