# 🚀 Nexubot: Institutional-Grade AI Trading System

![Version](https://img.shields.io/badge/version-v1.1.0-blue.svg) ![Platform](https://img.shields.io/badge/platform-MetaTrader5-green.svg) ![Account](https://img.shields.io/badge/currency-ZAR-orange.svg)

**Nexubot** is an advanced algorithmic trading engine designed for the MetaTrader 5 ecosystem. Moving beyond standard lagging indicators, Nexubot utilizes a state-of-the-art **Smart Money Concept (SMC) Engine** fused with deep **Neural Network (ML) validation** to identify high-probability institutional liquidity sweeps and trend continuations on the M1 timeframe.

## 🧠 Core Architecture

### 1. Smart Money Concepts (SMC) Engine

Nexubot reads price action exactly how institutional traders do, featuring a stateless, dynamically updating memory system:

- **Structural Mapping:** Real-time detection of Break of Structure (BOS) and Change of Character (CHoCH) aligned with Higher Timeframe (HTF) trends.
- **Liquidity Zones:** Dynamic, unmitigated mapping of Fair Value Gaps (FVGs) and Order Blocks (OBs). The bot tracks price sweeps and performs automatic garbage collection to ensure pristine memory management.
- **Session Awareness:** Automatically adapts strategies (Trend, Breakout, Mean Reversion) based on the current active global trading session (Asian, London, NY).

### 2. Deep Learning Validation (Continuous Learning Loop)

A bespoke TensorFlow/Keras neural network acts as the final gatekeeper:

- **Entry Model:** Evaluates strict SMC features (Distance to VWAP, MTF Alignment, Volatility Ratio, FVG Proximity) to predict trade success probability.
- **Exit Model:** Dynamically predicts optimal Take Profit ranges based on real-time Average True Range (ATR) expansion.
- **Self-Correction (The 20k Loop):** Every live trade outcome and its exact feature state is silently logged to a capped 20,000-row training dataset. Nexubot auto-trains its neural weights on startup to adapt to shifting market regimes.

### 3. Dynamic ZAR Risk Core, Multi-TP & Offline Recovery System

Designed specifically for precision risk management and trailing profits:

- **Multi-Tier Targets:** Calculates and executes staggered TP1, TP2, and TP3 milestones.
- **Aggressive Trailing & Ghost Tracking:** Automatically locks in breakeven at TP1, trails stops, and invisibly ghost-tracks exited trades to map theoretical TP3 hits.
- **Offline Trade Recovery:** Gracefully resumes monitoring active trades upon system reboot to prevent orphaned orders.
- **Cross-Market Scaling:** Dynamically adjusts slippage tolerances and volatility thresholds for **Indices (US30, NAS100)**, **Forex**, and **Crypto**.
- **Auto-Conversion:** Automatically calculates precise lot sizing based on live `USDZAR` rates.

### 4. Asynchronous Telegram Command Center

Operates a dedicated, non-blocking Telegram application alongside the trading engine:

- **Live Notifications:** Instant alerts for detected setups, milestone hits (TP1/TP2), and closed Pips/PnL.
- **On-Demand AI Analyst:** Query specific markets via `/analyze [SYMBOL]` to receive a deep-dive breakdown of HTF flow, local structure, and the neural network's live probability assessment.
- **Market Isolation:** Use `/focus [SYMBOLS]` to selectively isolate pairs for the scanner, or `/focus ALL` to resume global market scanning.

## 🛠️ Tech Stack

- **Core:** Python 3.12+ (AsyncIO concurrent event loop)
- **Connectivity:** MetaTrader 5 Python API
- **AI/ML:** TensorFlow, Scikit-Learn (StandardScaler)
- **Data & Math:** Pandas, NumPy
- **Database:** PostgreSQL (NeonDB) via SQLAlchemy & AsyncPG
- **Interface:** `python-telegram-bot` (Fully Asynchronous)

## ⚙️ Configuration

The bot is fully configurable via `src/config.py`:

| Setting            | Default      | Description                                                           |
| :----------------- | :----------- | :-------------------------------------------------------------------- |
| **Timeframe**      | `M15`        | Optimized for intraday structural stability.                          |
| **Risk Per Trade** | `2.0%`       | Hard cap on equity risk per signal.                                   |
| **Max Signals**    | `3 per scan` | Limits concurrent exposure.                                           |
| **Markets**        | `Dynamic`    | Auto-fetches active Crypto, Forex, and Indices from MT5 Market Watch. |

## 🛠️ Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/QuintonCodes/nexubot.git
cd nexubot
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment Setup**

Create a .env file in the root directory:

```
MT5_LOGIN=your_broker_login_id
MT5_PASSWORD=your_broker_password
MT5_SERVER=Your-Broker-Server
DATABASE_URL="postgresql+asyncpg://user:pass@host/dbname"
TELEGRAM_BOT_TOKEN="your_botfather_token"
TELEGRAM_CHAT_ID="your_personal_chat_id"
```

4. **Initialize Training Data**

Generate the baseline ML dataset by simulating past market environments:

```bash
python run_backfill.py
```

5. **Launch the Engine**

```bash
python main.py
```

Nexubot will auto-train its neural network on boot, connect to your broker, initialize the Telegram listener, and begin scanning.

## ⚠️ Disclaimer

Algorithmic trading involves significant risk and is not suitable for all investors. This software is an educational tool for signal generation and automation, not a guarantee of profit. Deep learning models map historical probabilities, which do not guarantee future performance in unprecedented market conditions. Trade responsibly.

Copyright © 2026 Nexubot Systems.
