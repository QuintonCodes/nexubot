# 🚀 Nexubot: Institutional-Grade AI Trading System

![Version](https://img.shields.io/badge/version-v1.3.1-blue.svg) ![Platform](https://img.shields.io/badge/platform-MetaTrader5-green.svg) ![Account](https://img.shields.io/badge/currency-ZAR-orange.svg)

**Nexubot** is an advanced algorithmic trading engine designed for the **HFMarkets (HFM)** ecosystem. Unlike standard bots that rely solely on indicators, Nexubot utilizes a **Confluence Engine** that fuses Technical Strategies, Chart Pattern Recognition, and Neural Network (ML) validation to generate high-probability trade signals on the M15 timeframe.

## 🧠 Core Architecture

### 1. Tri-Factor Confluence Engine

Nexubot validates every trade through three distinct layers before execution:

- **Layer 1: Strategy Logic:** Checks trends (Ichimoku, EMA Flow) and mean reversions (Bollinger, RSI).
- **Layer 2: Pattern Recognition:** Detects institutional setups like **Head & Shoulders**, **Double Tops/Bottoms**, and **Bull/Bear Flags** using `scipy` signal processing.
- **Layer 3: Neural Validation:** A TensorFlow/Keras model predicts the probability of success. If the ML confidence < 45%, the trade is vetoed regardless of the setup.

### 2. Native ZAR Risk Core (South Africa Optimized)

Designed specifically for South African traders using ZAR-denominated accounts.

- **Auto-Conversion:** Automatically calculates pip values and risk based on `USDZAR` rates for accurate lot sizing.
- **Small Account Protection:** Optimized for balances starting at **R450**.
- **Dynamic Volatility Scaling:** Automatically halves risk during high-impact news or "Meme Coin" volatility spikes.

### 3. Machine Learning Integration

- **Entry Model:** Binary classification model to predict trade success probability.
- **Exit Model:** Regression model that predicts the optimal Take Profit distance (ATR Multiples) based on current market volatility.
- **Self-Correction:** Logs every trade outcome (Win/Loss/Excursion) to a CSV dataset for continuous model retraining.

## 🛠️ Tech Stack

- **Core:** Python 3.12+ (AsyncIO for non-blocking execution)
- **Connectivity:** MetaTrader 5 Python API (Direct HFM Integration)
- **Analysis:** Pandas, NumPy, SciPy (Signal processing)
- **AI/ML:** TensorFlow, Scikit-Learn (StandardScaler)
- **Database:** PostgreSQL (NeonDB) via SQLAlchemy & AsyncPG

## ⚙️ Configuration

The bot is fully configurable via `src/config.py`:

| Setting            | Default  | Description                                          |
| :----------------- | :------- | :--------------------------------------------------- |
| **Timeframe**      | `M15`    | Optimized for intraday stability.                    |
| **Risk Per Trade** | `2.0%`   | Hard cap on equity risk per signal.                  |
| **Max Confidence** | `95%`    | Capped realism to prevent overfitting.               |
| **Markets**        | `Hybrid` | Trades Crypto (#BTC, #ETH) & Forex (EURUSD, XAUUSD). |

## 🛠️ Installation & Setup

1. **Clone the Repository**

```bash
git clone [https://github.com/QuintonCodes/nexubot.git](https://github.com/QuintonCodes/nexubot.git)
cd nexubot
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment Setup**
   Create a .env file in the root directory:

```
MT5_LOGIN=your_hfm_login_id
MT5_PASSWORD=your_hfm_password
MT5_SERVER=HFMarketsSA-Demo2
DATABASE_URL="postgresql+asyncpg://user:pass@host/dbname"
```

4. **Launch the Engine**

```bash
python main.py
```

On first run, the bot will auto-backfill training data if none exists.

## 📊 Dashboard Output

Nexubot provides a real-time console dashboard tracking session performance:

```plaintext
╔════════════════════ SESSION DASHBOARD ════════════════════╗
║ Time Running: 04:12:05 | Total Signals: 12                ║
║ Win Rate: 75.0%        | Wins: 9 / Loss: 3                ║
║ Session PnL: +R1,250.50 (Realized)                        ║
╚═══════════════════════════════════════════════════════════╝
```

## ⚠️ Disclaimer

Algorithmic trading involves significant risk. This software is a tool for signal generation and automation, not a guarantee of profit. Past performance of the Neural Network does not guarantee future results.

Copyright © 2025 Nexubot Systems. Released under [MIT License](https://www.google.com/search?q=LICENSE)
