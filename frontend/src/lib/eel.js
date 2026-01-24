export const callEel = async (fnName, ...args) => {
  if (window.eel && window.eel[fnName]) {
    return await window.eel[fnName](...args)();
  } else {
    if (fnName === "attempt_login") return { success: true };

    if (fnName === "fetch_dashboard_update")
      return {
        balance: 500.0,
        equity: 520.0,
        total_pnl: 20.0,
        win_rate: 65.0,
        wins: 13,
        losses: 7,
        mode: "SIGNAL_ONLY",
        chart_labels: ["10:00", "10:05", "10:10"],
        chart_data: [0, 10, 20],
        recent_trades: [],
      };

    if (fnName === "fetch_signal_updates")
      return {
        account: { balance: 500.0, equity: 520.0 },
        stats: {
          active_count: 1,
          session_pnl: 150.52,
          lifetime_wr: 68.5,
          time_running: "01:45:20",
          session_total: 4,
          session_wins: 3,
          session_losses: 1,
        },
        signals: [
          {
            symbol: "BTCUSDm",
            strategy: "Crypto Ichimoku",
            direction: "LONG",
            confidence: 85.5,
            price: 65000.0,
            sl: 64500.0,
            tp: 66000.0,
            lot_size: 0.1,
            risk_zar: 150.0,
            profit_zar: 450.0,
            status: "OPEN",
            neural_info: {
              prediction: "88% WIN PROB",
              sentiment: "BULLISH STRUCT",
              volatility: "1.2x AVG",
            },
          },
          {
            symbol: "BTCUSDm",
            strategy: "Crypto Ichimoku",
            direction: "LONG",
            confidence: 85.5,
            price: 65000.0,
            sl: 64500.0,
            tp: 66000.0,
            lot_size: 0.1,
            risk_zar: 150.0,
            profit_zar: 450.0,
            status: "OPEN",
            neural_info: {
              prediction: "88% WIN PROB",
              sentiment: "BULLISH STRUCT",
              volatility: "1.2x AVG",
            },
          },
        ],
        logs: [
          "[INFO] System initialized",
          "[SUCCESS] Connection established",
          "[INFO] Scanning BTCUSDm...",
        ],
        mode: "SIGNAL_ONLY",
      };

    if (fnName === "fetch_trade_history") {
      const [filters] = args;
      const page = filters?.page || 1;

      return {
        stats: {
          balance: 500.0,
          lifetime_wr: 68.5,
          total_trades: 42,
          lifetime_pnl: 1250.5,
        },
        history: Array.from({ length: 10 }).map((_, i) => ({
          time: "2026-01-20 14:30",
          symbol: i % 2 === 0 ? "BTCUSDm" : "EURUSDm",
          signal_type: i % 3 === 0 ? "SELL" : "BUY",
          entry: 1.085,
          exit: 1.082,
          pnl: i % 3 === 0 ? -150 : 300,
          result: i % 3 === 0 ? 0 : 1, // 0 = Loss, 1 = Win
          confidence: 85,
        })),
        pagination: {
          current: page,
          total_pages: 5,
          total_records: 42,
        },
      };
    }

    if (fnName === "get_user_settings") {
      return {
        login: "12345678",
        server: "HFMarketsSA-Demo",
        password: "password123",
        lot_size: 0.1,
        risk: 2.0,
        high_vol: false,
        confidence: 75,
        neural_meta: {
          model: "Transformer-XL v1.4",
          epochs: "50,000",
          bias: "Balanced",
        },
      };
    }

    if (fnName === "save_settings") {
      console.log("[Eel] Settings Saved:", args[0]);
      await new Promise((r) => setTimeout(r, 1000)); // Simulate delay
      return true;
    }
    return null;
  }
};
