export async function callEel(fnName, ...args) {
  if (window.eel && window.eel[fnName]) {
    return await window.eel[fnName](...args)();
  } else {
    if (fnName === "attempt_login") return { success: true };
    if (fnName === "get_app_version") return "v1.5.0";
    if (fnName === "logout_user") return true;
    if (fnName === "set_mode") return true;
    if (fnName === "save_settings") return true;
    if (fnName === "trigger_training") return true;
    if (fnName === "force_close") return true;

    if (fnName === "fetch_dashboard_update")
      return {
        balance: 500.0,
        equity: 520.0,
        total_pnl: 20.0,
        win_rate: 65.0,
        wins: 13,
        losses: 7,
        mode: "SIGNAL_ONLY",
        chart_labels: ["10:00", "10:05", "10:10", "10:15"],
        chart_data: [0, 10, 5, 20],
        recent_trades: [
          {
            time: "10:15",
            symbol: "BTCUSDm",
            signal_type: "BUY",
            entry: 64000.5,
            exit: 64200.0,
            size: 0.1,
            pnl: 15.0,
            result: 1,
          },
          {
            time: "10:05",
            symbol: "EURUSDm",
            signal_type: "SELL",
            entry: 1.085,
            exit: 1.086,
            size: 0.5,
            pnl: -5.0,
            result: 0,
          },
        ],
        currency: "USD",
        system_status: "IDLE",
        latency: 45,
      };

    if (fnName === "fetch_signal_updates")
      return {
        account: { balance: 500.0, equity: 520.0, currency: "USD" },
        stats: {
          active_count: 1,
          session_pnl: 20.0,
          lifetime_wr: 68.5,
          time_running: "01:45:20",
          session_total: 4,
          session_wins: 3,
          session_losses: 1,
        },
        signals: [
          {
            symbol: "BTCUSDm",
            strategy: "H4/M15 Trend Continuation",
            direction: "LONG",
            confidence: 92.5,
            lot_size: 0.1,
            price: 64000.5,
            sl: 63800.0,
            tp: 65000.0,
            risk_account: 20.0, // Replaces risk_zar
            profit_account: 100.0, // Replaces profit_zar
            status: "FILLED",
            neural_info: {
              prediction: "94.2% WIN PROB",
              sentiment: "BULL STRUCT",
              smc_state: "BOS: BULL | CHoCH: NONE",
              volatility: "1.2x AVG",
            },
          },
        ],
        logs: [
          "10:00:00 - INFO - ✅ Connected to MT5",
          "10:01:00 - INFO - 🧠 AI Engine Ready.",
          "10:05:00 - WARNING - High volatility detected on BTCUSDm",
          "10:15:00 - SUCCESS - Signal placed for BTCUSDm",
        ],
        mode: "SIGNAL_ONLY",
        monitored_symbols: [
          "BTCUSDm",
          "ETHUSDm",
          "EURUSDm",
          "XAUUSDm",
          "GBPJPYm",
          "AUDUSDm",
        ],
        latency: 45,
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
          currency: "USD",
        },
        history: Array.from({ length: 10 }).map((_, i) => ({
          time: "2026-01-20 14:30",
          symbol: i % 2 === 0 ? "BTCUSDm" : "EURUSDm",
          signal_type: i % 3 === 0 ? "SELL" : "BUY",
          entry: 1.085,
          exit: 1.082,
          pnl: i % 3 === 0 ? -15.5 : 30.0,
          result: i % 3 === 0 ? 0 : 1, // 0 = Loss, 1 = Win
          confidence: 85.5,
          size: 0.1,
        })),
        pagination: {
          current: page,
          total_pages: 5,
          total_records: 42,
        },
        latency: 45,
      };
    }

    if (fnName === "get_user_settings") {
      return {
        login: "12345678",
        server: "HFMarketsSA-Demo",
        password: "password123",
        mt5_path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        lot_size: 0.1,
        risk: 2.0,
        high_vol: false,
        confidence: 75,
        neural_meta: {
          model: "Transformer-XL v1.5.0",
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
}
