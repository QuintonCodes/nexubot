import Chart from "chart.js/auto";
import { useEffect, useRef, useState } from "react";
import {
  MdAccountBalance,
  MdAccountBalanceWallet,
  MdArrowForwardIos,
  MdCheck,
  MdClose,
  MdEmojiEvents,
  MdHistory,
  MdHourglassTop,
  MdSettingsSuggest,
  MdShowChart,
} from "react-icons/md";
import { Link } from "react-router-dom";

import { useDashboardData } from "../hooks/useEelQuery";
import { callEel } from "../lib/eel";

function Dashboard() {
  const { data, isLoading } = useDashboardData();
  const [localModeOverride, setLocalModeOverride] = useState(null);

  // Derive the final display value: Local state takes priority, then server data
  const isAutoMode =
    localModeOverride !== null ? localModeOverride : data?.mode === "FULL_AUTO";

  // Check System Status
  const systemStatus = data?.system_status || "IDLE";
  const isBusy = systemStatus !== "IDLE";

  // --- Chart Refs ---
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  // Update Chart when data changes
  useEffect(() => {
    if (!chartRef.current || !data || !data.chart_labels || !data.chart_data)
      return;

    // Initialize Chart if not exists
    if (!chartInstance.current) {
      const ctx = chartRef.current.getContext("2d");
      const gradient = ctx.createLinearGradient(0, 0, 0, 400);
      gradient.addColorStop(0, "rgba(0, 255, 65, 0.2)");
      gradient.addColorStop(1, "rgba(0, 255, 65, 0)");

      chartInstance.current = new Chart(ctx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Cumulative PnL (ZAR)",
              data: [],
              borderColor: "#00FF41",
              backgroundColor: gradient,
              borderWidth: 2,
              pointBackgroundColor: "#000",
              pointBorderColor: "#00FF41",
              pointRadius: 3,
              pointHoverRadius: 5,
              fill: true,
              tension: 0.4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              mode: "index",
              intersect: false,
              backgroundColor: "rgba(10, 10, 10, 0.9)",
              titleColor: "#00FF41",
              bodyFont: { family: "Geist Mono" },
            },
          },
          scales: {
            x: {
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: {
                color: "#666",
                font: { size: 10, family: "Geist Mono" },
              },
            },
            y: {
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: {
                color: "#666",
                font: { size: 10, family: "Geist Mono" },
              },
            },
          },
        },
      });
    }

    // Update Chart Data
    if (data.chart_data?.length > 0) {
      const chart = chartInstance.current;
      const currentLen = chart.data.labels?.length || 0;
      const newDataLen = data.chart_labels?.length || 0;

      // Only update if data changed (simple length/last-value check)
      if (currentLen !== newDataLen) {
        chart.data.labels = data.chart_labels;
        chart.data.datasets[0].data = data.chart_data;
        chart.update();
      }
    }
  }, [data]);

  const handleModeToggle = async (e) => {
    const newState = e.target.checked;
    setLocalModeOverride(newState);

    try {
      await callEel("set_mode", newState);
    } catch (error) {
      setLocalModeOverride(null);
      console.error("Mode toggle error", error);
    }
  };

  const fmtPnL = (val) => {
    if (val === undefined || val === null || Math.abs(val) < 0.005) {
      return <span className="text-white">R 0.00</span>;
    }
    const isWin = val > 0;
    return (
      <span className={isWin ? "text-primary" : "text-danger"}>
        R {isWin ? "+" : ""}
        {val.toFixed(2)}
      </span>
    );
  };

  // Helper for currency formatting
  const fmtCurrency = (val) => {
    const safeVal = Math.abs(val) < 0.005 ? 0 : val;
    return `R ${(safeVal || 0).toFixed(2)}`;
  };

  if (isLoading || !data) {
    return (
      <div className="flex items-center justify-center h-full text-primary font-mono animate-pulse">
        INITIALIZING DASHBOARD...
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Top Stats Grid */}

      {/* SYSTEM BUSY BANNER */}
      {isBusy && (
        <div className="w-full bg-purple-900/30 border border-purple-500/50 p-4 rounded-sm flex items-center justify-center gap-4 animate-pulse">
          <MdHourglassTop className="text-purple-400 text-2xl animate-spin" />
          <div className="text-purple-100 font-bold tracking-widest">
            SYSTEM IS {systemStatus}... TRADING PAUSED.
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Balance Card */}
        <div className="bg-panel-dark border border-border-dark p-4 flex items-center justify-between shadow-[0_0_10px_rgba(0,0,0,0.3)] hover:border-primary/50 transition-colors group">
          <div>
            <div className="text-[10px] uppercase text-gray-500 tracking-wider mb-1 group-hover:text-primary transition-colors">
              Account Balance
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">
              {fmtCurrency(data.balance)}
            </div>
            <div className="text-[10px] text-primary mt-1 flex items-center gap-1">
              <span className="text-sm material-icons">
                <MdAccountBalance />
              </span>{" "}
              MT5 LIVE
            </div>
          </div>
          <div className="h-10 w-10 rounded bg-gray-800 flex items-center justify-center">
            <MdAccountBalanceWallet className="text-gray-500 text-xl" />
          </div>
        </div>

        {/* PnL Card */}
        <div className="bg-panel-dark border border-border-dark p-4 flex items-center justify-between shadow-[0_0_10px_rgba(0,0,0,0.3)] hover:border-primary/50 transition-colors group">
          <div>
            <div className="text-[10px] uppercase text-gray-500 tracking-wider mb-1 group-hover:text-primary transition-colors">
              Total PnL
            </div>
            <div
              className={`text-2xl font-bold tracking-tight ${data.total_pnl >= 0 ? "text-primary" : "text-danger"}`}
            >
              {fmtPnL(data.total_pnl)}
            </div>
            <div className="text-[10px] text-gray-500 mt-1">
              All Time Performance
            </div>
          </div>
          <div className="h-10 w-10 rounded bg-primary/10 flex items-center justify-center border border-primary/20">
            <MdShowChart className="text-primary text-xl" />
          </div>
        </div>

        {/* Win Rate Card */}
        <div className="bg-panel-dark border border-border-dark p-4 flex items-center justify-between shadow-[0_0_10px_rgba(0,0,0,0.3)] hover:border-primary/50 transition-colors group">
          <div>
            <div className="text-[10px] uppercase text-gray-500 tracking-wider mb-1 group-hover:text-primary transition-colors">
              Win Rate
            </div>
            <div className="text-2xl font-bold text-secondary tracking-tight">
              {(data.win_rate || 0).toFixed(1)}%
            </div>
            <div className="text-[10px] text-gray-500 mt-1">
              {data.wins || 0} Wins / {data.losses || 0} Losses
            </div>
          </div>
          <div className="h-10 w-10 rounded bg-secondary/10 flex items-center justify-center border border-secondary/20">
            <MdEmojiEvents className="text-secondary text-xl" />
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart Section */}
        <div className="lg:col-span-2 bg-panel-dark border border-border-dark p-1 relative shadow-neon-cyan/20">
          {/* Decorative Corners */}
          <div className="absolute top-0 left-0 w-2 h-2 border-l border-t border-secondary"></div>
          <div className="absolute top-0 right-0 w-2 h-2 border-r border-t border-secondary"></div>
          <div className="absolute bottom-0 left-0 w-2 h-2 border-l border-b border-secondary"></div>
          <div className="absolute bottom-0 right-0 w-2 h-2 border-r border-b border-secondary"></div>

          <div className="h-full p-4 flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-bold text-lg text-white tracking-wider flex items-center gap-2">
                <span className="w-1.5 h-4 bg-secondary"></span> PERFORMANCE
                CURVE
              </h3>
            </div>
            <div className="relative w-full h-75 bg-[#0c0c0c] border border-gray-800 rounded-sm p-2">
              <canvas ref={chartRef}></canvas>
            </div>
          </div>
        </div>

        {/* Status Panel */}
        <div className="bg-panel-dark border border-border-dark p-6 relative">
          <div className="absolute top-0 right-0 p-3 opacity-20">
            <MdSettingsSuggest className="text-4xl text-gray-500" />
          </div>

          <h3 className="font-bold text-lg text-white tracking-wider mb-6 border-b border-gray-700 pb-2">
            TRADING STATUS
          </h3>

          <div className="mb-8">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-gray-400 uppercase tracking-widest">
                Execution Mode
              </span>
              <span
                className={`text-xs font-bold animate-pulse ${isAutoMode ? "text-primary" : "text-warning"}`}
              >
                {isAutoMode ? "AUTOMATED" : "SIGNAL ONLY"}
              </span>
            </div>

            <div className="relative inline-block w-full align-middle select-none transition duration-200 ease-in">
              <input
                type="checkbox"
                name="toggle"
                id="mode-toggle"
                checked={isAutoMode}
                onChange={handleModeToggle}
                className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 border-gray-700 appearance-none cursor-pointer transition-all duration-300 left-0 checked:left-[calc(100%-1.5rem)] checked:border-primary"
              />
              <label
                htmlFor="mode-toggle"
                className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-800 cursor-pointer border border-gray-700"
              ></label>
            </div>

            <div className="flex justify-between mt-2 text-[10px] text-gray-500">
              <span className={!isAutoMode ? "text-warning" : ""}>
                SIGNAL ONLY
              </span>
              <span className={isAutoMode ? "text-primary" : ""}>
                FULL AUTO
              </span>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center border-b border-gray-800 pb-2">
              <span className="text-xs text-gray-400">Risk Profile</span>
              <span className="text-xs text-warning border border-warning/30 px-2 py-0.5 bg-warning/10 rounded">
                AGGRESSIVE
              </span>
            </div>
            <div className="flex justify-between items-center border-b border-gray-800 pb-2">
              <span className="text-xs text-gray-400">Equity</span>
              <span className="text-xs text-white">
                {fmtCurrency(data.equity)}
              </span>
            </div>
            <div className="flex justify-between items-center border-b border-gray-800 pb-2">
              <span className="text-xs text-gray-400">Bridge Status</span>
              <span className="text-xs text-primary flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span>{" "}
                CONNECTED
              </span>
            </div>
          </div>

          <div className="mt-6 pt-4">
            <Link to="/settings">
              <button className="w-full py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs border border-gray-600 transition-colors uppercase cursor-pointer">
                Configure Engine
              </button>
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-panel-dark border border-border-dark p-6 relative">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-bold text-lg text-white tracking-wider flex items-center gap-2">
            <MdHistory className="text-gray-500 text-lg" /> RECENT ACTIVITY
          </h3>
          <Link
            to="/history"
            className="text-xs text-primary hover:text-white transition-colors flex items-center gap-1"
          >
            VIEW ALL <MdArrowForwardIos className="text-[10px]" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-700">
                <th className="py-3 pl-2 font-normal uppercase tracking-wider">
                  Time
                </th>
                <th className="py-3 font-normal uppercase tracking-wider">
                  Pair
                </th>
                <th className="py-3 font-normal uppercase tracking-wider">
                  Type
                </th>
                <th className="py-3 font-normal uppercase tracking-wider">
                  Entry
                </th>
                <th className="py-3 font-normal uppercase tracking-wider">
                  Exit
                </th>
                <th className="py-3 font-normal uppercase tracking-wider">
                  Size
                </th>
                <th className="py-3 pr-2 text-right font-normal uppercase tracking-wider">
                  PnL
                </th>
                <th className="py-3 pr-2 text-right font-normal uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              {data.recent_trades?.map((trade, idx) => {
                const isWin = trade.result === 1;
                const pnlClass = isWin ? "text-primary" : "text-danger";
                const statusColor = isWin ? "primary" : "danger";
                const Icon = isWin ? MdCheck : MdClose;

                return (
                  <tr
                    key={idx}
                    className="border-b border-gray-800 hover:bg-white/5 transition-colors group"
                  >
                    <td className="py-3 pl-2 text-xs text-gray-500">
                      {trade.time}
                    </td>
                    <td className="py-3 font-bold text-white group-hover:text-secondary">
                      {trade.symbol}
                    </td>
                    <td className="py-3">
                      <span
                        className={`text-${trade.signal_type === "BUY" ? "primary" : "danger"} bg-${trade.signal_type === "BUY" ? "primary" : "danger"}/10 px-1.5 py-0.5 text-[10px] rounded border border-${trade.signal_type === "BUY" ? "primary" : "danger"}/20`}
                      >
                        {trade.signal_type}
                      </span>
                    </td>
                    <td className="py-3">{trade.entry.toFixed(5)}</td>
                    <td className="py-3">{trade.exit.toFixed(5)}</td>
                    <td className="py-3 text-xs">{trade.size}</td>
                    <td
                      className={`py-3 pr-2 text-right ${pnlClass} font-bold`}
                    >
                      {fmtPnL(trade.pnl)}
                    </td>
                    <td className="py-3 pr-2 text-right">
                      <span
                        className={`text-${statusColor} text-[10px] uppercase border border-${statusColor}/30 px-2 py-0.5 rounded-full flex items-center justify-end gap-1 ml-auto w-fit`}
                      >
                        <Icon className="text-[12px] inline mr-1" />
                        {isWin ? "TP Hit" : "SL Hit"}
                      </span>
                    </td>
                  </tr>
                );
              })}
              {(!data.recent_trades || data.recent_trades.length === 0) && (
                <tr>
                  <td
                    colSpan="8"
                    className="text-center py-4 text-gray-500 text-xs italic"
                  >
                    No recent trades recorded this session.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <footer className="mt-4 py-3 text-center text-xs text-gray-700">
        <p>
          NEXUBOT INSTITUTIONAL ENGINE © {new Date().getFullYear()}. ALL RIGHTS
          RESERVED.
        </p>
        <p className="mt-1">
          WARNING: Trading involves substantial risk of loss.
        </p>
      </footer>
    </div>
  );
}

export default Dashboard;
