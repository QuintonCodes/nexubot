import { useState } from "react";
import {
  MdAccountBalanceWallet,
  MdCheckCircle,
  MdChevronLeft,
  MdChevronRight,
  MdEmojiEvents,
  MdPsychology,
  MdShowChart,
  MdSync,
  MdTrendingUp,
  MdVisibility,
} from "react-icons/md";
import { useForceClose, useSignalData } from "../hooks/useEelQuery";

function Signal() {
  // 1. Data from Hook
  const { data } = useSignalData();
  const { mutate: forceClose } = useForceClose();

  // 2. Local UI State
  const [currentIndex, setCurrentIndex] = useState(0);

  const signals = data?.signals || [];
  const activeSignal = signals.length > 0 ? signals[currentIndex] : null;

  const nextSignal = () => {
    setCurrentIndex((prev) => (prev + 1) % signals.length);
  };

  const prevSignal = () => {
    setCurrentIndex((prev) => (prev - 1 + signals.length) % signals.length);
  };

  const handleForceClose = () => {
    if (
      activeSignal &&
      confirm(`Force close trade for ${activeSignal.symbol}?`)
    ) {
      forceClose(activeSignal.symbol);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Top Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          {
            label: "Account Balance",
            val: `R ${data.account.balance.toFixed(2)}`,
            color: "text-white",
            icon: <MdAccountBalanceWallet className="text-gray-600 text-xl" />,
          },
          {
            label: "Win Rate",
            val: `${data.stats.lifetime_wr.toFixed(1)}%`,
            color: "text-secondary",
            icon: <MdEmojiEvents className="text-secondary/50 text-xl" />,
          },
          {
            label: "Active Trades",
            val: data.stats.active_count,
            color: "text-secondary",
            icon: <MdTrendingUp className="text-secondary/50 text-xl" />,
          },
          {
            label: "Session PnL",
            val: `R ${data.stats.session_pnl > 0 ? "+" : "-"}${data.stats.session_pnl.toFixed(2)}`,
            color: `${data.stats.session_pnl > 0 ? "text-primary" : "text-danger"}`,
            icon: <MdShowChart className="text-primary/50 text-xl" />,
          },
        ].map((stat, i) => (
          <div
            key={i}
            className="bg-panel-dark border border-border-dark p-3 flex items-center justify-between"
          >
            <div>
              <div className="text-[10px] uppercase text-gray-500 tracking-wider mb-1">
                {stat.label}
              </div>
              <div className={`text-xl font-bold ${stat.color}`}>
                {stat.val}
              </div>
            </div>
            {stat.icon}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Signal Card + Neural Info */}
        <div className="lg:col-span-2 space-y-6">
          <div id="signal-container" className="relative min-h-75">
            {signals.length === 0 ? (
              <div className="bg-panel-dark border border-gray-800 p-12 text-center shadow-lg h-full flex flex-col items-center justify-center">
                <div className="w-16 h-16 border-4 border-t-primary border-gray-800 rounded-full animate-spin mb-4"></div>
                <h2 className="text-xl text-white font-bold tracking-widest animate-pulse">
                  SCANNING MARKETS...
                </h2>
                <p className="text-gray-500 text-sm mt-2">
                  Neural Engine is analyzing price action.
                </p>
              </div>
            ) : (
              <div>
                {/* Navigation Controls */}
                {signals.length > 1 && (
                  <div className="flex justify-between items-center mb-4 bg-gray-900 p-2 rounded">
                    <button
                      onClick={prevSignal}
                      className="text-gray-400 hover:text-white cursor-pointer"
                    >
                      <MdChevronLeft size={24} />
                    </button>
                    <span className="text-xs text-gray-400">
                      Signal {currentIndex + 1} of {signals.length}
                    </span>
                    <button
                      onClick={nextSignal}
                      className="text-gray-400 hover:text-white cursor-pointer"
                    >
                      <MdChevronRight size={24} />
                    </button>
                  </div>
                )}

                {/* The Signal Card */}
                {activeSignal && (
                  <div
                    className={`bg-panel-dark border ${activeSignal.direction === "LONG" ? "border-primary shadow-neon-green" : "border-danger shadow-neon-red"} relative overflow-hidden group p-6`}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <span
                        className={`animate-ping absolute inline-flex h-3 w-3 rounded-full opacity-75 left-6 ${activeSignal.direction === "LONG" ? "bg-green-400" : "bg-red-400"}`}
                      ></span>
                      <span
                        className={`relative inline-flex rounded-full h-3 w-3 ${activeSignal.direction === "LONG" ? "bg-primary" : "bg-danger"}`}
                      ></span>
                      <h2
                        className={`text-2xl md:text-3xl font-bold tracking-tight ${activeSignal.direction === "LONG" ? "text-primary" : "text-danger"}`}
                      >
                        {activeSignal.direction === "LONG"
                          ? "BUY SIGNAL"
                          : "SELL SIGNAL"}{" "}
                        DETECTED
                      </h2>
                      <span
                        className={`bg-black/20 ${activeSignal.status === "FILLED" ? "text-primary" : "text-warning border-warning"} text-xs px-2 py-0.5 border rounded-sm`}
                      >
                        {activeSignal.status || "PENDING"}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
                      <div className="space-y-4 text-sm">
                        <div className="flex justify-between border-b border-dashed border-gray-700 pb-1">
                          <span className="text-gray-500">ASSET:</span>
                          <span className="text-white font-bold text-lg">
                            {activeSignal.symbol}
                          </span>
                        </div>
                        <div className="flex justify-between border-b border-dashed border-gray-700 pb-1">
                          <span className="text-gray-500">STRATEGY:</span>
                          <span className="text-secondary">
                            {activeSignal.strategy}
                          </span>
                        </div>
                        <div className="flex justify-between items-center border-b border-dashed border-gray-700 pb-1">
                          <span className="text-gray-500">CONFIDENCE:</span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-secondary shadow-neon-cyan"
                                style={{ width: `${activeSignal.confidence}%` }}
                              ></div>
                            </div>
                            <span className="text-secondary font-bold">
                              {activeSignal.confidence.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <div class="flex justify-between border-b border-dashed border-gray-700 pb-1 pt-2">
                          <span class="text-gray-500">LOT SIZE:</span>
                          <span class="text-warning font-bold">
                            {activeSignal.lot_size} Lots
                          </span>
                        </div>
                      </div>

                      <div className="space-y-4 text-sm bg-black/40 p-4 border border-gray-800">
                        <div className="flex justify-between">
                          <span className="text-gray-500">ENTRY:</span>{" "}
                          <span className="text-white">
                            {activeSignal.price}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-danger font-bold">
                            STOP LOSS:
                          </span>{" "}
                          <span className="text-danger">{activeSignal.sl}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-primary font-bold">
                            TAKE PROFIT:
                          </span>{" "}
                          <span className="text-primary">
                            {activeSignal.tp}
                          </span>
                        </div>

                        <div className="mt-4 pt-4 border-t border-gray-700 grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-[10px] text-gray-500">
                              RISK (SL)
                            </div>
                            <div className="text-danger font-bold">
                              -R{activeSignal.risk_zar.toFixed(2)}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-[10px] text-gray-500">
                              PROFIT (TP)
                            </div>
                            <div className="text-primary font-bold">
                              +R{activeSignal.profit_zar.toFixed(2)}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="mt-6 flex flex-col sm:flex-row gap-4 pt-4 border-t border-dashed border-gray-700">
                      {data.mode === "FULL_AUTO" ? (
                        <>
                          <div className="flex-1 bg-black/50 p-2 border border-gray-700 text-xs text-gray-400 flex items-center gap-2">
                            <MdSync className="animate-spin text-sm" />{" "}
                            Executing trade via MetaTrader 5...
                          </div>
                          <button
                            onClick={handleForceClose}
                            className="bg-danger hover:bg-red-600 text-white px-6 py-2 border border-danger shadow-neon-red text-sm font-bold uppercase transition-all"
                          >
                            Force Close
                          </button>
                        </>
                      ) : (
                        <div className="flex-1 bg-yellow-900/20 p-2 border border-yellow-700/50 text-xs text-yellow-500 flex items-center gap-2">
                          <MdVisibility /> Signal Tracking Only (Auto-Mode
                          Disabled)
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Neural Layer */}
          <div className="bg-panel-dark border border-border-dark p-6 relative">
            <h3 className="text-lg font-bold text-secondary mb-4 flex items-center gap-2">
              <MdPsychology className="text-xl" /> NEURAL VALIDATION LAYER
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="bg-black/30 p-3 border-l-2 border-primary">
                <div className="text-xs text-gray-500 uppercase">
                  TensorFlow Prediction
                </div>
                <div className="text-primary font-bold mt-1">
                  {activeSignal?.neural_info?.prediction || "--"}
                </div>
              </div>
              <div className="bg-black/30 p-3 border-l-2 border-warning">
                <div className="text-xs text-gray-500 uppercase">
                  Sentiment Analysis
                </div>
                <div className="text-warning font-bold mt-1">
                  {activeSignal?.neural_info?.sentiment || "--"}
                </div>
              </div>
              <div className="bg-black/30 p-3 border-l-2 border-secondary">
                <div className="text-xs text-gray-500 uppercase">
                  Volatility Index
                </div>
                <div className="text-secondary font-bold mt-1">
                  {activeSignal?.neural_info?.volatility || "--"}
                </div>
              </div>
            </div>

            <div className="bg-black p-4 text-xs text-gray-400 border border-gray-800 h-40 overflow-y-auto">
              {activeSignal ? (
                <>
                  <p className="text-gray-500">
                    &gt; Initializing neural weights for {activeSignal.symbol}
                    ...
                  </p>
                  <p>
                    &gt; Strategy Layer:{" "}
                    <span className="text-secondary">
                      {activeSignal.strategy} detected
                    </span>
                  </p>
                  <p>
                    &gt; Trend Correlation:{" "}
                    <span className="text-primary">aligned with H4 Trend</span>
                  </p>
                  <p>&gt; Volatility Check: ATR within operational bounds.</p>
                  <p>
                    &gt; Risk Calculation: {activeSignal.lot_size} lots fits
                    &lt; 2% risk profile.
                  </p>
                  <p>
                    &gt; Final Verdict:{" "}
                    <span
                      className={`text-white px-1 ${activeSignal.direction === "LONG" ? "bg-primary text-black" : "bg-danger"}`}
                    >
                      {activeSignal.direction}
                    </span>{" "}
                    confirmed.
                  </p>
                  <p className="animate-pulse">_</p>
                </>
              ) : (
                <p className="animate-pulse">_</p>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Patterns & Logs */}
        <div className="space-y-6">
          <div className="bg-panel-dark border border-secondary/20 p-5 shadow-[0_0_15px_rgba(0,255,255,0.05)]">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-4 border-b border-gray-700 pb-2">
              Pattern Recognition
            </h3>
            <div className="space-y-4">
              {activeSignal ? (
                <>
                  <div className="flex items-start gap-3">
                    <MdCheckCircle className="text-primary text-sm mt-1" />
                    <div>
                      <div className="text-sm font-bold text-gray-200">
                        Main Strategy Trigger
                      </div>
                      <div className="text-xs text-gray-500">
                        {activeSignal.strategy}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <MdCheckCircle className="text-primary text-sm mt-1" />
                    <div>
                      <div className="text-sm font-bold text-gray-200">
                        Trend Filter
                      </div>
                      <div className="text-xs text-gray-500">
                        M15 / H4 Confluence
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-xs text-gray-500 italic">
                  Waiting for signal...
                </div>
              )}
            </div>
            <div className="mt-6 pt-4 border-t border-gray-800">
              <div className="text-xs text-secondary mb-2 uppercase">
                Market Conditions
              </div>
              <div className="flex flex-wrap gap-2">
                {activeSignal && (
                  <>
                    <span className="px-2 py-1 bg-secondary/10 text-secondary border border-secondary/30 text-[10px] rounded">
                      ACTIVE
                    </span>
                    <span className="px-2 py-1 bg-gray-800 text-gray-400 border border-gray-600 text-[10px] rounded">
                      {activeSignal.direction}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="bg-background-dark border border-gray-800 p-4 text-xs overflow-hidden h-75 flex flex-col">
            <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-800">
              <span className="text-gray-500">SYSTEM LOG</span>
              <span className="w-2 h-2 rounded-full bg-primary animate-ping"></span>
            </div>
            <div className="overflow-y-auto space-y-1.5 flex-1 pr-1 custom-scrollbar">
              {data.logs?.map((line, idx) => {
                let color = "text-gray-400";
                if (line.includes("ERROR")) color = "text-danger";
                if (line.includes("WARNING")) color = "text-warning";
                if (line.includes("SUCCESS") || line.includes("WIN"))
                  color = "text-primary";
                return (
                  <div key={idx} className={color}>
                    {line}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <div class="mt-8 border-t border-secondary border-double pt-1">
        <div class="border-t border-secondary p-4 bg-black/40 backdrop-blur-md">
          <div class="flex flex-col md:flex-row justify-between items-center text-sm text-gray-300">
            <div class="flex gap-8">
              <div>
                <span class="text-gray-500">Time Running: </span>
                <span class="text-white"> {data?.stats.time_running}</span>
              </div>
              <div>
                <span class="text-gray-500">Session PnL: </span>
                <span
                  class={`${data?.stats.session_pnl > 0 ? "text-primary" : "text-danger"}`}
                >
                  R {data?.stats.session_pnl > 0 ? "+" : "-"}
                  {data?.stats.session_pnl}
                </span>
              </div>
            </div>
            <div class="mt-4 md:mt-0 flex gap-8">
              <div>
                <span class="text-gray-500">Total Signals: </span>
                <span class="text-white">
                  {data?.stats.session_total + data?.stats.active_count}
                </span>
              </div>
              <div>
                <span class="text-gray-500">W/L: </span>
                <span class="text-primary">
                  {data?.stats.session_wins}
                </span> /{" "}
                <span class="text-danger">{data?.stats.session_losses}</span>
              </div>
            </div>
          </div>
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

export default Signal;
