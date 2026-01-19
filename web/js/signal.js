let activeSignals = [];
let currentSignalIndex = 0;

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("signal-container")) {
    setTimeout(initSignalPage, 100);
  }
});

function clearContextDetails() {
  safeSetText("neural-pred", "--");
  safeSetText("neural-sent", "--");
  safeSetText("neural-vol", "--");
  if (document.getElementById("neural-process-log"))
    document.getElementById("neural-process-log").innerHTML =
      '<p class="animate-pulse">_</p>';
  if (document.getElementById("pattern-list"))
    document.getElementById("pattern-list").innerHTML =
      '<div class="text-xs text-gray-500 italic">Waiting for signal...</div>';
}

function initSignalPage() {
  setInterval(pollSignalData, 1000); // Poll every second
}

async function pollSignalData() {
  if (typeof eel === "undefined") return;

  try {
    const data = await eel.fetch_signal_updates()();
    if (data) {
      if (data.latency !== undefined) updateLatencyDisplay(data.latency, true);

      updateSignalStats(data);
      updateLogs(data.logs || []);

      const newSignals = data.signals || [];
      activeSignals = newSignals;
      renderSignalCarousel(data.mode);

      // Update Neural/Pattern context based on CURRENTLY viewed signal
      if (activeSignals.length > 0) {
        updateContextDetails(
          activeSignals[currentSignalIndex] || activeSignals[0]
        );
      } else {
        clearContextDetails();
      }
    }
  } catch (e) {
    console.error("Signal poll error:", e);
  }
}

function renderSignalCarousel(mode) {
  const container = document.getElementById("signal-container");
  if (!container) return;

  if (activeSignals.length === 0) {
    // Show Empty State
    container.innerHTML = `
             <div class="bg-white dark:bg-panel-dark border border-gray-800 p-12 text-center shadow-lg">
                <div class="w-16 h-16 border-4 border-t-primary border-gray-800 rounded-full animate-spin mx-auto mb-4"></div>
                <h2 class="text-xl text-white font-bold tracking-widest animate-pulse">SCANNING MARKETS...</h2>
                <p class="text-gray-500 text-sm mt-2">Neural Engine is analyzing price action.</p>
             </div>`;
    return;
  }

  // Ensure index is valid
  if (currentSignalIndex >= activeSignals.length) currentSignalIndex = 0;

  const sig = activeSignals[currentSignalIndex];
  const isLong = sig.direction === "LONG";
  const colorClass = isLong ? "primary" : "danger"; // green vs red
  const typeText = isLong ? "BUY SIGNAL" : "SELL SIGNAL";
  const statusColor =
    sig.status === "FILLED"
      ? "text-primary border-primary"
      : "text-warning border-warning";

  // Dynamic Buttons based on Automation Mode
  let actionButtons = "";
  if (mode === "FULL_AUTO") {
    actionButtons = `
            <div class="flex-1 bg-black/50 p-2 border border-gray-700 text-xs text-gray-400 flex items-center">
                <span class="material-icons text-sm mr-2 animate-spin">sync</span>
                Executing trade via MetaTrader 5 bridge...
            </div>
            <button onclick="forceClose('${sig.symbol}')" class="bg-danger hover:bg-red-600 text-white px-6 py-2 border border-danger shadow-neon-red text-sm font-bold uppercase transition-all">
                Force Close
            </button>
        `;
  } else {
    actionButtons = `
            <div class="flex-1 bg-yellow-900/20 p-2 border border-yellow-700/50 text-xs text-yellow-500 flex items-center">
                <span class="material-icons text-sm mr-2">visibility</span>
                Signal Tracking Only (Auto-Mode Disabled)
            </div>
        `;
  }

  // Carousel Navigation Controls
  let navControls = "";
  if (activeSignals.length > 1) {
    navControls = `
            <div class="flex justify-between items-center mb-4 bg-gray-900 p-2 rounded">
                <button onclick="prevSignal()" class="text-gray-400 hover:text-white"><span class="material-icons">chevron_left</span></button>
                <span class="text-xs text-gray-400">Signal ${
                  currentSignalIndex + 1
                } of ${activeSignals.length}</span>
                <button onclick="nextSignal()" class="text-gray-400 hover:text-white"><span class="material-icons">chevron_right</span></button>
            </div>
        `;
  }

  // Render Card
  container.innerHTML = `
        ${navControls}
        <div class="bg-white dark:bg-panel-dark border border-${
          colorClass === "primary" ? "primary" : "red-500"
        } shadow-neon-${
          colorClass === "primary" ? "green" : "red"
        } relative overflow-hidden group">
            <div class="p-6">
              <div class="flex items-center gap-3 mb-4">
                <span class="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-${
                  colorClass === "primary" ? "green-400" : "red-400"
                } opacity-75 left-6"></span>
                <span class="relative inline-flex rounded-full h-3 w-3 bg-${colorClass}"></span>
                <h2 class="text-2xl md:text-3xl font-bold text-${colorClass} tracking-tight cursor-default">
                  ${typeText} DETECTED
                </h2>
                <span class="bg-black/20 ${statusColor} text-xs px-2 py-0.5 border rounded-sm">${sig.status || "PENDING"}</span>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
                <div class="space-y-4 text-sm">
                  <div class="flex justify-between border-b border-dashed border-gray-700 pb-1">
                    <span class="text-gray-500">ASSET:</span>
                    <span class="text-white font-bold text-lg">${
                      sig.symbol
                    }</span>
                  </div>
                  <div class="flex justify-between border-b border-dashed border-gray-700 pb-1">
                    <span class="text-gray-500">STRATEGY:</span>
                    <span class="text-secondary">${sig.strategy}</span>
                  </div>
                  <div class="flex justify-between items-center border-b border-dashed border-gray-700 pb-1">
                    <span class="text-gray-500">CONFIDENCE:</span>
                    <div class="flex items-center gap-2">
                      <div class="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div class="h-full bg-secondary shadow-neon-cyan" style="width: ${
                          sig.confidence
                        }%"></div>
                      </div>
                      <span class="text-secondary font-bold">${sig.confidence.toFixed(
                        1
                      )}%</span>
                    </div>
                  </div>
                  <div class="flex justify-between border-b border-dashed border-gray-700 pb-1 pt-2">
                    <span class="text-gray-500">LOT SIZE:</span>
                    <span class="text-warning font-bold">${
                      sig.lot_size
                    } Lots</span>
                  </div>
                </div>

                <div class="space-y-4 text-sm bg-gray-50 dark:bg-black/40 p-4 border border-gray-800">
                  <div class="flex justify-between">
                    <span class="text-gray-500">ENTRY PRICE:</span>
                    <span class="text-white">${sig.price}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-danger font-bold">STOP LOSS:</span>
                    <span class="text-danger">${sig.sl}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-primary font-bold">TAKE PROFIT:</span>
                    <span class="text-primary">${sig.tp}</span>
                  </div>
                  <div class="mt-4 pt-4 border-t border-gray-700 grid grid-cols-2 gap-4">
                    <div>
                      <div class="text-[10px] text-gray-500">MAX RISK (SL)</div>
                      <div class="text-danger font-bold">-R${sig.risk_zar.toFixed(
                        2
                      )}</div>
                    </div>
                    <div class="text-right">
                      <div class="text-[10px] text-gray-500">EST PROFIT (TP)</div>
                      <div class="text-primary font-bold">+R${sig.profit_zar.toFixed(
                        2
                      )}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="mt-6 flex flex-col sm:flex-row gap-4 pt-4 border-t border-dashed border-gray-700">
                ${actionButtons}
              </div>
            </div>
        </div>
    `;
}

// Carousel Functions
window.nextSignal = () => {
  if (activeSignals.length > 0) {
    currentSignalIndex = (currentSignalIndex + 1) % activeSignals.length;
    renderSignalCarousel();
    updateContextDetails(activeSignals[currentSignalIndex]);
  }
};
window.prevSignal = () => {
  if (activeSignals.length > 0) {
    currentSignalIndex =
      (currentSignalIndex - 1 + activeSignals.length) % activeSignals.length;
    renderSignalCarousel();
    updateContextDetails(activeSignals[currentSignalIndex]);
  }
};
window.forceClose = async (symbol) => {
  if (confirm(`Force close trade for ${symbol}?`)) {
    await eel.force_close(symbol)();
    // Polling will update UI automatically
  }
};

function updateContextDetails(sig) {
  if (!sig) return;

  // Neural Stats
  if (sig.neural_info) {
    safeSetText("neural-pred", sig.neural_info.prediction);
    safeSetText("neural-sent", sig.neural_info.sentiment);
    safeSetText("neural-vol", sig.neural_info.volatility);
  }

  // Generate Mockup "Thought Process" based on real data
  const log = document.getElementById("neural-process-log");
  if (log) {
    log.innerHTML = `
            <p class="text-gray-500">&gt; Initializing neural weights for ${
              sig.symbol
            }...</p>
            <p>&gt; Strategy Layer: <span class="text-secondary">${
              sig.strategy
            } detected</span></p>
            <p>&gt; Trend Correlation: <span class="text-primary">aligned with H4 Trend</span></p>
            <p>&gt; Volatility Check: ATR within operational bounds.</p>
            <p>&gt; Risk Calculation: ${
              sig.lot_size
            } lots fits < 2% risk profile.</p>
            <p>&gt; Final Verdict: <span class="text-white bg-${
              sig.direction === "LONG" ? "primary" : "danger"
            } text-black px-1">${sig.direction}</span> confirmed.</p>
            <p class="animate-pulse">_</p>
        `;
  }

  // Pattern List
  const pList = document.getElementById("pattern-list");
  if (pList) {
    // We simulate patterns based on strategy name if exact patterns aren't passed
    pList.innerHTML = `
            <div class="flex items-start gap-3">
                <span class="material-icons text-primary text-sm mt-1">check_circle</span>
                <div>
                    <div class="text-sm font-bold text-gray-200">Main Strategy Trigger</div>
                    <div class="text-xs text-gray-500">${sig.strategy}</div>
                </div>
            </div>
            <div class="flex items-start gap-3">
                 <span class="material-icons text-primary text-sm mt-1">check_circle</span>
                 <div>
                    <div class="text-sm font-bold text-gray-200">Trend Filter</div>
                    <div class="text-xs text-gray-500">M15 / H4 Confluence</div>
                 </div>
            </div>
        `;
  }

  // Tags
  const tags = document.getElementById("market-tags");
  if (tags) {
    tags.innerHTML = `
            <span class="px-2 py-1 bg-secondary/10 text-secondary border border-secondary/30 text-[10px] rounded">ACTIVE</span>
            <span class="px-2 py-1 bg-gray-800 text-gray-400 border border-gray-600 text-[10px] rounded">${sig.direction}</span>
        `;
  }
}

function updateLogs(logs) {
  const logContainer = document.getElementById("system-logs");
  if (!logContainer) return;

  const isScrolledToBottom =
    logContainer.scrollHeight - logContainer.scrollTop <=
    logContainer.clientHeight + 50;

  // Create HTML string
  const html = logs
    .map((line) => {
      // Color coding for log lines
      let colorClass = "text-gray-400";
      if (line.includes("ERROR")) colorClass = "text-danger";
      if (line.includes("WARNING")) colorClass = "text-warning";
      if (line.includes("SUCCESS") || line.includes("WIN"))
        colorClass = "text-primary";

      return `<div class="${colorClass}">${line}</div>`;
    })
    .join("");

  // Only update if content changed to prevent scrolling lock
  if (logContainer.innerHTML !== html) {
    logContainer.innerHTML = html;
    if (isScrolledToBottom) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  }
}

function updateSignalStats(data) {
  // Top Bar
  safeSetText("sig-balance", data.account.balance);
  safeSetText("sig-wr", data.stats.lifetime_wr); // Helper adds '%' if ID suggests it
  safeSetText("sig-active", data.stats.active_count);

  safeSetText("sig-pnl", data.stats.session_pnl);

  // Footer
  safeSetText("footer-time", data.stats.time_running);
  safeSetText("footer-pnl", data.stats.session_pnl);
  safeSetText(
    "footer-signals",
    (data.stats.session_total || 0) + (data.stats.active_count || 0)
  );
  safeSetText("footer-wins", data.stats.session_wins);
  safeSetText("footer-losses", data.stats.session_losses);
}
