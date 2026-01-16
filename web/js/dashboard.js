let perfChart;

document.addEventListener("DOMContentLoaded", () => {
  // Dashboard Page
  if (document.getElementById("performanceChart")) {
    setTimeout(initDashboard, 100);
  }
});

function initDashboard() {
  const ctx = document.getElementById("performanceChart").getContext("2d");
  let gradient = ctx.createLinearGradient(0, 0, 0, 400);
  gradient.addColorStop(0, "rgba(0, 255, 65, 0.2)");
  gradient.addColorStop(1, "rgba(0, 255, 65, 0)");

  perfChart = new Chart(ctx, {
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
          ticks: { color: "#666", font: { size: 10, family: "Geist Mono" } },
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#666", font: { size: 10, family: "Geist Mono" } },
        },
      },
    },
  });

  // 2. Setup Toggle Listener
  const toggle = document.getElementById("mode-toggle");
  const modeText = document.getElementById("mode-text");

  function setModeUI(isAuto) {
    if (!modeText) return;
    if (isAuto) {
      modeText.textContent = "FULL AUTOMATION";
      modeText.classList.remove("text-warning");
      modeText.classList.add("text-primary");
    } else {
      modeText.textContent = "SIGNAL ONLY";
      modeText.classList.remove("text-primary");
      modeText.classList.add("text-warning");
    }
  }

  // Initial Sync
  if (toggle) {
    toggle.disabled = true; // Disable until initial state is known to prevent lag/jitter
    eel
      .fetch_dashboard_update()()
      .then((data) => {
        if (data && typeof data.mode !== "undefined") {
          toggle.checked = data.mode === "FULL_AUTO";
          setModeUI(toggle.checked);
        }
        toggle.disabled = false;
      });

    toggle.addEventListener("change", async (e) => {
      const newState = e.target.checked;

      // Optimistic Update (Instant feedback)
      setModeUI(newState);

      try {
        await eel.set_mode(newState)();
      } catch (err) {
        console.error("Failed to set mode:", err);
        // Revert on failure
        toggle.checked = !newState;
        setModeUI(!newState);
      }
    });
  }

  // 3. Start Data Polling
  refreshStats();
  setInterval(refreshStats, 2000);
}

async function refreshStats() {
  if (typeof eel === "undefined") return;

  const startTime = Date.now();

  try {
    const data = await eel.fetch_dashboard_update()();

    // Latency Calculation
    const latency = Date.now() - startTime;
    updateLatencyDisplay(latency, true);

    if (data) {
      updateTextStats(data);
      updateChart(data);
      if (data.recent_trades) {
        updateRecentTradesTable(data.recent_trades);
      }
    }
  } catch (e) {
    console.error("Stats refresh error:", e);
    // show offline latency on failure
    updateLatencyDisplay(null, false);
  }
}

function updateTextStats(data) {
  if (document.getElementById("balance-display")) {
    document.getElementById(
      "balance-display"
    ).innerText = `R ${data.balance.toLocaleString(undefined, {
      minimumFractionDigits: 2,
    })}`;
  }
  if (document.getElementById("equity-display")) {
    document.getElementById(
      "equity-display"
    ).innerText = `R ${data.equity.toLocaleString(undefined, {
      minimumFractionDigits: 2,
    })}`;
  }

  const pnlEl = document.getElementById("pnl-display");
  if (pnlEl) {
    pnlEl.innerText = `R ${data.total_pnl.toLocaleString(undefined, {
      minimumFractionDigits: 2,
    })}`;
    pnlEl.className =
      data.total_pnl >= 0
        ? "text-2xl font-bold text-primary tracking-tight"
        : "text-2xl font-bold text-danger tracking-tight";
  }

  if (document.getElementById("wr-display")) {
    document.getElementById("wr-display").innerText = `${data.win_rate.toFixed(
      1
    )}%`;
  }

  if (document.getElementById("wl-count-display")) {
    document.getElementById(
      "wl-count-display"
    ).innerText = `${data.wins} Wins / ${data.losses} Losses`;
  }
}

function updateChart(data) {
  if (perfChart && data.chart_data && data.chart_data.length > 0) {
    const currentLen = perfChart.data.labels.length;
    // Simple check to avoid redrawing same data
    if (
      currentLen !== data.chart_labels.length ||
      perfChart.data.datasets[0].data[currentLen - 1] !==
        data.chart_data[data.chart_data.length - 1]
    ) {
      perfChart.data.labels = data.chart_labels;
      perfChart.data.datasets[0].data = data.chart_data;
      perfChart.update();
    }
  }
}

function updateRecentTradesTable(trades) {
  const tbody = document.getElementById("recent-trades-body");
  if (!tbody) return;

  tbody.innerHTML = ""; // Clear existing rows

  trades.forEach((trade) => {
    const isWin = trade.result === 1;
    const pnlClass = isWin ? "text-primary" : "text-danger";
    const sign = trade.pnl >= 0 ? "+" : "-";
    const statusText = isWin ? "TP Hit" : "SL Hit"; // Or "Closed" based on your preference
    const statusIcon = isWin ? "check" : "close";
    const statusColor = isWin ? "primary" : "danger";

    const row = `
            <tr class="border-b border-gray-800 hover:bg-white/5 transition-colors group">
                <td class="py-3 pl-2 text-xs text-gray-500">${trade.time}</td>
                <td class="py-3 font-bold text-white group-hover:text-secondary">${
                  trade.symbol
                }</td>
                <td class="py-3">
                    <span class="text-${
                      trade.signal_type === "BUY" ? "primary" : "danger"
                    } bg-${
      trade.signal_type === "BUY" ? "primary" : "danger"
    }/10 px-1.5 py-0.5 text-[10px] rounded border border-${
      trade.signal_type === "BUY" ? "primary" : "danger"
    }/20">${trade.signal_type}</span>
                </td>
                <td class="py-3">${trade.entry.toFixed(5)}</td>
                <td class="py-3">${trade.exit.toFixed(5)}</td>
                <td class="py-3 text-xs">${trade.size}</td>
                <td class="py-3 pr-2 text-right ${pnlClass} font-bold">${sign} R ${trade.pnl.toFixed(
      2
    )}</td>
                <td class="py-3 pr-2 text-right">
                    <span class="text-${statusColor} text-[10px] uppercase border border-${statusColor}/30 px-2 py-0.5 rounded-full flex items-center justify-end gap-1 ml-auto w-fit">
                        <span class="material-symbols-outlined text-[10px]">${statusIcon}</span>
                        ${statusText}
                    </span>
                </td>
            </tr>
        `;
    tbody.innerHTML += row;
  });
}
