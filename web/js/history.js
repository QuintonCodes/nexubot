document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("history-table-body")) {
    initHistoryPage();
  }
});

// --- HISTORY PAGE LOGIC ---
function initHistoryPage() {
  // Initial Load
  fetchHistory();

  // Bind Apply Button
  const btn = document.getElementById("btn-apply-filters");
  if (btn) {
    btn.addEventListener("click", () => {
      fetchHistory(getFilters());
    });
  }
}

function getFilters() {
  const startDate = document.getElementById("filter-start").value;
  const endDate = document.getElementById("filter-end").value;
  const outcomeVal = document.getElementById("filter-outcome").value;

  // Outcome Map
  let outcome = "ALL";
  if (outcomeVal.includes("Wins")) outcome = "WINS";
  if (outcomeVal.includes("Losses")) outcome = "LOSSES";

  // Asset Checkboxes
  const assets = [];
  if (document.getElementById("chk-forex")?.checked) assets.push("FOREX");
  if (document.getElementById("chk-crypto")?.checked) assets.push("CRYPTO");
  if (document.getElementById("chk-indices")?.checked) assets.push("INDICES");

  return {
    startDate,
    endDate,
    outcome,
    assets,
  };
}

async function fetchHistory(filters = null) {
  if (typeof eel === "undefined") return;

  // UI Loading state
  const tbody = document.getElementById("history-table-body");
  if (tbody)
    tbody.innerHTML =
      '<tr><td colspan="6" class="text-center py-8 text-gray-500 animate-pulse">Querying Neural Ledger...</td></tr>';

  try {
    const data = await eel.fetch_trade_history(filters)();
    if (data) {
      updateHistoryStats(data.stats);
      renderHistoryTable(data.history);
    }
  } catch (e) {
    console.error("History Fetch Error:", e);
    if (tbody)
      tbody.innerHTML =
        '<tr><td colspan="6" class="text-center py-8 text-danger">Database Connection Failed</td></tr>';
  }
}

function updateHistoryStats(stats) {
  if (!stats) return;
  // IDs must match HTML
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.innerText = val;
  };

  set(
    "hist-balance",
    `R ${stats.balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
  );
  set("hist-wr", `${stats.lifetime_wr.toFixed(1)}%`);
  set("hist-total", stats.total_trades);

  const pnlEl = document.getElementById("hist-pnl");
  if (pnlEl) {
    pnlEl.innerText = `${
      stats.lifetime_pnl >= 0 ? "+" : ""
    } R ${stats.lifetime_pnl.toLocaleString(undefined, {
      minimumFractionDigits: 2,
    })}`;
    pnlEl.className =
      stats.lifetime_pnl >= 0
        ? "text-xl font-bold text-primary"
        : "text-xl font-bold text-danger";
  }
}

function renderHistoryTable(trades) {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;

  if (trades.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="6" class="text-center py-8 text-gray-500">No records found matching filters.</td></tr>';
    return;
  }

  tbody.innerHTML = trades
    .map((t) => {
      const isWin = t.result === 1;
      const colorClass = isWin ? "primary" : "danger"; // text-primary / text-danger
      const bgClass = isWin ? "green" : "red"; // for badge bg
      const sign = t.pnl >= 0 ? "+" : "";

      return `
        <tr class="group border-b border-gray-800/50 hover:bg-${
          isWin ? "green" : "red"
        }-900/10 transition-all">
            <td class="px-4 py-3">
                <div class="font-bold text-white group-hover:text-${colorClass} transition-colors">${
        t.symbol
      }</div>
                <div class="text-[10px] text-gray-500">${t.time}</div>
            </td>
            <td class="px-4 py-3">
                <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-${bgClass}-900/30 text-${bgClass}-400 border border-${bgClass}-900/50 uppercase">
                    ${t.signal_type}
                </span>
            </td>
            <td class="px-4 py-3">
                <div class="text-gray-400 text-xs"><span class="text-gray-600">IN:</span> ${
                  t.entry
                }</div>
                <div class="text-gray-400 text-xs"><span class="text-gray-600">OUT:</span> ${
                  t.exit
                }</div>
            </td>
            <td class="px-4 py-3 text-right">
                <div class="font-bold text-${colorClass} text-shadow-sm">${sign} R ${t.pnl.toFixed(
        2
      )}</div>
            </td>
            <td class="px-4 py-3 text-center">
                 <div class="flex items-center justify-center gap-2">
                    <div class="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div class="h-full bg-secondary" style="width: ${
                          t.confidence
                        }%"></div>
                    </div>
                    <span class="text-secondary text-xs">${t.confidence.toFixed(
                      0
                    )}%</span>
                </div>
            </td>
        </tr>
        `;
    })
    .join("");
}
