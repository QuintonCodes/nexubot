let currentPage = 1;
const ITEMS_PER_PAGE = 10;
let currentRange = null; // '24H', '7D', '30D'

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("history-table-body")) {
    setTimeout(initHistoryPage, 100);
  }
});

async function fetchHistory(filters = null) {
  if (typeof eel === "undefined") return;

  // Show Loading
  const tbody = document.getElementById("history-table-body");
  if (tbody) {
    tbody.innerHTML =
      '<tr><td colspan="6" class="text-center py-8 text-gray-500 animate-pulse">Querying Neural Ledger...</td></tr>';
  }

  const payload = filters || {
    page: currentPage,
    limit: ITEMS_PER_PAGE,
    outcome:
      document.getElementById("filter-outcome").value === "Wins Only"
        ? "WINS"
        : document.getElementById("filter-outcome").value === "Losses Only"
          ? "LOSSES"
          : "ALL",
    range: currentRange,
    startDate: document.getElementById("filter-start").value,
    endDate: document.getElementById("filter-end").value,
    assets: [], // Add check logic if needed
  };

  try {
    const data = await eel.fetch_trade_history(payload)();
    if (data && data.stats) {
      if (data.latency !== undefined) updateLatencyDisplay(data.latency, true);

      updateHistoryStats(data.stats);
      renderHistoryTable(data.history || []);
      renderPagination(
        data.pagination || { current: 1, total_pages: 1, total_records: 0 }
      );
    } else {
      throw new Error("Empty data returned");
    }
  } catch (e) {
    console.error("History Fetch Error:", e);
    if (tbody)
      tbody.innerHTML =
        '<tr><td colspan="6" class="text-center py-8 text-danger">Database Connection Failed</td></tr>';
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

function initHistoryPage() {
  // Initial Load
  fetchHistory();

  // Filter Buttons
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      // Visual toggle
      document.querySelectorAll(".filter-btn").forEach((b) => {
        b.classList.remove("bg-primary", "text-black", "border-primary");
        b.classList.add("bg-gray-800", "text-gray-400");
      });
      e.target.classList.remove("bg-gray-800", "text-gray-400");
      e.target.classList.add("bg-primary", "text-black", "border-primary");

      currentRange = e.target.dataset.range;

      // Clear custom dates if range is selected
      document.getElementById("filter-start").value = "";
      document.getElementById("filter-end").value = "";

      currentPage = 1;
      fetchHistory();
    });
  });

  const applyBtn = document.getElementById("btn-apply-filters");
  if (applyBtn) {
    applyBtn.addEventListener("click", () => {
      // If dates are present, clear range buttons
      if (
        document.getElementById("filter-start").value ||
        document.getElementById("filter-end").value
      ) {
        currentRange = null;
        document.querySelectorAll(".filter-btn").forEach((b) => {
          b.classList.remove("bg-primary", "text-black");
          b.classList.add("bg-gray-800", "text-gray-400");
        });
      }
      currentPage = 1;
      fetchHistory();
    });
  }
}

function renderHistoryTable(trades) {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;

  if (!trades || trades.length === 0) {
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

function renderPagination(meta) {
  const container = document.getElementById("pagination-controls");
  if (!container) return;

  const start = (meta.current - 1) * ITEMS_PER_PAGE + 1;
  const end = Math.min(meta.current * ITEMS_PER_PAGE, meta.total_records);

  const label = document.getElementById("filtered-hist");
  if (label)
    label.textContent = meta.total_records > 0 ? `${start}-${end}` : "0";

  const total = document.getElementById("hist-total-display");
  if (total) total.textContent = meta.total_records;

  let html = `<button onclick="changePage(${meta.current - 1})" class="text-xs px-3 py-1 border border-gray-800 hover:border-gray-600 ${meta.current === 1 ? "opacity-50 cursor-not-allowed" : ""}" ${meta.current === 1 ? "disabled" : ""}>&lt; PREV</button>`;

  for (let i = 1; i <= meta.total_pages; i++) {
    if (
      i === 1 ||
      i === meta.total_pages ||
      (i >= meta.current - 1 && i <= meta.current + 1)
    ) {
      const activeClass =
        i === meta.current
          ? "bg-primary text-black border-primary font-bold"
          : "border-gray-800 text-gray-500 hover:text-white";
      html += `<button onclick="changePage(${i})" class="px-3 py-1 text-xs border ${activeClass}">${i}</button>`;
    } else if (i === meta.current - 2 || i === meta.current + 2) {
      html += `<span class="px-2 text-gray-600 text-xs">...</span>`;
    }
  }

  html += `<button onclick="changePage(${meta.current + 1})" class="text-xs px-3 py-1 border border-gray-800 hover:border-gray-600 ${meta.current === meta.total_pages ? "opacity-50 cursor-not-allowed" : ""}" ${meta.current === meta.total_pages ? "disabled" : ""}>NEXT &gt;</button>`;

  container.innerHTML = html;
}

window.changePage = (page) => {
  currentPage = page;
  fetchHistory();
};

function updateHistoryStats(stats) {
  if (!stats) return;

  safeSetText("hist-balance", stats.balance);
  safeSetText("hist-wr", stats.lifetime_wr);
  safeSetText("hist-total", stats.total_trades);
  safeSetText("hist-pnl", stats.lifetime_pnl);
}
