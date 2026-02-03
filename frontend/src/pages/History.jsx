import { useState } from "react";
import {
  MdAccountBalance,
  MdAccountBalanceWallet,
  MdCalendarToday,
  MdChevronLeft,
  MdChevronRight,
  MdEmojiEvents,
  MdEvent,
  MdFilterAlt,
  MdHistory,
  MdShowChart,
} from "react-icons/md";
import { useHistoryData } from "../hooks/useEelQuery";

export default function History() {
  const [dateRange, setDateRange] = useState({ start: "", end: "" });
  const [selectedRangeBtn, setSelectedRangeBtn] = useState("");
  const [assetFilter, setAssetFilter] = useState({
    forex: false,
    crypto: false,
  });
  const [outcomeFilter, setOutcomeFilter] = useState("All Outcomes");

  // --- Active Query State ---
  const [queryFilters, setQueryFilters] = useState({
    page: 1,
    limit: 10,
    range: null,
    startDate: "",
    endDate: "",
    outcome: "ALL",
    assets: [],
  });

  const { data, isFetching } = useHistoryData(queryFilters);

  const stats = data?.stats || {
    balance: 0,
    lifetime_wr: 0,
    total_trades: 0,
    lifetime_pnl: 0,
  };
  const history = data?.history || [];
  const pagination = data?.pagination || {
    current: 1,
    total_pages: 1,
    total_records: 0,
  };

  // --- Handlers ---
  const handleRangeBtn = (range) => {
    setSelectedRangeBtn(range);
    setDateRange({ start: "", end: "" }); // Clear custom dates
    setQueryFilters((prev) => ({
      ...prev,
      page: 1,
      range: range,
      startDate: "",
      endDate: "",
    }));
  };

  const handleApplyFilters = () => {
    if (dateRange.start || dateRange.end) {
      setSelectedRangeBtn("");
    }

    const assets = [];
    if (assetFilter.forex) assets.push("FOREX");
    if (assetFilter.crypto) assets.push("CRYPTO");

    let outcomeVal = "ALL";
    if (outcomeFilter === "Wins Only") outcomeVal = "WINS";
    if (outcomeFilter === "Losses Only") outcomeVal = "LOSSES";

    setQueryFilters({
      page: 1,
      limit: 10,
      range: dateRange.start || dateRange.end ? null : selectedRangeBtn,
      startDate: dateRange.start,
      endDate: dateRange.end,
      outcome: outcomeVal,
      assets: assets,
    });
  };

  const handleDateChange = (e) => {
    const { name, value } = e.target;
    setDateRange((prev) => {
      const newRange = { ...prev, [name]: value };
      setSelectedRangeBtn("");
      // Only trigger query update if both dates are present or empty
      if (newRange.start && newRange.end) {
        setQueryFilters((q) => ({
          ...q,
          range: null,
          startDate: newRange.start,
          endDate: newRange.end,
          page: 1,
        }));
      }
      return newRange;
    });
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= pagination.total_pages) {
      setQueryFilters((prev) => ({ ...prev, page: newPage }));
    }
  };

  const handleOutcomeChange = (e) => {
    const val = e.target.value;
    setOutcomeFilter(val);
    let apiVal = "ALL";
    if (val === "Wins Only") apiVal = "WINS";
    if (val === "Losses Only") apiVal = "LOSSES";
    setQueryFilters((prev) => ({ ...prev, outcome: apiVal, page: 1 }));
  };

  const handleAssetToggle = (type) => {
    setAssetFilter((prev) => {
      const newState = { ...prev, [type]: !prev[type] };
      const assets = [];
      if (newState.forex) assets.push("FOREX");
      if (newState.crypto) assets.push("CRYPTO");
      setQueryFilters((q) => ({
        ...q,
        assets: assets.length > 0 ? assets : [],
        page: 1,
      }));
      return newState;
    });
  };

  const fmtCurrency = (val) => {
    const safeVal = Math.abs(val) < 0.005 ? 0 : val;
    return `R ${(safeVal || 0).toFixed(2)}`;
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

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Top Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          {
            label: "Current Balance",
            val: fmtCurrency(stats.balance),
            icon: <MdAccountBalanceWallet />,
          },
          {
            label: "Win Rate",
            val: `${(stats.lifetime_wr || 0).toFixed(1)}%`,
            icon: <MdEmojiEvents className="text-secondary" />,
          },
          {
            label: "Total Trades",
            val: stats.total_trades || 0,
            icon: <MdHistory />,
          },
          {
            label: "Net PnL",
            val: fmtPnL(stats.lifetime_pnl),
            icon: (
              <MdShowChart
                className={
                  (stats.lifetime_pnl || 0) >= 0
                    ? "text-primary"
                    : "text-danger"
                }
              />
            ),
          },
        ].map((item, i) => (
          <div
            key={i}
            className="bg-panel-dark border border-border-dark p-4 flex justify-between items-center"
          >
            <div>
              <div className="text-[10px] uppercase text-gray-500 tracking-wider mb-1">
                {item.label}
              </div>
              <div className="text-xl font-bold text-white">{item.val}</div>
            </div>
            <div className="text-2xl text-gray-600 bg-gray-900 p-2 rounded">
              {item.icon}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Filter Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-panel-dark border border-border-dark p-5 sticky top-24">
            <div className="flex items-center gap-2 mb-6 border-b border-gray-800 pb-3">
              <MdFilterAlt className="text-secondary text-xl" />
              <h2 className="font-bold text-lg text-white tracking-wide">
                FILTER DATA
              </h2>
            </div>

            <div className="space-y-6">
              {/* Date Range */}
              <div>
                <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider">
                  Date Range
                </label>
                <div className="space-y-2">
                  <div className="relative">
                    <MdCalendarToday className="absolute left-3 top-2.5 text-gray-600" />
                    <input
                      type="date"
                      name="start"
                      className="w-full bg-black border border-gray-800 text-gray-300 text-xs py-2 pl-9 pr-2 focus:border-primary outline-none"
                      value={dateRange.start}
                      onChange={handleDateChange}
                    />
                  </div>
                  <div className="relative">
                    <MdEvent className="absolute left-3 top-2.5 text-gray-600" />
                    <input
                      type="date"
                      name="end"
                      className="w-full bg-black border border-gray-800 text-gray-300 text-xs py-2 pl-9 pr-2 focus:border-primary outline-none"
                      value={dateRange.end}
                      onChange={handleDateChange}
                    />
                  </div>
                </div>

                {/* Quick Range Buttons */}
                <div className="flex gap-2 mt-2">
                  {["24H", "7D", "30D"].map((range) => (
                    <button
                      key={range}
                      onClick={() => handleRangeBtn(range)}
                      className={`flex-1 text-[10px] py-1 border transition-colors ${
                        selectedRangeBtn === range
                          ? "bg-primary text-black border-primary font-bold"
                          : "bg-gray-800 text-gray-400 border-gray-700 hover:bg-gray-700"
                      }`}
                    >
                      {range}
                    </button>
                  ))}
                </div>
              </div>

              {/* Asset Class */}
              <div>
                <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider">
                  Asset Class
                </label>
                <div className="space-y-2">
                  {[
                    { id: "forex", label: "Forex Pairs" },
                    { id: "crypto", label: "Crypto Assets" },
                  ].map((asset) => (
                    <label
                      key={asset.id}
                      className="flex items-center gap-2 cursor-pointer group"
                    >
                      <input
                        type="checkbox"
                        className="form-checkbox bg-black border-gray-700 text-primary rounded-sm focus:ring-0 focus:ring-offset-0"
                        checked={assetFilter[asset.id]}
                        onChange={() => handleAssetToggle(asset.id)}
                      />
                      <span className="text-sm text-gray-400 group-hover:text-white transition-colors">
                        {asset.label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Outcome */}
              <div>
                <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider">
                  Outcome
                </label>
                <select
                  className="w-full bg-black border border-gray-800 text-gray-300 text-xs py-2 px-3 focus:border-primary outline-none"
                  value={outcomeFilter}
                  onChange={handleOutcomeChange}
                >
                  <option>All Outcomes</option>
                  <option>Wins Only</option>
                  <option>Losses Only</option>
                </select>
              </div>

              <button
                onClick={handleApplyFilters}
                className="w-full mt-4 bg-white/5 hover:bg-primary hover:text-black text-primary border border-primary/50 py-2 text-xs font-bold uppercase tracking-widest transition-all shadow-neon-green/20 hover:shadow-neon-green"
              >
                Apply Filters
              </button>
            </div>
          </div>
        </div>

        {/* Data Table */}
        <div className="lg:col-span-3">
          <div className="bg-panel-dark border border-border-dark overflow-hidden relative min-h-150 flex flex-col">
            {/* Table Header Info */}
            <div className="p-4 border-b border-gray-800 flex justify-between items-center bg-black/20">
              <div className="text-xs text-gray-500 uppercase tracking-wider">
                Displaying{" "}
                <span className="text-white">
                  {(pagination.current - 1) * queryFilters.limit + 1}-
                  {Math.min(
                    pagination.current * queryFilters.limit,
                    pagination.total_records,
                  )}
                </span>{" "}
                of{" "}
                <span className="text-white">{pagination.total_records}</span>{" "}
                records
              </div>
              {isFetching && (
                <span className="text-xs text-primary animate-pulse">
                  Refreshing...
                </span>
              )}
            </div>

            {/* Table Body */}
            <div className="overflow-x-auto flex-1">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-gray-500 border-b border-gray-800 bg-black/40">
                    <th className="px-4 py-3 font-medium">Asset</th>
                    <th className="px-4 py-3 font-medium">Type</th>
                    <th className="px-4 py-3 font-medium">Entry / Exit</th>
                    <th className="px-4 py-3 font-medium text-right">
                      PnL (ZAR)
                    </th>
                    <th className="px-4 py-3 font-medium text-center">
                      Confidence
                    </th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {isFetching ? (
                    <tr>
                      <td
                        colSpan="5"
                        className="text-center py-8 text-gray-500 animate-pulse"
                      >
                        Loading Trade History...
                      </td>
                    </tr>
                  ) : history.length === 0 ? (
                    <tr>
                      <td
                        colSpan="5"
                        className="text-center py-8 text-gray-500 italic"
                      >
                        No records found matching filters.
                      </td>
                    </tr>
                  ) : (
                    history.map((t, idx) => {
                      const isWin = t.result === 1;
                      const colorClass = isWin ? "text-primary" : "text-danger";
                      const bgClass = isWin
                        ? "bg-green-900/30 text-green-400 border-green-900/50"
                        : "bg-red-900/30 text-red-400 border-red-900/50";

                      return (
                        <tr
                          key={idx}
                          className={`group border-b border-gray-800/50 hover:bg-white/5 transition-all`}
                        >
                          <td className="px-4 py-3">
                            <div
                              className={`font-bold text-white group-hover:${colorClass} transition-colors`}
                            >
                              {t.symbol}
                            </div>
                            <div className="text-[10px] text-gray-500">
                              {t.time}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${bgClass}`}
                            >
                              {t.signal_type}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-gray-400 text-xs">
                              <span className="text-gray-600">IN:</span>{" "}
                              {t.entry}
                            </div>
                            <div className="text-gray-400 text-xs">
                              <span className="text-gray-600">OUT:</span>{" "}
                              {t.exit}
                            </div>
                          </td>
                          <td className="px-4 py-3 text-right">
                            {fmtPnL(t.pnl)}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <div className="flex items-center justify-center gap-2">
                              <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-secondary"
                                  style={{ width: `${t.confidence}%` }}
                                ></div>
                              </div>
                              <span className="text-secondary text-xs">
                                {t.confidence?.toFixed(0)}%
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div className="border-t border-gray-800 p-3 flex justify-end gap-2 bg-black/40">
              <button
                onClick={() => handlePageChange(pagination.current - 1)}
                disabled={pagination.current === 1}
                className="text-xs px-3 py-2 border border-gray-800 hover:border-gray-600 text-gray-400 disabled:opacity-50 disabled:cursor-not-allowed gap-2 flex items-center cursor-pointer"
              >
                <MdChevronLeft size={20} />
                PREV
              </button>

              {Array.from(
                { length: pagination.total_pages },
                (_, i) => i + 1,
              ).map((pageNum) => (
                <button
                  key={pageNum}
                  onClick={() => handlePageChange(pageNum)}
                  className={`px-3 py-2 text-xs border cursor-pointer ${
                    pageNum === pagination.current
                      ? "bg-primary text-black border-primary font-bold"
                      : "border-gray-800 text-gray-500 hover:text-white"
                  }`}
                >
                  {pageNum}
                </button>
              ))}

              <button
                onClick={() => handlePageChange(pagination.current + 1)}
                disabled={pagination.current === pagination.total_pages}
                className="text-xs px-3 py-2 border border-gray-800 hover:border-gray-600 text-gray-400 disabled:opacity-50 disabled:cursor-not-allowed gap-2 flex items-center cursor-pointer"
              >
                NEXT
                <MdChevronRight size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-8 py-6 text-center text-xs text-gray-700">
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
