export function getCurrencySymbol(currency) {
  const curr = currency || "USD";
  return curr === "USD" ? "$" : curr === "ZAR" ? "R" : curr + " ";
}

export function fmtCurrency(val, currSym = "$") {
  const safeVal = Math.abs(val) < 0.005 ? 0 : val;
  return `${currSym} ${(safeVal || 0).toFixed(2)}`;
}

export function fmtPnL(val, currSym = "$") {
  if (val === undefined || val === null || Math.abs(val) < 0.005) {
    return <span className="text-white">{currSym} 0.00</span>;
  }
  const isWin = val > 0;
  return (
    <span className={isWin ? "text-primary" : "text-danger"}>
      {isWin ? "+" : "-"}
      {currSym} {Math.abs(val).toFixed(2)}
    </span>
  );
}
