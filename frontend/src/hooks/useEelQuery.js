import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { callEel } from "../lib/eel";

// --- DASHBOARD HOOK ---
export function useDashboardData() {
  return useQuery({
    queryKey: ["dashboard"],
    queryFn: () => callEel("fetch_dashboard_update"),
    // Dashboard updates every 2 seconds
    refetchInterval: 2000,
    placeholderData: {
      balance: 0,
      equity: 0,
      total_pnl: 0,
      win_rate: 0,
      wins: 0,
      losses: 0,
      recent_trades: [],
      chart_labels: [],
      chart_data: [],
    },
  });
}

// --- SIGNAL HOOK ---
export function useSignalData() {
  return useQuery({
    queryKey: ["signals"],
    queryFn: () => callEel("fetch_signal_updates"),
    refetchInterval: 1000,
    refetchIntervalInBackground: true,
    placeholderData: {
      account: { balance: 0, equity: 0 },
      stats: {
        active_count: 0,
        session_pnl: 0,
        lifetime_wr: 0,
        time_running: "--",
        session_wins: 0,
        session_losses: 0,
      },
      signals: [],
      logs: [],
      mode: "SIGNAL_ONLY",
    },
  });
}

export function useHistoryData(filterParams) {
  return useQuery({
    queryKey: ["history", filterParams], // Refetch when filters/page change
    queryFn: () => callEel("fetch_trade_history", filterParams),
    keepPreviousData: true, // Keep showing old data while fetching new page
    placeholderData: {
      stats: { balance: 0, lifetime_wr: 0, total_trades: 0, lifetime_pnl: 0 },
      history: [],
      pagination: { current: 1, total_pages: 1, total_records: 0 },
    },
  });
}

export function useSettingsData() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: () => callEel("get_user_settings"),
    // Don't refetch automatically, only on mount or invalidation
    refetchOnWindowFocus: false,
    placeholderData: {
      login: "",
      server: "",
      password: "",
      lot_size: 0.1,
      risk: 2.0,
      high_vol: false,
      confidence: 75,
      neural_meta: { model: "--", epochs: "--", bias: "--" },
    },
  });
}

export function useSaveSettings() {
  return useMutation({
    mutationFn: (newSettings) => callEel("save_settings", newSettings),
  });
}

export function useForceClose() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (symbol) => callEel("force_close", symbol),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["signals"] });
    },
  });
}
