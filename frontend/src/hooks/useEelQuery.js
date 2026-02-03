import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { callEel } from "../lib/eel";

// --- DASHBOARD HOOK ---
export function useDashboardData() {
  return useQuery({
    queryKey: ["dashboard"],
    queryFn: () => callEel("fetch_dashboard_update"),
    refetchInterval: 2000,
    staleTime: 0,
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
    staleTime: 0,
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
    },
  });
}

export function useHistoryData(filterParams) {
  return useQuery({
    queryKey: ["history", filterParams], // Refetch when filters/page change
    queryFn: () => callEel("fetch_trade_history", filterParams),
    keepPreviousData: true,
    staleTime: 5000,
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
    refetchOnWindowFocus: false,
    staleTime: Infinity,
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
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (newSettings) => callEel("save_settings", newSettings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
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
