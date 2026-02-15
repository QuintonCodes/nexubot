import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { callEel } from "../lib/eel";

let globalDashboardCache = null;
let globalSignalCache = null;

const DEFAULT_SIGNAL_DATA = {
  account: { balance: 0, equity: 0 },
  stats: {
    active_count: 0,
    session_pnl: 0,
    lifetime_wr: 0,
    time_running: "--",
    session_wins: 0,
    session_losses: 0,
    session_total: 0,
  },
  signals: [],
  logs: [],
  mode: "SIGNAL_ONLY",
};

const DEFAULT_DASHBOARD_DATA = {
  balance: 0,
  equity: 0,
  total_pnl: 0,
  win_rate: 0,
  wins: 0,
  losses: 0,
  recent_trades: [],
  chart_labels: [],
  chart_data: [],
  mode: "SIGNAL_ONLY",
  system_status: "IDLE",
};

const DEFAULT_HISTORY_DATA = {
  stats: {
    balance: 0,
    lifetime_wr: 0,
    total_trades: 0,
    lifetime_pnl: 0,
  },
  history: [],
  pagination: { current: 1, total_pages: 1, total_records: 0 },
};

const DEFAULT_SETTINGS_DATA = {
  login: "",
  server: "",
  password: "",
  lot_size: 0.1,
  risk: 2.0,
  high_vol: false,
  confidence: 75,
  neural_meta: { model: "--", epochs: "--", bias: "--" },
};

const isValidDashboardData = (data) => {
  if (!data || typeof data !== "object") return false;
  if (data.balance === 0 && data.equity === 0) return false;
  return true;
};

const isValidSignalData = (data) => {
  if (!data || typeof data !== "object") return false;
  if (data.account?.balance === 0 && data.account?.equity === 0) return false;
  return true;
};

// --- DASHBOARD HOOK ---
export function useDashboardData() {
  const query = useQuery({
    queryKey: ["dashboard"],
    queryFn: async () => {
      const res = await callEel("fetch_dashboard_update");
      if (!isValidDashboardData(res)) {
        throw new Error("Dashboard data not ready");
      }
      return res;
    },
    refetchInterval: 1000,
    refetchIntervalInBackground: true,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  });

  useEffect(() => {
    if (query.data) {
      globalDashboardCache = query.data;
    }
  }, [query.data]);

  const safeData = query.data || globalDashboardCache || DEFAULT_DASHBOARD_DATA;

  return { ...query, data: safeData };
}

// --- SIGNAL HOOK ---
export function useSignalData() {
  const query = useQuery({
    queryKey: ["signals"],
    queryFn: async () => {
      const res = await callEel("fetch_signal_updates");
      if (!isValidSignalData(res)) {
        throw new Error("Signal data not ready");
      }
      return res;
    },
    refetchInterval: 1000,
    refetchIntervalInBackground: true,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  });

  useEffect(() => {
    if (query.data) {
      globalSignalCache = query.data;
    }
  }, [query.data]);

  const safeData = query.data || globalSignalCache || DEFAULT_SIGNAL_DATA;

  return { ...query, data: safeData };
}

export function useHistoryData(filterParams) {
  const query = useQuery({
    queryKey: ["history", filterParams],
    queryFn: async () => {
      const res = await callEel("fetch_trade_history", filterParams);
      if (!res) throw new Error("History fetch failed");
      return res;
    },
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    retry: false,
  });

  return { ...query, data: query.data || DEFAULT_HISTORY_DATA };
}

export function useSettingsData() {
  const query = useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const res = await callEel("get_user_settings");
      if (!res) throw new Error("Settings fetch failed");
      return res;
    },
    refetchOnWindowFocus: false,
    staleTime: Infinity,
  });

  return { ...query, data: query.data || DEFAULT_SETTINGS_DATA };
}

export function useSaveSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (newSettings) =>
      await callEel("save_settings", newSettings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
}

export function useForceClose() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (symbol) => await callEel("force_close", symbol),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["signals"] });
    },
  });
}
