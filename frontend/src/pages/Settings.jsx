import { useState } from "react";
import {
  MdBadge,
  MdCloudDownload,
  MdDns,
  MdLink,
  MdPsychology,
  MdRefresh,
  MdSettingsSuggest,
  MdShield,
  MdStorage,
  MdSync,
  MdVpnKey,
} from "react-icons/md";
import { useNavigate } from "react-router-dom";

import { useSaveSettings, useSettingsData } from "../hooks/useEelQuery";
import { callEel } from "../lib/eel";

export default function Settings() {
  const navigate = useNavigate();
  const {
    data: initialSettings,
    isLoading,
    isPlaceholderData,
  } = useSettingsData();
  const { mutate: saveSettings, isPending } = useSaveSettings();

  if (isLoading || isPlaceholderData) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background-dark text-primary">
        <div className="flex flex-col items-center gap-4">
          <MdSync className="animate-spin text-4xl" />
          <span className="animate-pulse text-sm uppercase tracking-widest">
            Loading Configuration...
          </span>
        </div>
      </div>
    );
  }

  return (
    <SettingsForm
      initialData={initialSettings}
      saveSettings={saveSettings}
      isPending={isPending}
      navigate={navigate}
    />
  );
}

function SettingsForm({ initialData, saveSettings, isPending, navigate }) {
  const [form, setForm] = useState({
    login: initialData.login || "",
    server: initialData.server || "",
    password: initialData.password || "",
    lot_size: initialData.lot_size ?? 0.1,
    risk: initialData.risk ?? 2.0,
    high_vol: initialData.high_vol ?? false,
    confidence: initialData.confidence ?? 75,
  });

  const [trainSymbol, setTrainSymbol] = useState("");
  const [trainingTriggered, setTrainingTriggered] = useState(false);

  const handleChange = (e) => {
    const { id, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [id]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    saveSettings(form, {
      onSuccess: () => {
        setTimeout(() => {
          navigate("/signal");
        }, 1500);
      },
      onError: () => {
        alert("Failed to save settings.");
      },
    });
  };

  const handleManualTrain = async (symbol = null) => {
    if (
      confirm(
        symbol
          ? `Refresh data for ${symbol}?`
          : "Refresh ALL data and retrain? This may take time.",
      )
    ) {
      setTrainingTriggered(true);
      try {
        await callEel("trigger_training", symbol);
        alert("Training process started in background.");
      } catch (e) {
        console.error(e);
        alert("Failed to trigger training.");
      } finally {
        setTimeout(() => setTrainingTriggered(false), 2000);
      }
    }
  };

  // Helper for Neural Meta info
  const meta = initialData?.neural_meta || {
    model: "--",
    epochs: "--",
    bias: "--",
  };

  return (
    <div className="mx-auto max-w-5xl py-10 animate-in fade-in duration-500">
      <div className="mb-8 flex items-end justify-between border-b border-gray-800 pb-4">
        <h1 className="text-3xl font-bold uppercase tracking-tight text-white">
          Configuration Settings
        </h1>
        <div className="animate-pulse text-xs text-secondary">
          &lt;EDIT MODE ACTIVE&gt;
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* MT5 Connectivity Panel */}
          <div className="bg-panel-dark group relative overflow-hidden border border-gray-800 p-1">
            <div className="absolute top-0 right-0 h-16 w-16 bg-linear-to-bl from-secondary/10 to-transparent"></div>
            <div className="h-full border border-gray-800/50 bg-background-dark/50 p-6 transition-all group-hover:border-secondary/30">
              <h3 className="mb-6 flex items-center gap-2 border-b border-gray-800 pb-2 text-lg font-bold text-secondary">
                <MdLink className="text-sm" /> MT5 CONNECTIVITY
              </h3>

              <div className="space-y-5">
                <div className="space-y-1">
                  <label className="block text-[10px] uppercase tracking-wider text-gray-500">
                    MT5 Login Number
                  </label>
                  <div className="relative">
                    <MdBadge className="absolute top-3 left-3 text-lg text-gray-600" />
                    <input
                      id="login"
                      type="text"
                      value={form.login}
                      onChange={handleChange}
                      className="w-full rounded-none border border-gray-700 bg-black py-2 pl-9 text-sm text-gray-300 placeholder-gray-600 transition-colors focus:border-secondary focus:ring-1 focus:ring-secondary"
                      placeholder="Enter Login ID"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="block text-[10px] uppercase tracking-wider text-gray-500">
                    MT5 Server
                  </label>
                  <div className="relative">
                    <MdDns className="absolute top-3 left-3 text-lg text-gray-600" />
                    <input
                      id="server"
                      type="text"
                      value={form.server}
                      onChange={handleChange}
                      className="w-full rounded-none border border-gray-700 bg-black py-2 pl-9 text-sm text-gray-300 placeholder-gray-600 transition-colors focus:border-secondary focus:ring-1 focus:ring-secondary"
                      placeholder="e.g. Broker-Server"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="block text-[10px] uppercase tracking-wider text-gray-500">
                    Account Password
                  </label>
                  <div className="relative">
                    <MdVpnKey className="absolute top-3 left-3 text-lg text-gray-600" />
                    <input
                      id="password"
                      type="password"
                      value={form.password}
                      onChange={handleChange}
                      className="w-full rounded-none border border-gray-700 bg-black py-2 pl-9 text-sm text-gray-300 placeholder-gray-600 transition-colors focus:border-secondary focus:ring-1 focus:ring-secondary"
                      placeholder="••••••••"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Risk Management Panel */}
          <div className="bg-panel-dark group relative overflow-hidden border border-gray-800 p-1">
            <div className="absolute top-0 right-0 h-16 w-16 bg-linear-to-bl from-primary/10 to-transparent"></div>
            <div className="h-full border border-gray-800/50 bg-background-dark/50 p-6 transition-all group-hover:border-primary/30">
              <h3 className="mb-6 flex items-center gap-2 border-b border-gray-800 pb-2 text-lg font-bold text-primary">
                <MdShield className="text-sm" /> RISK MANAGEMENT
              </h3>

              <div className="space-y-5">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="block text-[10px] uppercase tracking-wider text-gray-500">
                      Max Lot Size
                    </label>
                    <input
                      id="lot_size"
                      type="number"
                      step="0.01"
                      value={form.lot_size}
                      onChange={handleChange}
                      className="w-full rounded-none border border-gray-700 bg-black px-3 py-2 text-sm font-bold text-white transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  <div className="space-y-1">
                    <label className="block text-[10px] uppercase tracking-wider text-gray-500">
                      Risk Per Trade %
                    </label>
                    <input
                      id="risk"
                      type="number"
                      step="0.01"
                      value={form.risk}
                      onChange={handleChange}
                      className="w-full rounded-none border border-gray-700 bg-black px-3 py-2 text-sm font-bold text-white transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
                    />
                  </div>
                </div>

                <div className="border-t border-dashed border-gray-800 pt-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-bold text-gray-300">
                        Allow High Volatility Pairs
                      </div>
                      <div className="text-[10px] text-gray-500">
                        Trade XAUUSD, GBPJPY, etc.
                      </div>
                    </div>

                    <label className="relative flex cursor-pointer items-center">
                      <input
                        id="high_vol"
                        type="checkbox"
                        className="peer sr-only"
                        checked={form.high_vol}
                        onChange={handleChange}
                      />
                      <div className="peer h-6 w-11 rounded-none border border-gray-600 after:absolute after:top-0.5 after:left-0.5 after:h-5 after:w-5 after:border after:border-gray-300 after:bg-gray-500 after:transition-all after:content-[''] peer-checked:border-primary peer-checked:bg-primary/20 peer-checked:after:translate-x-full peer-checked:after:border-white peer-checked:after:bg-primary peer-checked:after:shadow-[0_0_10px_#00FF41] peer-focus:outline-none bg-black"></div>
                    </label>
                  </div>
                </div>

                <div className="border-l-2 border-warning bg-gray-900/50 p-3 text-[10px] text-gray-400">
                  WARNING: High volatility scaling may increase drawdown during
                  news events.
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* NEW: Data & Model Management Panel */}
        <div className="bg-panel-dark relative border border-gray-800 p-1">
          <div className="border border-gray-800/50 bg-background-dark/50 p-6">
            <h3 className="mb-6 flex items-center gap-2 text-lg font-bold text-white">
              <MdStorage className="text-sm text-purple-400" /> DATA & MODEL
              MANAGEMENT
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Left: Global Operations */}
              <div className="space-y-4">
                <div className="text-sm text-gray-400">
                  Perform a full backfill on all symbols in your Market Watch,
                  clean the dataset, and retrain the Neural Network.
                </div>
                <button
                  type="button"
                  onClick={() => handleManualTrain(null)}
                  disabled={trainingTriggered}
                  className="flex w-full items-center justify-center gap-2 bg-purple-500/10 border border-purple-500/50 hover:bg-purple-500/20 text-purple-300 py-3 px-4 uppercase text-xs font-bold tracking-wider transition-all disabled:opacity-50 cursor-pointer"
                >
                  {trainingTriggered ? (
                    <MdSync className="animate-spin" />
                  ) : (
                    <MdCloudDownload />
                  )}
                  Refresh All Data & Retrain
                </button>
              </div>

              {/* Right: Single Symbol */}
              <div className="space-y-4 border-l border-gray-800 pl-8">
                <div className="text-sm text-gray-400">
                  Partial update: Refresh data for a specific symbol only (e.g.,
                  BTCUSD).
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={trainSymbol}
                    onChange={(e) => setTrainSymbol(e.target.value)}
                    placeholder="SYMBOL (e.g. EURUSD)"
                    className="w-full bg-black border border-gray-700 px-3 text-sm text-white placeholder-gray-600 focus:border-purple-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => handleManualTrain(trainSymbol)}
                    disabled={!trainSymbol || trainingTriggered}
                    className="bg-gray-800 hover:bg-gray-700 border border-gray-600 text-white px-4 py-2 uppercase text-xs font-bold transition-all disabled:opacity-50 cursor-pointer"
                  >
                    <MdRefresh className="text-lg" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Neural Engine Panel */}
        <div className="bg-panel-dark relative border border-gray-800 p-1">
          <div className="border border-gray-800/50 bg-background-dark/50 p-6">
            <h3 className="mb-6 flex items-center gap-2 text-lg font-bold text-white">
              <MdPsychology className="text-sm text-secondary" /> NEURAL ENGINE
              CALIBRATION
            </h3>
            <div className="flex flex-col items-center gap-8 md:flex-row">
              <div className="w-full space-y-6 md:w-2/3">
                <div className="mb-2 flex items-end justify-between">
                  <label className="text-xs uppercase tracking-wider text-gray-400">
                    Confidence Threshold
                  </label>
                  <span className="text-xl font-bold text-secondary">
                    {form.confidence}%
                  </span>
                </div>
                <div className="relative flex h-8 w-full items-center">
                  <div className="absolute top-1/2 z-0 h-px w-full -translate-y-1/2 bg-gray-800"></div>
                  <div className="absolute top-full mt-1 flex w-full justify-between px-1">
                    {[...Array(5)].map((_, i) => (
                      <span key={i} className="h-1 w-px bg-gray-700"></span>
                    ))}
                  </div>
                  <input
                    id="confidence"
                    type="range"
                    min="50"
                    max="99"
                    value={form.confidence}
                    onChange={handleChange}
                    className="relative z-10 w-full cursor-pointer accent-secondary"
                  />
                </div>
                <div className="flex justify-between text-[10px] text-gray-600">
                  <span>AGGRESSIVE (50%)</span>
                  <span>BALANCED (75%)</span>
                  <span>PRECISION (99%)</span>
                </div>
              </div>

              <div className="w-full border border-gray-800 bg-black p-4 text-xs md:w-1/3">
                <div className="mb-2 border-b border-gray-800 pb-1 text-gray-500">
                  PREDICTION LOGIC
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-primary">{meta.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Epochs:</span>
                    <span className="text-white">{meta.epochs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Bias:</span>
                    <span className="text-danger">{meta.bias}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center gap-4 pt-6">
          <button
            type="submit"
            disabled={isPending}
            className="group relative overflow-hidden rounded-sm border border-primary bg-primary px-12 py-4 text-lg font-bold uppercase tracking-widest text-black shadow-neon-green transition-all hover:scale-[1.02] hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
          >
            <span className="absolute inset-0 -translate-x-full bg-linear-to-r from-transparent via-white/40 to-transparent group-hover:animate-shimmer"></span>
            <span className="flex items-center gap-3">
              {isPending ? (
                <>
                  <MdSync className="animate-spin" /> Saving & Restarting...
                </>
              ) : (
                <>
                  <MdSettingsSuggest /> Save and Restart Engine
                </>
              )}
            </span>
          </button>

          <div className="text-[10px] text-gray-500">
            * Engine restart required to apply new neural weights.
          </div>
        </div>
      </form>
    </div>
  );
}
