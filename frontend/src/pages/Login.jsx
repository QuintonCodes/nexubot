import { useEffect, useRef, useState } from "react";
import {
  MdBadge,
  MdDns,
  MdHub,
  MdMemory,
  MdNetworkCheck,
  MdPowerSettingsNew,
  MdVpnKey,
} from "react-icons/md";
import { useNavigate } from "react-router-dom";

import { callEel } from "../lib/eel";

export default function Login() {
  const navigate = useNavigate();
  const hasCheckedAutoLogin = useRef(false);

  const [formData, setFormData] = useState({
    login_id: "",
    server: "",
    password: "",
    mt5_path: "",
  });

  const [status, setStatus] = useState({
    loading: false,
    error: null,
    message: "INITIALIZING SYSTEM...",
  });
  const [latency, setLatency] = useState("--");
  const [version, setVersion] = useState("v1.0.0");

  function handleChange(e) {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  }

  async function handleLogin(e) {
    e.preventDefault();
    setStatus({
      loading: true,
      error: null,
      message: "CONNECTING TO NEURAL NET...",
    });
    sessionStorage.removeItem("manual_logout");

    try {
      const response = await callEel(
        "attempt_login",
        formData.login_id,
        formData.server,
        formData.password,
        formData.mt5_path,
      );

      if (response && response.success) {
        navigate("/dashboard");
      } else {
        throw new Error(response?.message || "Login Failed");
      }
    } catch (error) {
      setStatus({
        loading: false,
        error: ">> ERROR: " + (error.message || error),
        message: null,
      });
    }
  }

  // Latency check on mount (simulated from login.js)
  useEffect(() => {
    let isMounted = true;

    // 1. Fetch Backend Version
    callEel("get_app_version")
      .then((v) => {
        if (v && isMounted) setVersion(v);
      })
      .catch(console.error);

    const start = Date.now();
    setTimeout(() => {
      if (isMounted) setLatency(Date.now() - start);
    }, 100);

    // 3. Robust Auto Login Flow
    async function checkAutoLogin() {
      if (hasCheckedAutoLogin.current) return;
      hasCheckedAutoLogin.current = true;

      // Check if user explicitly clicked "Disconnect" to avoid an infinite auto-login loop
      const isManualLogout = sessionStorage.getItem("manual_logout") === "true";

      try {
        setStatus({
          loading: true,
          error: null,
          message: "CHECKING SAVED CREDENTIALS...",
        });
        const settings = await callEel("get_user_settings");

        if (isMounted && settings && settings.login) {
          // Immediately pre-fill the form so details are remembered visually
          setFormData({
            login_id: String(settings.login),
            server: settings.server || "",
            password: settings.password || "",
            mt5_path: settings.mt5_path || "",
          });

          // Only auto-connect if it wasn't a manual logout and we have all required fields
          if (!isManualLogout && settings.server && settings.password) {
            setStatus({
              loading: true,
              error: null,
              message: "AUTO-CONNECTING TO MT5...",
            });

            const response = await callEel(
              "attempt_login",
              settings.login,
              settings.server,
              settings.password,
              settings.mt5_path || "",
            );

            if (isMounted) {
              if (response && response.success) {
                navigate("/dashboard");
              } else {
                setStatus({
                  loading: false,
                  error:
                    response?.message ||
                    "Auto-Login Failed. Please verify and login manually.",
                  message: null,
                });
              }
            }
            return; // Exit early if we handled the auto-login sequence
          }
        }

        // If no valid auto-login was triggered, restore the UI to active state
        if (isMounted) {
          if (isManualLogout) {
            sessionStorage.removeItem("manual_logout"); // Clear flag for next app launch
          }
          setStatus({ loading: false, error: null, message: null });
        }
      } catch (error) {
        if (isMounted) {
          setStatus({
            loading: false,
            error: ">> ERROR: " + (error.message || error),
            message: null,
          });
        }
      }
    }

    checkAutoLogin();

    return () => {
      isMounted = false;
    };
  }, [navigate]);

  return (
    <div className="bg-background-dark text-gray-300 min-h-screen flex flex-col relative overflow-x-hidden transition-colors duration-300">
      {/* Background Effects */}
      <div className="absolute inset-0 z-0 bg-grid-pattern opacity-[0.15] grid-bg pointer-events-none"></div>
      <div className="scanline pointer-events-none block"></div>

      {/* Header */}
      <header className="relative z-20 w-full p-6 flex justify-between items-center border-b border-gray-800 backdrop-blur-sm bg-black/50">
        <div className="flex items-center gap-3 group cursor-pointer">
          <div className="w-10 h-10 border border-primary flex items-center justify-center bg-transparent shadow-neon-green transition-all duration-300 group-hover:bg-primary group-hover:text-black">
            <MdHub className="text-xl text-primary group-hover:text-black transition-colors" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wider text-white group-hover:text-primary transition-colors">
              NEXUBOT
              <span className="text-xs font-normal text-gray-400 ml-1">
                {version}
              </span>
            </h1>
            <div className="text-[10px] uppercase tracking-[0.2em] text-primary block animate-pulse">
              System Online
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-medium">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-gray-600 animate-pulse"></span>
            <span className="text-gray-400">
              SERVER: <span className="text-white">DISCONNECTED</span>
            </span>
          </div>
          <div className="hidden sm:block text-gray-600">|</div>
          <div className="hidden sm:block text-gray-400">
            LATENCY:{" "}
            <span className={latency !== "--" ? "text-primary" : "text-danger"}>
              {latency}
            </span>
            ms
          </div>
        </div>
      </header>

      {/* Main Form */}
      <main className="grow relative z-10 flex items-center justify-center p-4 sm:p-8">
        <div className="w-full max-w-lg relative group">
          {/* Decorative Corners */}
          <div className="absolute -top-3 -left-3 w-6 h-6 border-t-2 border-l-2 border-primary block transition-all duration-500 group-hover:-top-4 group-hover:-left-4"></div>
          <div className="absolute -top-3 -right-3 w-6 h-6 border-t-2 border-r-2 border-primary block transition-all duration-500 group-hover:-top-4 group-hover:-right-4"></div>
          <div className="absolute -bottom-3 -left-3 w-6 h-6 border-b-2 border-l-2 border-primary block transition-all duration-500 group-hover:-bottom-4 group-hover:-left-4"></div>
          <div className="absolute -bottom-3 -right-3 w-6 h-6 border-b-2 border-r-2 border-primary block transition-all duration-500 group-hover:-bottom-4 group-hover:-right-4"></div>

          <div className="bg-panel-dark border border-gray-800 p-8 shadow-none relative overflow-hidden">
            <div className="mb-8 text-center relative">
              <div className="inline-block px-4 py-1 border border-primary/30 bg-primary/5 text-primary text-xs uppercase tracking-widest mb-4 rounded">
                Secure Gateway
              </div>
              <h2 className="text-3xl font-bold text-white mb-2 uppercase tracking-tight">
                System Access
              </h2>
              <p className="text-xs text-gray-400 max-w-xs mx-auto">
                Authenticate to initialize neural trading engine modules.
              </p>
            </div>

            <form className="space-y-6" onSubmit={handleLogin}>
              <div className="group/input relative">
                <label className="block text-xs font-bold text-primary uppercase tracking-wider mb-2 group-focus-within/input:text-secondary transition-colors">
                  MT5 Login Number
                </label>
                <div className="relative">
                  <input
                    id="login_id"
                    value={formData.login_id}
                    onChange={handleChange}
                    className="w-full bg-black border border-gray-700 text-secondary p-3 pl-10 focus:ring-0 focus:border-primary focus:shadow-neon-green transition-all duration-300 placeholder-gray-700 outline-none"
                    type="text"
                    placeholder="ENTER ID..."
                    required
                  />
                  <MdBadge className="absolute left-3 top-3 text-gray-600 text-lg" />
                </div>
              </div>

              <div className="group/input relative">
                <label className="block text-xs font-bold text-primary uppercase tracking-wider mb-2 group-focus-within/input:text-secondary transition-colors">
                  MT5 Server
                </label>
                <div className="relative">
                  <input
                    id="server"
                    value={formData.server}
                    onChange={handleChange}
                    className="w-full bg-black border border-gray-700 text-secondary p-3 pl-10 focus:ring-0 focus:border-primary focus:shadow-neon-green transition-all duration-300 outline-none"
                    type="text"
                    placeholder="e.g. HFMarketsSA-Demo"
                    required
                  />
                  <MdDns className="absolute left-3 top-3 text-gray-600 text-lg" />
                </div>
              </div>

              <div className="group/input relative">
                <label className="block text-xs font-bold text-primary uppercase tracking-wider mb-2 group-focus-within/input:text-secondary transition-colors">
                  Account Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    value={formData.password}
                    onChange={handleChange}
                    className="w-full bg-black border border-gray-700 text-secondary p-3 pl-10 focus:ring-0 focus:border-primary focus:shadow-neon-green transition-all duration-300 placeholder-gray-700 outline-none"
                    type="password"
                    placeholder="••••••••••••"
                    required
                  />
                  <MdVpnKey className="absolute left-3 top-3 text-gray-600 text-lg" />
                </div>
              </div>

              <div className="group/input relative">
                <label className="block text-xs font-bold text-primary uppercase tracking-wider mb-2 group-focus-within/input:text-secondary transition-colors">
                  MT5 Terminal Path
                </label>
                <div className="relative">
                  <input
                    id="mt5_path"
                    value={formData.mt5_path}
                    onChange={handleChange}
                    className="w-full bg-black border border-gray-700 text-secondary p-3 pl-10 focus:ring-0 focus:border-primary focus:shadow-neon-green transition-all duration-300 placeholder-gray-700 outline-none"
                    type="text"
                    placeholder="C:\Program Files\MetaTrader 5\terminal64.exe"
                  />
                  <MdDns className="absolute left-3 top-3 text-gray-600 text-lg" />
                </div>
              </div>

              <div className="pt-4">
                <button
                  id="loginBtn"
                  type="submit"
                  disabled={status.loading}
                  className="w-full relative overflow-hidden group/btn bg-transparent border-2 border-secondary text-secondary hover:bg-secondary hover:text-black font-bold py-4 uppercase tracking-widest transition-all duration-300 shadow-neon-cyan disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  <span className="relative z-10 flex items-center justify-center gap-2">
                    {status.loading ? (
                      <span className="animate-pulse">
                        {status.message || "CONNECTING TO NEURAL NET..."}
                      </span>
                    ) : (
                      <>
                        <MdPowerSettingsNew className="text-lg" /> Initialize
                        Connection
                      </>
                    )}
                  </span>
                  <div className="absolute inset-0 bg-white/10 translate-y-full group-hover/btn:translate-y-0 transition-transform duration-300"></div>
                </button>
              </div>
            </form>

            {status.error && (
              <div
                id="statusMsg"
                className="mt-4 text-center text-xs text-red-500 animate-pulse"
              >
                {status.error}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="relative z-20 w-full p-4 border-t border-gray-800 bg-black/80 backdrop-blur text-xs text-center sm:text-left flex flex-col sm:flex-row justify-between items-center gap-2">
        <div className="text-gray-500">
          © {new Date().getFullYear()} NEXUBOT SYSTEMS. ALL RIGHTS RESERVED.
        </div>
        <div className="flex gap-4 text-gray-600">
          <div className="flex items-center gap-1">
            <MdMemory className="text-[14px]" /> MEM: 14%
          </div>
          <div className="flex items-center gap-1">
            <MdNetworkCheck className="text-[14px]" /> NET: IDLE
          </div>
        </div>
      </footer>
    </div>
  );
}
