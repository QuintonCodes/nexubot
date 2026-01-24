import { useEffect, useState } from "react";
import {
  MdBadge,
  MdDns,
  MdHelpOutline,
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
  const [formData, setFormData] = useState({
    login_id: "",
    server: "",
    password: "",
  });
  const [status, setStatus] = useState({ loading: false, error: null });
  const [latency, setLatency] = useState("--");

  // Latency check on mount (simulated from login.js)
  useEffect(() => {
    const start = Date.now();
    setTimeout(() => {
      setLatency(Date.now() - start);
    }, 100);
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setStatus({ loading: true, error: null });

    try {
      const response = await callEel(
        "attempt_login",
        formData.login_id,
        formData.server,
        formData.password,
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
      });
    }
  };

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
                v1.4.0
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
                        CONNECTING TO NEURAL NET...
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

              <div className="flex justify-between items-center text-[10px] uppercase text-gray-600 pt-2 border-t border-dashed border-gray-800">
                <a
                  className="hover:text-primary transition-colors flex items-center gap-1"
                  href="#"
                >
                  <MdHelpOutline className="text-[12px]" /> Need Assistance?
                </a>
                <span className="text-right">Encrypted via TLS 1.3</span>
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
