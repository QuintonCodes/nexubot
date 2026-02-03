import { useEffect, useState } from "react";
import { MdHub } from "react-icons/md";
import { Link, useLocation } from "react-router-dom";

import { callEel } from "./lib/eel";

export default function MainLayout({ children }) {
  const location = useLocation();
  const [latency, setLatency] = useState("--");

  // Latency check on mount (simulated from login.js)
  useEffect(() => {
    const start = Date.now();
    setTimeout(() => {
      setLatency(Date.now() - start);
    }, 100);
  }, []);

  const handleStop = async () => {
    if (confirm("Stop Engine, Save Session and Exit?")) {
      await callEel("shutdown_bot");
      window.close();
    }
  };

  const navLinks = [
    { path: "/dashboard", label: "DASHBOARD" },
    { path: "/signal", label: "INTELLIGENCE & SIGNAL" },
    { path: "/history", label: "HISTORY" },
    { path: "/settings", label: "SETTINGS" },
  ];

  return (
    <div className="min-h-screen bg-background-dark text-gray-300 relative overflow-x-hidden selection:bg-primary selection:text-black">
      {/* Background Effects */}
      <div className="absolute inset-0 z-0 bg-grid-pattern opacity-20 pointer-events-none grid-bg"></div>
      <div className="scanline pointer-events-none block"></div>

      {/* Navigation */}
      <nav className="relative z-10 border-b border-border-dark bg-panel-dark/90 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-10">
              {/* Logo */}
              <div className="flex items-center gap-3 group cursor-pointer">
                <div className="w-10 h-10 border border-primary flex items-center justify-center bg-transparent shadow-neon-green group-hover:bg-primary group-hover:text-black transition-all">
                  <MdHub className="text-xl dark:text-primary text-gray-800 group-hover:text-black transition-colors" />
                </div>
                <div>
                  <h1 className="text-xl font-bold tracking-wider text-white group-hover:text-primary transition-colors">
                    NEXUBOT{" "}
                    <span className="text-xs font-normal text-gray-400 ml-1">
                      v1.4.0
                    </span>
                  </h1>
                  <div className="text-[10px] uppercase tracking-[0.2em] text-primary animate-pulse">
                    System Online
                  </div>
                </div>
              </div>

              {/* Links */}
              <div className="hidden md:flex items-baseline space-x-4">
                {navLinks.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      location.pathname === link.path
                        ? "text-primary border-b-2 border-primary bg-primary/10"
                        : "text-gray-400 hover:text-primary"
                    }`}
                  >
                    {link.label}
                  </Link>
                ))}
              </div>
            </div>

            {/* Right Side */}
            <div className="flex items-center gap-4">
              <div className="hidden md:flex flex-col items-end text-xs text-gray-400">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>{" "}
                  SYSTEM ONLINE
                </span>
                <span className="text-[10px] text-primary">
                  LATENCY: {latency}ms
                </span>
              </div>
              <button
                onClick={handleStop}
                className="bg-primary hover:bg-green-400 text-black font-bold py-1.5 px-4 text-sm border border-primary shadow-neon-green transition-transform hover:scale-105 uppercase tracking-wider cursor-pointer"
              >
                Stop Engine
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="grow relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
}
