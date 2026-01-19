// 1. Global Dark Mode & Tailwind Config
document.documentElement.classList.add("dark");

if (typeof tailwind !== "undefined") {
  tailwind.config = {
    darkMode: "class",
    theme: {
      extend: {
        colors: {
          primary: "#00FF41",
          secondary: "#00FFFF",
          danger: "#FF0055",
          warning: "#F0E68C",
          "background-light": "#f0f2f5",
          "background-dark": "#050505",
          "panel-dark": "#0a0a0a",
          "border-dark": "#1f2937",
        },
        fontFamily: {
          display: ["Geist Mono", "monospace"],
          body: ["Geist Mono", "monospace"],
        },
        boxShadow: {
          "neon-green": "0 0 10px #00FF41",
          "neon-cyan": "0 0 10px #00FFFF",
          "neon-red": "0 0 5px #ff4d4d",
        },
        backgroundImage: {
          "grid-pattern":
            "linear-gradient(to right, #1f2937 1px, transparent 1px), linear-gradient(to bottom, #1f2937 1px, transparent 1px)",
        },
      },
    },
  };
}

document.addEventListener("DOMContentLoaded", () => {
  // Inject loader
  if (!document.getElementById("global-loader")) {
    const loader = document.createElement("div");
    loader.id = "global-loader";
    loader.className =
      "fixed inset-0 bg-black/80 z-50 flex items-center justify-center opacity-0 pointer-events-none transition-opacity duration-300";
    loader.innerHTML =
      '<div class="flex flex-col items-center gap-4"><div class="w-12 h-12 border-4 border-t-primary border-gray-800 rounded-full animate-spin"></div><div class="text-primary text-xs tracking-widest animate-pulse">SYSTEM PROCESSING</div></div>';
    document.body.appendChild(loader);
  }

  // Intercept Links
  document.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (href && href !== "#" && !href.startsWith("javascript")) {
        e.preventDefault();
        showLoader();
        setTimeout(() => {
          window.location.href = href;
        }, 150);
      }
    });
  });

  // Hide loader on load (if it was showing)
  hideLoader();

  const stopBtn = document.getElementById("stop-btn");
  if (stopBtn) {
    stopBtn.onclick = null;
    stopBtn.addEventListener("click", logoutAndStop);
  }
});

function hideLoader() {
  const loader = document.getElementById("global-loader");
  if (loader) loader.classList.add("opacity-0", "pointer-events-none");
}

function showLoader() {
  const loader = document.getElementById("global-loader");
  if (loader) {
    loader.classList.remove("opacity-0", "pointer-events-none");
  }
}

async function logoutAndStop() {
  if (
    !confirm(
      "Are you sure you want to stop the engine and logout? Active trades will be managed by MT5 settings."
    )
  )
    return;

  showLoader();
  try {
    if (typeof eel !== "undefined") {
      await eel.stop_and_reset()();
    }
  } catch (e) {
    console.error("Logout error:", e);
  }
  // Redirect regardless of backend success to ensure user isn't stuck
  setTimeout(() => {
    window.location.href = "login.html";
  }, 1000);
}

// 2. Shared Helper Functions

/**
 * Safely sets text content of an element by ID.
 * Prevents errors if element is missing on specific page.
 */
function safeSetText(id, value) {
  const el = document.getElementById(id);
  if (!el) return;

  if (value === null || value === undefined) {
    el.innerText = "--";
    return;
  }

  // If it's already a string (e.g., "R 500.00"), just set it
  if (typeof value === "string") {
    el.innerText = value;
    return;
  }

  // Auto-formatting for raw numbers
  if (typeof value === "number") {
    const isPnL = /pnl|profit/i.test(id);
    const isCurrency = /balance|equity|amount/i.test(id) || isPnL;
    const isPercentage = /wr|rate|percent/i.test(id);

    if (isCurrency) {
      // Currency Formatting (R ZAR)
      const sign = isPnL && value > 0 ? "+" : "";
      el.innerText = `${sign}R ${value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`;

      // Color Logic for PnL
      if (isPnL) {
        el.classList.remove(
          "text-primary",
          "text-danger",
          "text-white",
          "text-gray-500"
        );
        if (value >= 0) el.classList.add("text-primary");
        else el.classList.add("text-danger");
      }
    } else if (isPercentage) {
      // Percentage Formatting
      el.innerText = `${value.toFixed(1)}%`;
    } else {
      // Default Number (Integers, Counts)
      el.innerText = value.toLocaleString();
    }
  }
}

/**
 * Updates the latency indicator in the UI.
 * Used by Dashboard and Signal pages.
 */
function updateLatencyDisplay(ms, isOnline) {
  const el = document.getElementById("latency-display");
  if (!el) return;

  const parent = el.parentElement || el;

  if (isOnline && Number.isFinite(ms)) {
    el.textContent = `${ms}`;
    parent.classList.remove("text-danger");
    parent.classList.add("text-primary");

    // Color coding based on strength
    if (ms < 50)
      parent.style.color = "#00FF41"; // Good
    else if (ms < 150)
      parent.style.color = "#F0E68C"; // Okay
    else parent.style.color = "#FF0055"; // Bad
  } else {
    el.textContent = `--`;
    parent.classList.remove("text-primary");
    parent.classList.add("text-danger");
  }
}

// Set footer year dynamically (moved from inline in dashboard.html)
(function setFooterYear() {
  function apply() {
    const el = document.getElementById("footer-year");
    if (el) el.textContent = String(new Date().getFullYear());
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }
})();
