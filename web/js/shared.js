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

// 2. Shared Helper Functions

/**
 * Updates the latency indicator in the UI.
 * Used by Dashboard and Signal pages.
 */
function updateLatencyDisplay(ms, isOnline) {
  const el = document.getElementById("latency-display");
  if (!el) return;

  const parent = el.parentElement || el;

  if (isOnline && Number.isFinite(ms)) {
    const rounded = Math.round(ms);
    el.textContent = `${rounded}`;
    parent.classList.remove("text-danger");
    parent.classList.add("text-primary");
  } else {
    el.textContent = `--`;
    parent.classList.remove("text-primary");
    parent.classList.add("text-danger");
  }
}

/**
 * Safely sets text content of an element by ID.
 * Prevents errors if element is missing on specific page.
 */
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
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
