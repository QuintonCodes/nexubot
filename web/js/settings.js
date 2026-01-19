document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("settings-form")) {
    setTimeout(initSettingsPage, 100);
  }
});

async function handleSaveSettings(e) {
  e.preventDefault();
  const btn = document.getElementById("save-settings-btn");
  if (!btn) return;

  const originalText = btn.innerHTML;
  btn.innerHTML =
    '<span class="material-icons animate-spin">sync</span> Saving & Restarting...';
  btn.disabled = true;

  const data = {
    login: document.getElementById("setting-login").value,
    server: document.getElementById("setting-server").value,
    password: document.getElementById("setting-password").value,
    lot_size: parseFloat(document.getElementById("setting-lot-size").value),
    risk: parseFloat(document.getElementById("setting-risk").value),
    high_vol: document.getElementById("setting-high-vol").checked,
    confidence: parseInt(document.getElementById("confidenceRange").value),
  };

  try {
    const success = await eel.save_settings(data)();
    if (success) {
      // Wait a moment for backend restart then redirect
      setTimeout(() => {
        window.location.href = "signal.html";
      }, 2000);
    } else {
      alert("Failed to save settings.");
      btn.innerHTML = originalText;
      btn.disabled = false;
    }
  } catch (err) {
    console.error(err);
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

async function initSettingsPage() {
  if (!document.getElementById("setting-login")) return;
  if (typeof eel === "undefined") return;

  // 1. Fetch Current Settings
  try {
    const settings = await eel.get_user_settings()();
    if (settings) {
      if (settings.latency) updateLatencyDisplay(settings.latency, true);

      setField("setting-login", settings.login);
      setField("setting-server", settings.server);
      setField("setting-password", settings.password);

      setField(
        "setting-lot-size",
        settings.lot_size !== undefined ? settings.lot_size : 0.1
      );
      setField(
        "setting-risk",
        settings.risk !== undefined ? settings.risk : 2.0
      );

      const chk = document.getElementById("setting-high-vol");
      if (chk) chk.checked = settings.high_vol || false;

      const confRange = document.getElementById("confidenceRange");
      if (confRange) {
        const val = settings.confidence || 75;
        confRange.value = val;
        document.getElementById("confidence-val").textContent = val + "%";

        // Attach Listener properly
        confRange.addEventListener("input", (e) => {
          document.getElementById("confidence-val").textContent =
            e.target.value + "%";
        });
      }

      // Neural Prediction Logic (Dynamic)
      if (settings.neural_meta) {
        safeSetText("meta-model", settings.neural_meta.model);
        safeSetText("meta-epochs", settings.neural_meta.epochs);
        safeSetText("meta-bias", settings.neural_meta.bias);
      }
    }
  } catch (error) {
    console.error("Settings Load Error", error);
  }

  // 2. Bind Save Button
  const form = document.querySelector("form");
  if (form) form.onsubmit = handleSaveSettings;
}

function setField(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  const tag = el.tagName && el.tagName.toUpperCase();
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value !== undefined && value !== null ? value : "";
  } else {
    el.innerText = value !== undefined && value !== null ? value : "";
  }
}
