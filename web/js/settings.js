document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("setting-login")) {
    initSettingsPage();
  }
});

async function initSettingsPage() {
  if (typeof eel === "undefined") return;

  // 1. Fetch Current Settings
  const settings = await eel.get_user_settings()();
  if (settings) {
    if (document.getElementById("setting-login"))
      document.getElementById("setting-login").value = settings.login || "";
    if (document.getElementById("setting-server"))
      document.getElementById("setting-server").value = settings.server || "";
    if (document.getElementById("setting-password"))
      document.getElementById("setting-password").value =
        settings.password || "";

    if (document.getElementById("setting-lot-size"))
      document.getElementById("setting-lot-size").value =
        settings.lot_size || 0.1;
    if (document.getElementById("setting-risk"))
      document.getElementById("setting-risk").value = settings.risk || 2.0;
    if (document.getElementById("setting-high-vol"))
      document.getElementById("setting-high-vol").checked =
        settings.high_vol || false;

    if (document.getElementById("confidenceRange")) {
      const conf = settings.confidence || 75;
      document.getElementById("confidenceRange").value = conf;
      document.getElementById("confidenceVal").innerText = conf + "%";
    }
  }

  // 2. Bind Save Button
  const form = document.querySelector("form");
  if (form) {
    form.onsubmit = handleSaveSettings;
  }
}

async function handleSaveSettings(e) {
  e.preventDefault();
  const btn = document.querySelector("button[type=submit]"); // or the big button
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
