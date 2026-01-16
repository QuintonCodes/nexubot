document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", handleLogin);
  }
});

async function handleLogin(e) {
  e.preventDefault();
  const btn = document.getElementById("loginBtn");
  const status = document.getElementById("statusMsg");

  // UI Loading State
  btn.innerHTML =
    '<span class="animate-pulse">CONNECTING TO NEURAL NET...</span>';
  btn.disabled = true;
  status.classList.add("hidden");

  const login_id = document.getElementById("login_id").value;
  const server = document.getElementById("server").value;
  const password = document.getElementById("password").value;

  // Call Python Function via Eel
  try {
    let response = await eel.attempt_login(login_id, server, password)();
    if (response.success) {
      window.location.href = "dashboard.html";
    } else {
      throw new Error(response.message);
    }
  } catch (error) {
    btn.innerHTML =
      '<span class="material-icons text-sm">power_settings_new</span> Initialize Connection';
    btn.disabled = false;
    status.textContent = ">> ERROR: " + (error.message || error);
    status.classList.remove("hidden");
  }
}
