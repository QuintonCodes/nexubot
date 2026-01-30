import eel
import os
import sys
import time
import warnings

# --- 1. Filter np.object Warning ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.object.*")

# Ensure 'src' is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.bot.gui_backend import (
    attempt_login,
    close_app,
    fetch_dashboard_update,
    fetch_signal_updates,
    fetch_trade_history,
    force_close,
    get_user_settings,
    save_settings,
    set_mode,
    shutdown_bot,
    stop_and_reset,
)
from src.utils.logger import setup_logging

setup_logging()


def on_close(page, sockets):
    """
    Modified callback to prevent shutdown during page redirection.
    """
    time.sleep(1)

    if not sockets:
        print("❌ Final window closed. Shutting down Nexubot...")
        sys.exit(0)
        shutdown_bot()
        # Force kill process to prevent hanging threads
        os._exit(0)


def start_app():
    # Point to the web folder
    eel.init("web")
    print("🚀 Starting Nexubot GUI...")

    # Start the app
    try:
        eel.start(
            "index.html",
            size=(1280, 720),
            position=(100, 100),
            close_callback=on_close,
            cmdline_args=["--disable-http-cache"],
        )
    except (SystemExit, KeyboardInterrupt):
        pass
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_app()
