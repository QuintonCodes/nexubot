import eel
import sys
import os

# Ensure 'src' is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.bot.gui_backend import (
    attempt_login,
    close_app,
    fetch_dashboard_update,
    fetch_signal_updates,
    fetch_trade_history,
    force_close,
    set_mode,
    get_user_settings,
    save_settings,
)


def on_close(page, sockets):
    """
    Modified callback to prevent shutdown during page redirection.
    """
    if not sockets:
        print("❌ Final window closed. Shutting down Nexubot...")
        sys.exit(0)
    else:
        pass


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
        pass  # Expected exit
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_app()
