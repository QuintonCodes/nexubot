import dearpygui.dearpygui as dpg
import asyncio
import threading
from src.gui.theme_manager import ThemeManager
from src.gui.login_window import LoginWindow
from src.gui.console_window import ConsoleWindow

from src.bot.console import NexubotConsole

bot_instance = None
login_screen = None


def start_bot_async():
    """Runs the Async Bot in a separate thread so it doesn't freeze the GUI."""
    global bot_instance

    # Create new event loop for the bot thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot_instance = NexubotConsole()

    # Run the bot
    try:
        loop.run_until_complete(bot_instance.start())
    except Exception as e:
        print(f"Bot Error: {e}")
    finally:
        loop.close()


def on_login_complete():
    """Called when user successfully logs in."""
    print("Login successful. Launching Main Console...")

    # Show the main console window
    console_gui = ConsoleWindow()
    console_gui.show()

    # Start the Bot Logic in a background thread
    bot_thread = threading.Thread(target=start_bot_async, daemon=True)
    bot_thread.start()


def resize_callback(sender, app_data):
    """Triggered whenever the viewport size changes."""
    # app_data contains [width, height, minimize_bool, maximize_bool]
    global login_screen
    if login_screen:
        try:
            login_screen.update_positions()
        except:
            pass


def main():
    global login_screen

    dpg.create_context()

    # 1. Setup Theme & Font
    ThemeManager.setup_global_theme()

    # 2. Configure Viewport
    dpg.create_viewport(title="NEXUBOT v1.4.0 [GUI]", width=1280, height=800, small_icon="icon.ico")
    dpg.setup_dearpygui()
    dpg.show_viewport()

    dpg.set_viewport_resize_callback(resize_callback)

    # 3. Show Login Window First
    login_screen = LoginWindow(on_login_success_callback=on_login_complete)
    login_screen.show()

    # 4. Main DPG Loop
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
