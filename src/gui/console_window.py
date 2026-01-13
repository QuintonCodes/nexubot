import dearpygui.dearpygui as dpg
import sys
import logging
from src.gui.theme_manager import COLOR_PRIMARY


class GUIStreamHandler(logging.Handler):
    """Redirects Python logging to the DPG Console Window."""

    def __init__(self, log_tag):
        super().__init__()
        self.log_tag = log_tag

    def emit(self, record):
        try:
            msg = self.format(record)
            # Safely add text to DPG (must be on main thread, usually fine in callbacks)
            if dpg.does_item_exist(self.log_tag):
                # Add text
                dpg.add_text(msg, parent=self.log_tag, color=(200, 200, 200, 255), wrap=800)
                # Auto scroll logic could go here
                dpg.set_y_scroll(self.log_tag, dpg.get_y_scroll_max(self.log_tag))
        except Exception:
            self.handleError(record)


class ConsoleWindow:
    def __init__(self):
        self.window_tag = "main_console_window"
        self.log_output_tag = "console_log_output"

    def show(self):
        """Displays the Main Bot Console GUI."""
        dpg.add_window(tag=self.window_tag, label="NEXUBOT CONSOLE", width=1280, height=800, pos=[0, 0], no_close=True)

        with dpg.group(parent=self.window_tag):
            # Top Bar
            with dpg.group(horizontal=True):
                dpg.add_text("STATUS:", color=COLOR_PRIMARY)
                dpg.add_text("RUNNING", color=(0, 255, 0, 255))
                dpg.add_spacer(width=20)
                dpg.add_button(label="STOP BOT", callback=lambda: print("Stopping..."))

            dpg.add_separator()

            # Log Area (Scrollable)
            dpg.add_text("SYSTEM LOGS", color=COLOR_PRIMARY)
            with dpg.child_window(
                tag=self.log_output_tag, width=-1, height=-1, border=True, autosize_x=False, autosize_y=False
            ):
                dpg.add_text(">> System Initialized. Waiting for data...", color=COLOR_PRIMARY)

        # Setup Logging Redirection
        self._setup_log_redirect()

    def _setup_log_redirect(self):
        """Attaches the GUI handler to the root logger."""
        logger = logging.getLogger()
        handler = GUIStreamHandler(self.log_output_tag)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also redirect stdout for simple print statements
        sys.stdout.write = lambda s: self._log_stdout(s)

    def _log_stdout(self, text):
        if text.strip():
            if dpg.does_item_exist(self.log_output_tag):
                dpg.add_text(f">> {text.strip()}", parent=self.log_output_tag, color=(100, 255, 100, 255))
