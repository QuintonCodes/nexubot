import dearpygui.dearpygui as dpg
from pathlib import Path

COLOR_PRIMARY = (0, 255, 65, 255)  # #00FF41 (Matrix Green)
COLOR_SECONDARY = (0, 255, 255, 255)  # #00FFFF (Cyber Cyan)
COLOR_BG_DARK = (5, 5, 5, 255)  # #050505 (Very Dark BG)
COLOR_PANEL_DARK = (10, 10, 10, 255)  # #0a0a0a
COLOR_BORDER = (31, 41, 55, 255)  # #1f2937
COLOR_TEXT_WHITE = (255, 255, 255, 255)
COLOR_TEXT_GRAY = (156, 163, 175, 255)
COLOR_TEXT_DARK = (20, 20, 20, 255)  # For text on primary color
COLOR_RED = (255, 77, 77, 255)


class ThemeManager:

    @staticmethod
    def setup_global_theme():
        """Applies the Cyberpunk/Matrix theme globally."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Window & Background
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLOR_BG_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLOR_PANEL_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_TEXT_WHITE)

                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)

                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, COLOR_BG_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, COLOR_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, COLOR_PRIMARY)

        dpg.bind_theme(global_theme)

    @staticmethod
    def get_input_theme():
        """Theme specifically for Input fields (Dark BG, Light Text)."""
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0, 0, 0, 255))  # Pitch black input bg
                dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_SECONDARY)  # Cyan text typing
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 10)  # Taller inputs
        return t

    @staticmethod
    def get_button_theme():
        """Theme for the main Action Button (Cyan Glow look)."""
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))  # Transparent fill
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 255, 255, 20))  # Slight cyan tint hover
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 255, 255, 50))
                dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_SECONDARY)
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_SECONDARY)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
        return t

    @staticmethod
    def get_badge_theme():
        """Theme for the 'Secure Gateway' text badge."""
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 255, 65, 10))  # Very faint green bg
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 255, 65, 10))  # No hover change
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 255, 65, 10))
                dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_PRIMARY)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 2)
        return t

    @staticmethod
    def get_transparent_border_theme():
        """Used to remove the green border from the background window."""
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0))
        return t

    @staticmethod
    def load_fonts():
        """Loads Geist Mono. Assumes files are in the Project Root."""

        # Resolve project root (two parents up from this file: src/gui -> src -> project root)
        root = Path(__file__).resolve().parents[2]
        font_path = root / "GeistMono-Regular.ttf"
        font_path_bold = root / "GeistMono-Bold.ttf"

        with dpg.font_registry():
            try:
                if not font_path.exists():
                    raise FileNotFoundError(str(font_path))

                default_font = dpg.add_font(str(font_path), 16)

                if font_path_bold.exists():
                    large_font = dpg.add_font(str(font_path_bold), 24)
                    header_font = dpg.add_font(str(font_path_bold), 36)

                small_font = dpg.add_font(str(font_path), 15)

                dpg.bind_font(default_font)
                return default_font, large_font, header_font, small_font
            except Exception:
                print(f"⚠️ Geist Mono font not found at {root}. Using default.")
                return None, None, None, None
