import dearpygui.dearpygui as dpg
import src.config as config
from src.gui.theme_manager import ThemeManager, COLOR_PRIMARY, COLOR_TEXT_GRAY, COLOR_PANEL_DARK, COLOR_RED


class LoginWindow:

    def __init__(self, on_login_success_callback: None):
        self.on_login_success = on_login_success_callback

        # Tags
        self.window_tag = "login_window_card"
        self.bg_tag = "background_window"
        self.bg_draw_layer = "background_draw_layer"

        self.input_login = "input_mt5_login"
        self.input_pass = "input_mt5_pass"
        self.input_server = "input_mt5_server"
        self.status_text = "login_status_text"

        self.corners_layer_tag = "login_corners_layer"
        self.header_status_tag = "header_status_group"
        self.footer_tag = "footer_group"
        self.footer_stats_tag = "footer_stats_group"
        self.corners_layer_tag = "login_corners_layer"

        # Dimensions (Increased resolution)
        self.card_width = 450
        self.card_height = 550

        # Fonts
        self.font_reg, self.font_lg, self.font_xl, self.font_sm = ThemeManager.load_fonts()

    def show(self):
        """Builds and displays the login window."""

        # 1. Background Window (Acts as the dark canvas)
        # We ensure it starts at 0,0 and fills viewport
        vp_w = dpg.get_viewport_width()
        vp_h = dpg.get_viewport_height()

        with dpg.window(
            tag=self.bg_tag,
            pos=[0, 0],
            width=vp_w,
            height=vp_h,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            no_collapse=True,
            no_bring_to_front_on_focus=True,
        ):
            dpg.bind_item_theme(self.bg_tag, ThemeManager.get_transparent_border_theme())

            # Create a drawing layer for the separator lines
            dpg.add_draw_layer(tag=self.bg_draw_layer)

            # --- HEADER SECTION ---
            with dpg.group(horizontal=True, pos=[20, 15]):
                # Simulate Logo Box (Green Square)
                dpg.add_button(label="N", width=35, height=35)  # Placeholder for icon
                dpg.bind_item_theme(dpg.last_item(), self._create_logo_theme())

                with dpg.group(horizontal=False, horizontal_spacing=0):
                    with dpg.group(horizontal=True):
                        t = dpg.add_text("NEXUBOT")
                        if self.font_lg:
                            dpg.bind_item_font(t, self.font_lg)

                        t_ver = dpg.add_text("v1.4.0", color=COLOR_TEXT_GRAY)
                        if self.font_sm:
                            dpg.bind_item_font(t_ver, self.font_sm)

                    dpg.add_text("SYSTEM ONLINE", color=COLOR_PRIMARY)
                    if self.font_reg:
                        dpg.bind_item_font(t_ver, self.font_reg)

            # --- HEADER STATUS (Right Aligned) ---
            with dpg.group(tag=self.header_status_tag, horizontal=True, pos=[vp_w - 350, 25]):
                # Server Status
                dpg.add_text("●", color=(100, 100, 100, 255))  # Grey dot
                dpg.add_text("SERVER:", color=COLOR_TEXT_GRAY)
                dpg.add_text("DISCONNECTED", color=(255, 255, 255, 255))
                dpg.add_text("|", color=COLOR_TEXT_GRAY)
                dpg.add_text("LATENCY:", color=COLOR_TEXT_GRAY)
                dpg.add_text("--ms", color=COLOR_TEXT_GRAY)

            # --- FOOTER SECTION ---
            # We calculate position in update_positions()
            with dpg.group(tag=self.footer_tag, horizontal=True):
                dpg.add_text("© 2026 NEXUBOT SYSTEMS. ALL RIGHTS RESERVED.", color=COLOR_TEXT_GRAY)

            with dpg.group(tag=self.footer_stats_tag, horizontal=True):
                dpg.add_text("MEM: 14%", color=COLOR_TEXT_GRAY)
                dpg.add_text("NET: IDLE", color=COLOR_TEXT_GRAY)

            # Draw Layer for Corners
            dpg.add_draw_layer(tag=self.corners_layer_tag)

        # 2. Main Login Card
        # Calculate center immediately
        pos_x = (vp_w - self.card_width) // 2
        pos_y = (vp_h - self.card_height) // 2

        with dpg.window(
            label="LOGIN_CARD",
            tag=self.window_tag,
            pos=[pos_x, pos_y],
            width=self.card_width,
            height=self.card_height,
            no_resize=True,
            no_move=True,
            no_collapse=True,
            no_close=True,
            no_title_bar=True,
            no_scrollbar=True,
            no_background=False,
            no_scroll_with_mouse=True,
        ):
            # Apply Card Background Color specifically
            dpg.bind_item_theme(self.window_tag, self._create_card_theme())

            # Ensure inner content never exceeds card width (account for window padding from theme)
            padding_x = 40
            inner_w = max(0, self.card_width - (padding_x * 2) - 20)
            btn_width = 140
            spacer_len = (inner_w - btn_width) / 2

            # --- HEADER BADGE ---
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=max(0, spacer_len))
                btn_badge = dpg.add_button(label="SECURE GATEWAY", width=btn_width)
                dpg.bind_item_theme(btn_badge, ThemeManager.get_badge_theme())

            dpg.add_spacer(height=5)

            # --- MAIN TITLE ---
            self._add_centered_text("SYSTEM ACCESS", inner_w, font=self.font_xl)

            self._add_centered_text(
                "Authenticate to initialize neural trading engine modules.",
                inner_w,
                color=COLOR_TEXT_GRAY,
                font=self.font_sm,
                wrap=True,
            )

            dpg.add_spacer(height=10)

            # --- FORM INPUTS ---
            input_theme = ThemeManager.get_input_theme()

            # Field 1: Login
            dpg.add_text("MT5 LOGIN NUMBER", color=COLOR_PRIMARY)
            if self.font_sm:
                dpg.bind_item_font(dpg.last_item(), self.font_sm)
            inp1 = dpg.add_input_text(
                tag=self.input_login,
                hint="ENTER ID...",
                width=inner_w + 30,
                default_value=str(config.MT5_LOGIN) if config.MT5_LOGIN != 0 else "",
            )
            dpg.bind_item_theme(inp1, input_theme)
            dpg.add_spacer(height=10)

            # Field 2: Server
            dpg.add_text("MT5 SERVER", color=COLOR_PRIMARY)
            if self.font_sm:
                dpg.bind_item_font(dpg.last_item(), self.font_sm)
            inp2 = dpg.add_input_text(
                tag=self.input_server,
                hint="ENTER SERVER...",
                default_value=config.MT5_SERVER,
                width=inner_w + 30,
            )
            dpg.bind_item_theme(inp2, input_theme)
            dpg.add_spacer(height=10)

            # Field 3: Password
            dpg.add_text("ACCESS KEY (PASSWORD)", color=COLOR_PRIMARY)
            if self.font_sm:
                dpg.bind_item_font(dpg.last_item(), self.font_sm)
            inp3 = dpg.add_input_text(
                tag=self.input_pass,
                password=True,
                hint="............",
                width=inner_w + 30,
                default_value=config.MT5_PASSWORD,
            )
            dpg.bind_item_theme(inp3, input_theme)

            dpg.add_spacer(height=10)

            # --- ACTION BUTTON ---
            btn = dpg.add_button(
                label="INITIALIZE CONNECTION", width=inner_w + 30, height=50, callback=self._on_login_click
            )
            dpg.bind_item_theme(btn, ThemeManager.get_button_theme())
            if self.font_lg:
                dpg.bind_item_font(btn, self.font_lg)

            dpg.add_spacer(height=5)

            # --- STATUS & FOOTER ---
            dpg.add_spacer(height=5)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Need Assistance?", color=COLOR_TEXT_GRAY)
                dpg.add_spacer(width=max(0, inner_w - 255))
                dpg.add_text("Encrypted via TLS 1.3", color=COLOR_TEXT_GRAY)
                dpg.bind_item_font(btn, self.font_reg)

            dpg.add_text("", tag=self.status_text, color=COLOR_RED)

        # Trigger initial position calculation
        self.update_positions()

        dpg.set_frame_callback(1, self.update_positions)

    def _create_logo_theme(self):
        """Simple Green Box Theme for logo placeholder"""
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_PRIMARY)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        return t

    def _add_centered_text(self, text, container_width, color=None, font=None, wrap=False):
        """Helper to horizontally center text."""
        try:
            # Approximate character width factors
            char_width = 7  # Default small/regular
            if font == self.font_xl:
                char_width = 18
            elif font == self.font_lg:
                char_width = 12

            txt_len = len(text) * char_width
            indent = (container_width - txt_len) / 2
            if indent < 0:
                indent = 0

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=indent)
                t = dpg.add_text(text, color=color, wrap=container_width if wrap else 0)
                if font:
                    dpg.bind_item_font(t, font)
        except:
            dpg.add_text(text, color=color)

    def _create_card_theme(self):
        """Specific theme for the card itself (Dark Gray, Thin Border)."""
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLOR_PANEL_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (50, 50, 50, 255))
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 40, 40)
        return t

    def _draw_corners(self):
        """Draws the Cyberpunk corner brackets manually."""
        # Ensure the draw layer exists, create if missing
        dpg.delete_item(self.corners_layer_tag, children_only=True)

        x1, y1 = 0, 0

        try:
            r_min = dpg.get_item_rect_min(self.window_tag)

            # Sometimes DPG returns [0,0] even if keys exist
            if r_min[0] == 0 and r_min[1] == 0:
                raise KeyError("Not rendered yet")

            x1, y1 = r_min[0], r_min[1]
        except Exception:
            pos = dpg.get_item_pos(self.window_tag)
            if not pos:
                pos = [0, 0]  # Safety

            x1, y1 = pos[0], pos[1]

        x2 = x1 + self.card_width
        y2 = y1 + self.card_height

        thickness = 3
        len_line = 30
        padding = 30
        col = COLOR_PRIMARY
        dx = -15  # negative -> move left
        dy = -15

        # Draw onto the layer created in show()
        parent_tag = self.corners_layer_tag

        # Top Left
        dpg.draw_polyline(
            [
                (x1 - padding, y1 - padding + len_line),
                (x1 - padding, y1 - padding),
                (x1 - padding + len_line, y1 - padding),
            ],
            color=col,
            thickness=thickness,
            parent=parent_tag,
        )

        # Top Right
        dpg.draw_polyline(
            [
                (x2 + padding - len_line + dx, y1 - padding),
                (x2 + padding + dx, y1 - padding),
                (x2 + padding + dx, y1 - padding + len_line),
            ],
            color=col,
            thickness=thickness,
            parent=parent_tag,
        )

        # Bottom Left
        dpg.draw_polyline(
            [
                (x1 - padding, y2 + padding - len_line + dy),
                (x1 - padding, y2 + padding + dy),
                (x1 - padding + len_line, y2 + padding + dy),
            ],
            color=col,
            thickness=thickness,
            parent=parent_tag,
        )

        # Bottom Right
        dpg.draw_polyline(
            [
                (x2 + padding - len_line + dx, y2 + padding + dy),
                (x2 + padding + dx, y2 + padding + dy),
                (x2 + padding + dx, y2 + padding - len_line + dy),
            ],
            color=col,
            thickness=thickness,
            parent=parent_tag,
        )

    def update_positions(self):
        """Centering Logic: Runs on window resize."""
        """Public method to update positions on resize."""
        if not dpg.does_item_exist(self.window_tag):
            return

        vp_width = dpg.get_viewport_width()
        vp_height = dpg.get_viewport_height()

        # Update Background to fill screen
        dpg.configure_item(self.bg_tag, width=vp_width, height=vp_height, pos=[0, 0])

        # Update Card Position to Center
        pos_x = (vp_width - self.card_width) // 2
        pos_y = (vp_height - self.card_height) // 2
        dpg.configure_item(self.window_tag, pos=[pos_x, pos_y])

        # 3. Header Status (Top Right)
        # Approx width of that group is ~300px
        dpg.set_item_pos(self.header_status_tag, [vp_width - 320, 25])

        # 4. Footer (Bottom)
        # Copyright Left
        footer_y = vp_height - 70
        dpg.set_item_pos(self.footer_tag, [20, footer_y])
        # Stats Right
        dpg.set_item_pos(self.footer_stats_tag, [vp_width - 180, footer_y])

        # 4. Draw Lines (Dynamic Width)
        # Clear old lines
        dpg.delete_item(self.bg_draw_layer, children_only=True)

        # Header Line
        dpg.draw_line((0, 65), (vp_width, 65), color=(40, 40, 40, 255), thickness=1, parent=self.bg_draw_layer)

        # Footer Line (Just above the text)
        dpg.draw_line(
            (0, footer_y - 25),
            (vp_width, footer_y - 25),
            color=(40, 40, 40, 255),
            thickness=1,
            parent=self.bg_draw_layer,
        )

        # Redraw corners at new coordinates
        self._draw_corners()

    def _on_login_click(self):
        """Validates input and updates the running config."""
        login = dpg.get_value(self.input_login)
        password = dpg.get_value(self.input_pass)
        server = dpg.get_value(self.input_server)

        if not login or not password or not server:
            dpg.set_value(self.status_text, ">> ERROR: MISSING CREDENTIALS")
            return

        try:
            # Update Config in Memory
            config.MT5_LOGIN = int(login)
            config.MT5_PASSWORD = password
            config.MT5_SERVER = server

            dpg.set_value(self.status_text, ">> AUTHENTICATED. INITIALIZING NEURAL ENGINE...")

            # Trigger transition
            dpg.delete_item(self.window_tag)
            dpg.delete_item(self.bg_tag)
            self.on_login_success()

        except ValueError:
            dpg.set_value(self.status_text, ">> ERROR: ID MUST BE NUMERIC")
