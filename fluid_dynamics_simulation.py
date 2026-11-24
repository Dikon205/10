"""Interactive multi-density fluid simulator using pygame-ce GPU Renderer and Numba."""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pygame
from numba import njit, prange
from pygame._sdl2.video import Renderer, Texture
from scipy.ndimage import distance_transform_edt

from forces import Bounds, FluidPhysicsSolver
from particles import (
    FLUID_PRESETS,
    FluidPreset,
    ParticleSystem,
    color_for_density,
    rgb_to_u8,
)

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    from ctypes import wintypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    _user32 = ctypes.windll.user32

    def get_global_cursor_pos() -> Tuple[int, int]:
        pt = wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
else:
    def get_global_cursor_pos() -> Tuple[int, int]:
        return pygame.mouse.get_pos()


@njit(fastmath=True, cache=True)
def _stamp_dots(
    buf: np.ndarray,
    pos: np.ndarray,
    col: np.ndarray,
    n: int,
    ox: float,
    oy: float,
    w: int,
    h: int,
    radius: int,
) -> None:
    """Stamp dense shaded circular particle dots into (W, H, 3) uint8 buffer."""
    r2_max = (radius + 0.35) * (radius + 0.35)
    inv_r2 = 1.0 / r2_max
    for i in range(n):
        cx = int(pos[i, 0] - ox + 0.5)
        cy = int(pos[i, 1] - oy + 0.5)
        x0 = cx - radius
        x1 = cx + radius
        y0 = cy - radius
        y1 = cy + radius
        if x1 < 0 or x0 >= w or y1 < 0 or y0 >= h:
            continue
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= w:
            x1 = w - 1
        if y1 >= h:
            y1 = h - 1

        cr = col[i, 0] * 255.0
        cg = col[i, 1] * 255.0
        cb = col[i, 2] * 255.0
        for xx in range(x0, x1 + 1):
            dx = xx - cx
            for yy in range(y0, y1 + 1):
                dy = yy - cy
                d2 = dx * dx + dy * dy
                if d2 <= r2_max:
                    if dx <= 0 and dy <= 0 and d2 <= 1.25:
                        s = 1.15
                    else:
                        s = 1.02 - 0.18 * (d2 * inv_r2)
                    rr = int(cr * s)
                    gg = int(cg * s)
                    bb = int(cb * s)
                    buf[xx, yy, 0] = 255 if rr > 255 else rr
                    buf[xx, yy, 1] = 255 if gg > 255 else gg
                    buf[xx, yy, 2] = 255 if bb > 255 else bb


@njit(fastmath=True, cache=True)
def _stamp_pixels(
    buf: np.ndarray,
    pos: np.ndarray,
    col: np.ndarray,
    n: int,
    ox: float,
    oy: float,
    w: int,
    h: int,
    half: int,
) -> None:
    """Stamp dense square fluid pixels with subtle 1px top-left highlight."""
    for i in range(n):
        cx = int(pos[i, 0] - ox + 0.5)
        cy = int(pos[i, 1] - oy + 0.5)
        x0 = cx - half
        x1 = cx + half
        y0 = cy - half
        y1 = cy + half
        if x1 < 0 or x0 >= w or y1 < 0 or y0 >= h:
            continue
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= w:
            x1 = w - 1
        if y1 >= h:
            y1 = h - 1

        cr = col[i, 0] * 255.0
        cg = col[i, 1] * 255.0
        cb = col[i, 2] * 255.0
        for xx in range(x0, x1 + 1):
            for yy in range(y0, y1 + 1):
                if xx == x0 or yy == y0:
                    s = 1.10
                elif xx == x1 or yy == y1:
                    s = 0.84
                else:
                    s = 1.00
                rr = int(cr * s)
                gg = int(cg * s)
                bb = int(cb * s)
                buf[xx, yy, 0] = 255 if rr > 255 else rr
                buf[xx, yy, 1] = 255 if gg > 255 else gg
                buf[xx, yy, 2] = 255 if bb > 255 else bb


@njit(fastmath=True, parallel=True, cache=True)
def _compute_speed_colors(vel: np.ndarray, out: np.ndarray, n: int) -> None:
    """Map particle speeds to navy -> cyan -> emerald -> gold -> white."""
    for i in prange(n):
        vx = vel[i, 0]
        vy = vel[i, 1]
        spd = math.sqrt(vx * vx + vy * vy)
        t = spd * (1.0 / 1100.0)
        if t > 1.0:
            t = 1.0
        if t < 0.35:
            k = t / 0.35
            r = 0.14 + 0.08 * k
            g = 0.36 + 0.44 * k
            b = 0.88 + 0.10 * k
        elif t < 0.70:
            k = (t - 0.35) / 0.35
            r = 0.22 + 0.72 * k
            g = 0.80 + 0.12 * k
            b = 0.98 - 0.68 * k
        else:
            k = (t - 0.70) / 0.30
            r = 0.94 + 0.06 * k
            g = 0.92 - 0.38 * k
            b = 0.30 + 0.55 * k
        out[i, 0] = r
        out[i, 1] = g
        out[i, 2] = b


@njit(fastmath=True, parallel=True, cache=True)
def _splat_fluid_banded(
    dens: np.ndarray,
    col_acc: np.ndarray,
    kernel: np.ndarray,
    pos: np.ndarray,
    col: np.ndarray,
    n: int,
    ox: float,
    oy: float,
    sw: int,
    sh: int,
    radius: int,
) -> None:
    """Parallel race-free column-banded density and color accumulation."""
    bands = 16
    band_w = (sw + bands - 1) // bands
    for b in prange(bands):
        bx0 = b * band_w
        bx1 = bx0 + band_w
        if bx1 > sw:
            bx1 = sw
        for x in range(bx0, bx1):
            for y in range(sh):
                dens[x, y] = 0.0
                col_acc[x, y, 0] = 0.0
                col_acc[x, y, 1] = 0.0
                col_acc[x, y, 2] = 0.0

        for i in range(n):
            cx = int((pos[i, 0] - ox) * 0.5 + 0.5)
            if cx + radius < bx0 or cx - radius >= bx1:
                continue
            cy = int((pos[i, 1] - oy) * 0.5 + 0.5)
            y0 = cy - radius
            y1 = cy + radius
            if y1 < 0 or y0 >= sh:
                continue
            x0 = cx - radius
            x1 = cx + radius
            if x0 < bx0:
                x0 = bx0
            if x1 >= bx1:
                x1 = bx1 - 1
            if y0 < 0:
                y0 = 0
            if y1 >= sh:
                y1 = sh - 1

            cr = col[i, 0]
            cg = col[i, 1]
            cb = col[i, 2]
            for xx in range(x0, x1 + 1):
                kx = xx - cx + radius
                for yy in range(y0, y1 + 1):
                    ky = yy - cy + radius
                    kw = kernel[kx, ky]
                    if kw > 0.0:
                        dens[xx, yy] += kw
                        col_acc[xx, yy, 0] += cr * kw
                        col_acc[xx, yy, 1] += cg * kw
                        col_acc[xx, yy, 2] += cb * kw


@njit(fastmath=True, parallel=True, cache=True)
def _shade_fluid_half(
    buf: np.ndarray,
    bg_buf: np.ndarray,
    dens: np.ndarray,
    col_acc: np.ndarray,
    sw: int,
    sh: int,
) -> None:
    """Shade continuous fluid surface with crisp AA meniscus, depth tint, and specular shine."""
    thresh = 0.65
    aa_w = 0.18
    inv_aa = 1.0 / aa_w
    for x in prange(sw):
        for y in range(sh):
            d = dens[x, y]
            if d <= thresh - aa_w:
                buf[x, y, 0] = bg_buf[x, y, 0]
                buf[x, y, 1] = bg_buf[x, y, 1]
                buf[x, y, 2] = bg_buf[x, y, 2]
            else:
                inv_d = 1.0 / d
                fr = col_acc[x, y, 0] * inv_d * 255.0
                fg = col_acc[x, y, 1] * inv_d * 255.0
                fb = col_acc[x, y, 2] * inv_d * 255.0

                xm = x - 1 if x > 0 else 0
                xp = x + 1 if x + 1 < sw else sw - 1
                ym = y - 1 if y > 0 else 0
                yp = y + 1 if y + 1 < sh else sh - 1
                nx = dens[xm, y] - dens[xp, y]
                ny = dens[x, ym] - dens[x, yp]
                nlen = math.sqrt(nx * nx + ny * ny + 1e-5)

                spec = 0.0
                if d < 2.6:
                    ndotl = (-0.55 * nx - 0.80 * ny) / (nlen if nlen > 0.14 else 0.14)
                    if ndotl > 0.0:
                        nd2 = ndotl * ndotl
                        fade = (2.6 - d) * 0.65
                        if fade > 1.0:
                            fade = 1.0
                        spec = nd2 * nd2 * 105.0 * fade

                depth = 0.86 + 0.15 * min(1.0, d * 0.28)
                rim = max(0.0, min(1.0, (1.25 - d) * 1.1)) * 22.0
                fr = fr * depth + rim + spec
                fg = fg * depth + rim + spec
                fb = fb * depth + rim + spec
                if fr > 255.0:
                    fr = 255.0
                if fg > 255.0:
                    fg = 255.0
                if fb > 255.0:
                    fb = 255.0

                if d >= thresh:
                    buf[x, y, 0] = int(fr)
                    buf[x, y, 1] = int(fg)
                    buf[x, y, 2] = int(fb)
                else:
                    a = (d - (thresh - aa_w)) * inv_aa
                    ia = 1.0 - a
                    buf[x, y, 0] = int(bg_buf[x, y, 0] * ia + fr * a)
                    buf[x, y, 1] = int(bg_buf[x, y, 1] * ia + fg * a)
                    buf[x, y, 2] = int(bg_buf[x, y, 2] * ia + fb * a)


class FluidRenderer:
    """Hardware-accelerated SDL2 GPU texture renderer with parallel Numba fluid shaders."""

    STYLES = ("Fluid", "Dots", "Pixels", "Velocity")

    def __init__(self, gpu_renderer: Renderer, width: int, height: int) -> None:
        self.gpu = gpu_renderer
        self.style_idx = 0
        self.width = 0
        self.height = 0
        self.sw = 0
        self.sh = 0
        self.last_render_ms = 0.0

        self._r_half = 5
        d_k = 2 * self._r_half + 1
        kernel = np.zeros((d_k, d_k), dtype=np.float32)
        denom = (self._r_half + 0.5) ** 2
        for oy in range(-self._r_half, self._r_half + 1):
            for ox in range(-self._r_half, self._r_half + 1):
                d2 = (ox * ox + oy * oy) / denom
                if d2 < 1.0:
                    v = 1.0 - d2
                    kernel[ox + self._r_half, oy + self._r_half] = v * v
        self._kernel = kernel

        self._bg_buf: Optional[np.ndarray] = None
        self._buf: Optional[np.ndarray] = None
        self._full_surf: Optional[pygame.Surface] = None
        self._full_tex: Optional[Texture] = None

        self._half_bg_buf: Optional[np.ndarray] = None
        self._half_buf: Optional[np.ndarray] = None
        self._dens_half: Optional[np.ndarray] = None
        self._col_half: Optional[np.ndarray] = None
        self._half_surf: Optional[pygame.Surface] = None
        self._half_tex: Optional[Texture] = None

        self._speed_col = np.zeros((10000, 3), dtype=np.float32)
        self.resize(max(64, width), max(64, height))

    @property
    def style(self) -> str:
        return self.STYLES[self.style_idx]

    def cycle_style(self) -> str:
        self.style_idx = (self.style_idx + 1) % len(self.STYLES)
        return self.style

    def resize(self, width: int, height: int) -> None:
        width = max(64, int(width))
        height = max(64, int(height))
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        y_t = np.linspace(0.0, 1.0, height, dtype=np.float32)[None, :]
        bg = np.empty((width, height, 3), dtype=np.uint8)
        bg[:, :, 0] = (8.0 + 4.0 * y_t).astype(np.uint8)
        bg[:, :, 1] = (11.0 + 6.0 * y_t).astype(np.uint8)
        bg[:, :, 2] = (17.0 + 9.0 * y_t).astype(np.uint8)
        bg[::48, :, :] += np.array([5, 7, 10], dtype=np.uint8)
        bg[:, ::48, :] += np.array([5, 7, 10], dtype=np.uint8)

        self._bg_buf = bg
        self._buf = np.empty_like(bg)
        self._full_surf = pygame.Surface((width, height))
        self._full_tex = Texture(self.gpu, (width, height), streaming=True)

        self.sw = max(32, width // 2)
        self.sh = max(32, height // 2)
        yh_t = np.linspace(0.0, 1.0, self.sh, dtype=np.float32)[None, :]
        hbg = np.empty((self.sw, self.sh, 3), dtype=np.uint8)
        hbg[:, :, 0] = (8.0 + 4.0 * yh_t).astype(np.uint8)
        hbg[:, :, 1] = (11.0 + 6.0 * yh_t).astype(np.uint8)
        hbg[:, :, 2] = (17.0 + 9.0 * yh_t).astype(np.uint8)
        hbg[::24, :, :] += np.array([5, 7, 10], dtype=np.uint8)
        hbg[:, ::24, :] += np.array([5, 7, 10], dtype=np.uint8)

        self._half_bg_buf = hbg
        self._half_buf = np.empty_like(hbg)
        self._dens_half = np.zeros((self.sw, self.sh), dtype=np.float32)
        self._col_half = np.zeros((self.sw, self.sh, 3), dtype=np.float32)
        self._half_surf = pygame.Surface((self.sw, self.sh))
        self._half_tex = Texture(self.gpu, (self.sw, self.sh), streaming=True)

    def draw_fluid(
        self,
        dest_y: int,
        particles: ParticleSystem,
        origin_x: float,
        origin_y: float,
    ) -> None:
        """Rasterize fluid field/particles and draw via GPU texture."""
        t0 = time.perf_counter()
        n = particles.n
        style = self.style

        if style in ("Fluid", "Velocity"):
            if n > 0:
                if style == "Velocity":
                    if n > len(self._speed_col):
                        self._speed_col = np.zeros((n * 2, 3), dtype=np.float32)
                    _compute_speed_colors(particles.vel, self._speed_col, n)
                    col = self._speed_col[:n]
                else:
                    col = particles.color
                _splat_fluid_banded(
                    self._dens_half,
                    self._col_half,
                    self._kernel,
                    particles.pos,
                    col,
                    n,
                    origin_x,
                    origin_y,
                    self.sw,
                    self.sh,
                    self._r_half,
                )
                _shade_fluid_half(
                    self._half_buf,
                    self._half_bg_buf,
                    self._dens_half,
                    self._col_half,
                    self.sw,
                    self.sh,
                )
            else:
                np.copyto(self._half_buf, self._half_bg_buf)

            pygame.pixelcopy.array_to_surface(self._half_surf, self._half_buf)
            self._half_tex.update(self._half_surf)
            self._half_tex.draw(dstrect=(0, dest_y, self.width, self.height))
        else:
            np.copyto(self._buf, self._bg_buf)
            if n > 0:
                if style == "Dots":
                    _stamp_dots(
                        self._buf, particles.pos, particles.color, n, origin_x, origin_y, self.width, self.height, 4
                    )
                else:
                    _stamp_pixels(
                        self._buf, particles.pos, particles.color, n, origin_x, origin_y, self.width, self.height, 3
                    )
            pygame.pixelcopy.array_to_surface(self._full_surf, self._buf)
            self._full_tex.update(self._full_surf)
            self._full_tex.draw(dstrect=(0, dest_y, self.width, self.height))

        self.last_render_ms = (time.perf_counter() - t0) * 1000.0


class SolidTextObstacles:
    """Manages placed solid text strings, their GPU texture overlay, and 2D SDF collider."""

    def __init__(self, width: int, height: int, gpu: Renderer) -> None:
        self.gpu = gpu
        self.width = max(64, width)
        self.height = max(64, height)
        self.items: List[Tuple[str, float, float, int]] = []  # (text, rel_x, rel_y, font_px)
        self.sdf: Optional[np.ndarray] = None
        self.tex: Optional[Texture] = None
        self.overlay_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

    def resize(self, width: int, height: int) -> None:
        width = max(64, width)
        height = max(64, height)
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        self.overlay_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if self.items:
            self.rebuild()

    def add_text(self, text: str, tank_x: float, tank_y: float, font_px: int) -> None:
        if not text.strip():
            return
        rel_x = max(0.05, min(0.95, tank_x / max(1.0, float(self.width))))
        rel_y = max(0.05, min(0.95, tank_y / max(1.0, float(self.height))))
        self.items.append((text, rel_x, rel_y, font_px))
        self.rebuild()

    def clear(self) -> None:
        self.items.clear()
        self.sdf = None
        self.tex = None

    @staticmethod
    def render_text_mask(text: str, font_px: int) -> Tuple[pygame.Surface, np.ndarray]:
        """Render crisp styled text surface and return (alpha_surface, bool_mask)."""
        font = pygame.font.SysFont("Segoe UI", font_px, bold=True)
        surf = font.render(text, True, (226, 232, 240))
        alpha = pygame.surfarray.array_alpha(surf)
        return surf, (alpha > 110)

    def rebuild(self) -> None:
        """Rebuild combined alpha texture and 2D Signed Distance Field from placed items."""
        if not self.items:
            self.sdf = None
            self.tex = None
            return

        self.overlay_surf.fill((0, 0, 0, 0))
        mask = np.zeros((self.width, self.height), dtype=bool)

        for text, rel_x, rel_y, font_px in self.items:
            glyph_s, gmask = self.render_text_mask(text, font_px)
            gw, gh = gmask.shape
            px = int(round(rel_x * self.width - gw * 0.5))
            py = int(round(rel_y * self.height - gh * 0.5))
            px = max(4, min(self.width - gw - 4, px))
            py = max(4, min(self.height - gh - 4, py))

            # Draw subtle dark drop shadow + crisp metallic letter surface.
            shadow = pygame.font.SysFont("Segoe UI", font_px, bold=True).render(text, True, (8, 12, 20))
            self.overlay_surf.blit(shadow, (px + 2, py + 2))
            self.overlay_surf.blit(glyph_s, (px, py))

            x1 = min(self.width, px + gw)
            y1 = min(self.height, py + gh)
            if x1 > px and y1 > py:
                mask[px:x1, py:y1] |= gmask[: x1 - px, : y1 - py]

        if not np.any(mask):
            self.sdf = None
            self.tex = None
            return

        d_out = distance_transform_edt(~mask).astype(np.float32)
        d_in = distance_transform_edt(mask).astype(np.float32)
        self.sdf = np.where(mask, -d_in, d_out).astype(np.float32)

        self.tex = Texture.from_surface(self.gpu, self.overlay_surf)
        self.tex.blend_mode = 1


@dataclass
class PillButton:
    rect: pygame.Rect
    label: str
    on_click: Callable[[], None]
    dot_color: Optional[Tuple[int, int, int]] = None
    active: bool = False
    accent: Tuple[int, int, int] = (37, 99, 235)


@dataclass
class SliderWidget:
    rect: pygame.Rect
    label: str
    min_val: float
    max_val: float
    value: float
    fmt: str
    accent: Tuple[int, int, int] = (56, 189, 248)
    on_change: Optional[Callable[[float], None]] = None

    def set_from_x(self, mx: float) -> None:
        t = (mx - self.rect.x) / max(1.0, float(self.rect.width))
        t = max(0.0, min(1.0, t))
        self.value = self.min_val + t * (self.max_val - self.min_val)
        if self.on_change is not None:
            self.on_change(self.value)

    @property
    def frac(self) -> float:
        span = max(1e-6, self.max_val - self.min_val)
        return max(0.0, min(1.0, (self.value - self.min_val) / span))


class FluidSimulatorApp:
    """Frameless dark-mode SDL2 GPU + Numba multi-density fluid simulator."""

    TOP_BAR_H = 38
    BOT_BAR_H = 84
    TEXT_BAR_H = 38
    RESIZE_MARGIN = 7
    TEXT_SIZES = (("S", 52), ("M", 84), ("L", 122))

    def __init__(self) -> None:
        os.environ["SDL_RENDER_SCALE_QUALITY"] = "1"
        pygame.init()
        self.win_w = 1180
        self.win_h = 740
        self.window = pygame.Window(
            "Fluid Lab",
            size=(self.win_w, self.win_h),
            borderless=True,
            resizable=True,
        )
        self.window.minimum_size = (1020, 580)
        self.window.position = (110, 70)

        self.gpu = Renderer(self.window)
        self.font_title = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.font_ui = pygame.font.SysFont("Segoe UI", 12, bold=True)
        self.font_sub = pygame.font.SysFont("Segoe UI", 11)
        self.font_mono = pygame.font.SysFont("Consolas", 12, bold=True)

        self.particles = ParticleSystem(initial_capacity=4096, max_particles=10000)
        self.solver = FluidPhysicsSolver(smoothing_radius=12.2, spacing=6.2)
        self.renderer = FluidRenderer(self.gpu, self.win_w, self._tank_height())
        self.obstacles = SolidTextObstacles(self.win_w, self._tank_height(), self.gpu)

        # Active tool & physical settings.
        self.active_preset_name = "Water"
        self.active_tool = "Spawn"
        self.selected_color_rgb = FLUID_PRESETS[2].color_rgb
        self.selected_cohesion = FLUID_PRESETS[2].cohesion
        self.paused = False
        self.is_maximized = False
        self._restore_geom = (110, 70, 1180, 740)
        self.zero_g_saved = 1.0

        # Jet / Stream emitter settings.
        self.jet_angle = math.radians(-22.0)  # Default upward-right parabolic arc.
        self.jet_pinned = False
        self.jet_pin_local = (140.0, 220.0)
        self.jet_firing = False

        # Text tool settings.
        self.text_input = "FLUID"
        self.text_solid_mode = True
        self.text_size_idx = 1  # M (84px)

        # Chrome surfaces + streaming textures.
        self._top_surf = pygame.Surface((self.win_w, self.TOP_BAR_H))
        self._top_tex = Texture(self.gpu, (self.win_w, self.TOP_BAR_H), streaming=True)
        self._bot_surf = pygame.Surface((self.win_w, self.BOT_BAR_H))
        self._bot_tex = Texture(self.gpu, (self.win_w, self.BOT_BAR_H), streaming=True)
        self._tbar_surf = pygame.Surface((self.win_w, self.TEXT_BAR_H), pygame.SRCALPHA)
        self._tbar_tex = Texture(self.gpu, (self.win_w, self.TEXT_BAR_H), streaming=True)
        self._tbar_tex.blend_mode = 1

        # Widgets.
        self.sliders: List[SliderWidget] = []
        self.preset_buttons: List[PillButton] = []
        self.tool_buttons: List[PillButton] = []
        self.action_buttons: List[PillButton] = []
        self.win_buttons: List[PillButton] = []
        self.text_bar_buttons: List[PillButton] = []
        self._init_widgets()
        self._layout_widgets()

        # Interaction state.
        self._active_slider: Optional[SliderWidget] = None
        self._dragging_window = False
        self._drag_start_global = (0, 0)
        self._drag_start_win = (0, 0)

        self._resizing_edge = ""
        self._resize_start_global = (0, 0)
        self._resize_start_size = (self.win_w, self.win_h)

        self._mouse_btn = 0
        self._mouse_pos = (0, 0)
        self._prev_mouse_tank: Optional[Tuple[float, float]] = None

        self._shake_remaining = 0.0
        self._shake_prev_ox = 0
        self._shake_prev_oy = 0

        bounds = self._get_tank_bounds()
        self._prev_bounds: Bounds = bounds
        self.particles.spawn_demo_layers(bounds)
        self._warmup_jit(bounds)

        self._running = True
        self._last_time = time.perf_counter()
        self._fps_ema = 120.0

    def _tank_height(self) -> int:
        return max(120, self.win_h - self.TOP_BAR_H - self.BOT_BAR_H)

    def _get_tank_bounds(self) -> Bounds:
        wx, wy = self.window.position
        return (
            float(wx + 1),
            float(wy + self.TOP_BAR_H),
            float(wx + self.win_w - 1),
            float(wy + self.win_h - self.BOT_BAR_H),
        )

    def _warmup_jit(self, bounds: Bounds) -> None:
        self._top_surf.fill((12, 16, 24))
        msg = self.font_title.render("Initializing Numba Fluid & GPU Pipeline...", True, (148, 163, 184))
        self.gpu.draw_color = (10, 14, 22, 255)
        self.gpu.clear()
        t_tex = Texture.from_surface(self.gpu, msg)
        t_tex.draw(dstrect=(self.win_w // 2 - msg.get_width() // 2, self.win_h // 2 - 10, msg.get_width(), msg.get_height()))
        self.gpu.present()

        self.solver.step(self.particles, bounds, bounds, dt=1.0 / 120.0, gravity=720.0)
        self.renderer.draw_fluid(self.TOP_BAR_H, self.particles, bounds[0], bounds[1])
        # Warm up SDF branch + alternative pixel/dot/velocity shaders.
        dummy_sdf = np.ones((8, 8), dtype=np.float32) * 10.0
        self.solver.step(self.particles, bounds, bounds, dt=1.0 / 120.0, gravity=720.0, sdf=dummy_sdf)
        tmp_buf = np.zeros((32, 32, 3), dtype=np.uint8)
        tmp_pos = np.array([[10.0, 10.0]], dtype=np.float64)
        tmp_col = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        tmp_vel = np.array([[100.0, 0.0]], dtype=np.float64)
        _stamp_dots(tmp_buf, tmp_pos, tmp_col, 1, 0.0, 0.0, 32, 32, 4)
        _stamp_pixels(tmp_buf, tmp_pos, tmp_col, 1, 0.0, 0.0, 32, 32, 3)
        _compute_speed_colors(tmp_vel, tmp_col, 1)

    def _init_widgets(self) -> None:
        for preset in FLUID_PRESETS:
            self.preset_buttons.append(
                PillButton(
                    rect=pygame.Rect(0, 0, 10, 28),
                    label=f"{preset.name} {preset.density:.2f}",
                    dot_color=preset.rgb_u8,
                    active=(preset.name == self.active_preset_name),
                    accent=preset.rgb_u8,
                    on_click=lambda p=preset: self._select_preset(p),
                )
            )

        for tname in ("Spawn", "Jet", "Push", "Vortex", "Erase", "Text"):
            self.tool_buttons.append(
                PillButton(
                    rect=pygame.Rect(0, 0, 10, 28),
                    label=tname,
                    active=(tname == self.active_tool),
                    accent=(14, 165, 233) if tname == "Jet" else (37, 99, 235),
                    on_click=lambda t=tname: self._select_tool(t),
                )
            )

        self.slider_density = SliderWidget(
            pygame.Rect(0, 0, 90, 18),
            "ρ",
            0.20,
            6.00,
            1.00,
            "{:.2f}",
            accent=FLUID_PRESETS[2].rgb_u8,
            on_change=self._on_density_change,
        )
        self.slider_visc = SliderWidget(
            pygame.Rect(0, 0, 76, 18),
            "Visc",
            0.00,
            0.25,
            0.018,
            "{:.2f}",
            accent=(148, 163, 184),
        )
        self.slider_brush = SliderWidget(
            pygame.Rect(0, 0, 76, 18),
            "Size",
            14.0,
            90.0,
            34.0,
            "{:.0f}",
            accent=(56, 189, 248),
        )
        self.slider_force = SliderWidget(
            pygame.Rect(0, 0, 82, 18),
            "Force",
            250.0,
            2400.0,
            1250.0,
            "{:.0f}",
            accent=(14, 165, 233),
        )
        self.slider_grav = SliderWidget(
            pygame.Rect(0, 0, 76, 18),
            "Grav",
            -1.5,
            2.0,
            1.0,
            "{:+.1f}g",
            accent=(52, 211, 153),
        )
        self.sliders = [
            self.slider_density,
            self.slider_visc,
            self.slider_brush,
            self.slider_force,
            self.slider_grav,
        ]

        self.btn_zero_g = PillButton(
            pygame.Rect(0, 0, 32, 26), "0g", self._toggle_zero_g, accent=(52, 211, 153)
        )
        self.btn_style = PillButton(
            pygame.Rect(0, 0, 90, 26), "Style: Fluid", self._cycle_style, accent=(56, 189, 248)
        )
        self.btn_shake = PillButton(
            pygame.Rect(0, 0, 58, 26), "Shake", self._trigger_shake, accent=(14, 165, 233)
        )
        self.btn_demo = PillButton(
            pygame.Rect(0, 0, 54, 26), "Demo", self._reset_demo, accent=(37, 99, 235)
        )
        self.btn_pause = PillButton(
            pygame.Rect(0, 0, 58, 26), "Pause", self._toggle_pause, accent=(234, 179, 8)
        )
        self.btn_clear = PillButton(
            pygame.Rect(0, 0, 52, 26), "Clear", self._clear_all, accent=(220, 38, 38)
        )
        self.action_buttons = [
            self.btn_zero_g,
            self.btn_style,
            self.btn_shake,
            self.btn_demo,
            self.btn_pause,
            self.btn_clear,
        ]

        self.btn_min = PillButton(
            pygame.Rect(0, 0, 34, 26), "—", self._minimize_window, accent=(51, 65, 85)
        )
        self.btn_max = PillButton(
            pygame.Rect(0, 0, 34, 26), "□", self._toggle_maximize, accent=(51, 65, 85)
        )
        self.btn_close = PillButton(
            pygame.Rect(0, 0, 36, 26), "✕", self._quit, accent=(220, 38, 38)
        )
        self.win_buttons = [self.btn_min, self.btn_max, self.btn_close]

        # Inline Text Tool bar buttons (shown when Text tool is active).
        self.btn_txt_mode = PillButton(
            pygame.Rect(0, 0, 108, 26), "Mode: Solid", self._toggle_text_mode, accent=(37, 99, 235), active=True
        )
        self.btn_txt_size = PillButton(
            pygame.Rect(0, 0, 74, 26), "Size: M", self._cycle_text_size, accent=(56, 189, 248)
        )
        self.btn_txt_stamp = PillButton(
            pygame.Rect(0, 0, 88, 26), "Drop Center", self._stamp_text_center, accent=(16, 185, 129)
        )
        self.btn_txt_clear = PillButton(
            pygame.Rect(0, 0, 84, 26), "Clear Text", self._clear_text, accent=(220, 38, 38)
        )
        self.text_bar_buttons = [
            self.btn_txt_mode,
            self.btn_txt_size,
            self.btn_txt_stamp,
            self.btn_txt_clear,
        ]

    def _layout_widgets(self) -> None:
        w, h = self.win_w, self.win_h
        rx = w - 10
        for btn in reversed(self.win_buttons):
            btn.rect = pygame.Rect(rx - btn.rect.width, 6, btn.rect.width, 26)
            rx -= btn.rect.width + 4

        dock_y = h - self.BOT_BAR_H
        r1_y = dock_y + 8
        lx = 12
        for btn in self.preset_buttons:
            bw = max(76, self.font_ui.size(btn.label)[0] + 22)
            btn.rect = pygame.Rect(lx, r1_y, bw, 28)
            lx += bw + 5

        rx = w - 12
        for btn in reversed(self.tool_buttons):
            bw = max(48, self.font_ui.size(btn.label)[0] + 16)
            btn.rect = pygame.Rect(rx - bw, r1_y, bw, 28)
            rx -= bw + 4

        r2_y = dock_y + 44
        rx = w - 12
        for btn in reversed(self.action_buttons[1:]):
            bw = max(50, self.font_ui.size(btn.label)[0] + 16)
            btn.rect = pygame.Rect(rx - bw, r2_y, bw, 28)
            rx -= bw + 5

        avail = max(420, rx - 14)
        s_specs = [
            (self.slider_density, 0.21),
            (self.slider_visc, 0.19),
            (self.slider_brush, 0.19),
            (self.slider_force, 0.22),
            (self.slider_grav, 0.19),
        ]
        sx = 12
        label_pad = 72
        gap = 8
        track_budget = max(200, avail - 40 - len(s_specs) * (label_pad + gap))
        for slider, weight in s_specs:
            tw = max(42, min(115, int(track_budget * weight)))
            sx += 34
            slider.rect = pygame.Rect(sx, r2_y + 5, tw, 18)
            sx += tw + 38 + gap

        self.btn_zero_g.rect = pygame.Rect(min(sx - 2, rx - 38), r2_y + 1, 32, 26)

        # Inline Text bar inside top of tank.
        tb_y = self.TOP_BAR_H + 6
        tx = w - 14
        for btn in reversed(self.text_bar_buttons):
            bw = max(72, self.font_ui.size(btn.label)[0] + 18)
            btn.rect = pygame.Rect(tx - bw, tb_y, bw, 26)
            tx -= bw + 6

    def _select_preset(self, preset: FluidPreset) -> None:
        self.active_preset_name = preset.name
        self.selected_color_rgb = preset.color_rgb
        self.selected_cohesion = preset.cohesion
        self.slider_density.value = preset.density
        self.slider_density.accent = preset.rgb_u8
        self.slider_visc.value = preset.viscosity
        for btn in self.preset_buttons:
            btn.active = btn.label.startswith(preset.name + " ")
        if self.active_tool in ("Push", "Vortex", "Erase", "Text"):
            self._select_tool("Spawn")

    def _on_density_change(self, rho: float) -> None:
        self.active_preset_name = ""
        rgb = color_for_density(rho)
        self.selected_color_rgb = rgb
        self.selected_cohesion = 0.05 + 0.04 * min(1.0, rho / 3.0)
        self.slider_density.accent = rgb_to_u8(rgb)
        for btn in self.preset_buttons:
            btn.active = False

    def _select_tool(self, tool_name: str) -> None:
        self.active_tool = tool_name
        for btn in self.tool_buttons:
            btn.active = btn.label == tool_name
        self.jet_firing = False

    def _toggle_text_mode(self) -> None:
        self.text_solid_mode = not self.text_solid_mode
        self.btn_txt_mode.label = "Mode: Solid" if self.text_solid_mode else "Mode: Fluid"
        self.btn_txt_mode.active = self.text_solid_mode
        self._layout_widgets()

    def _cycle_text_size(self) -> None:
        self.text_size_idx = (self.text_size_idx + 1) % len(self.TEXT_SIZES)
        tag, _ = self.TEXT_SIZES[self.text_size_idx]
        self.btn_txt_size.label = f"Size: {tag}"
        self._layout_widgets()

    def _stamp_text_at(self, tank_x: float, tank_y: float) -> None:
        txt = self.text_input.strip()
        if not txt:
            return
        _, font_px = self.TEXT_SIZES[self.text_size_idx]
        if self.text_solid_mode:
            self.obstacles.add_text(txt, tank_x, tank_y, font_px)
        else:
            _, mask = SolidTextObstacles.render_text_mask(txt, font_px)
            mw, mh = mask.shape
            bounds = self._get_tank_bounds()
            lx0 = max(6.0, min(self.win_w - mw - 6.0, tank_x - mw * 0.5))
            ly0 = max(6.0, min(self._tank_height() - mh - 6.0, tank_y - mh * 0.5))
            self.particles.spawn_from_mask(
                mask,
                lx0,
                ly0,
                density=self.slider_density.value,
                viscosity=self.slider_visc.value,
                cohesion=self.selected_cohesion,
                color_rgb=self.selected_color_rgb,
                bounds=bounds,
            )

    def _stamp_text_center(self) -> None:
        self._stamp_text_at(self.win_w * 0.5, self._tank_height() * 0.35)

    def _clear_text(self) -> None:
        self.obstacles.clear()

    def _cycle_style(self) -> None:
        style = self.renderer.cycle_style()
        self.btn_style.label = f"Style: {style}"
        self._layout_widgets()

    def _toggle_zero_g(self) -> None:
        if abs(self.slider_grav.value) > 0.05:
            self.zero_g_saved = self.slider_grav.value
            self.slider_grav.value = 0.0
            self.btn_zero_g.active = True
        else:
            self.slider_grav.value = self.zero_g_saved if abs(self.zero_g_saved) > 0.05 else 1.0
            self.btn_zero_g.active = False

    def _trigger_shake(self) -> None:
        self._shake_remaining = 0.42
        self._shake_prev_ox = 0
        self._shake_prev_oy = 0

    def _reset_demo(self) -> None:
        bounds = self._get_tank_bounds()
        self.particles.spawn_demo_layers(bounds)
        self._prev_bounds = bounds
        self.solver.reset_kinematics()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.btn_pause.label = "Resume" if self.paused else "Pause"
        self.btn_pause.active = self.paused

    def _clear_all(self) -> None:
        self.particles.clear()
        self.obstacles.clear()

    def _minimize_window(self) -> None:
        self.window.minimize()

    def _toggle_maximize(self) -> None:
        if not self.is_maximized:
            wx, wy = self.window.position
            self._restore_geom = (wx, wy, self.win_w, self.win_h)
            self.window.maximize()
            self.is_maximized = True
        else:
            self.window.restore()
            rx, ry, rw, rh = self._restore_geom
            self.window.position = (rx, ry)
            self.window.size = (rw, rh)
            self.is_maximized = False
        self._sync_window_size()

    def _quit(self) -> None:
        self._running = False

    def _sync_window_size(self) -> None:
        sw, sh = self.window.size
        if sw != self.win_w or sh != self.win_h:
            self.win_w = max(1020, int(sw))
            self.win_h = max(580, int(sh))
            th = self._tank_height()
            self.renderer.resize(self.win_w, th)
            self.obstacles.resize(self.win_w, th)
            self._top_surf = pygame.Surface((self.win_w, self.TOP_BAR_H))
            self._top_tex = Texture(self.gpu, (self.win_w, self.TOP_BAR_H), streaming=True)
            self._bot_surf = pygame.Surface((self.win_w, self.BOT_BAR_H))
            self._bot_tex = Texture(self.gpu, (self.win_w, self.BOT_BAR_H), streaming=True)
            self._tbar_surf = pygame.Surface((self.win_w, self.TEXT_BAR_H), pygame.SRCALPHA)
            self._tbar_tex = Texture(self.gpu, (self.win_w, self.TEXT_BAR_H), streaming=True)
            self._tbar_tex.blend_mode = 1
            self._layout_widgets()

    def _detect_resize_edge(self, mx: int, my: int) -> str:
        if self.is_maximized:
            return ""
        m = self.RESIZE_MARGIN
        edge = ""
        if mx >= self.win_w - m:
            edge += "r"
        if my >= self.win_h - m:
            edge += "b"
        return edge

    def _update_cursor(self, mx: int, my: int) -> None:
        edge = self._resizing_edge or self._detect_resize_edge(mx, my)
        if edge == "rb":
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENWSE)
        elif edge == "r":
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
        elif edge == "b":
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
        elif my < self.TOP_BAR_H and not any(b.rect.collidepoint(mx, my) for b in self.win_buttons):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    def _handle_events(self) -> None:
        mods = pygame.key.get_mods()
        alt_held = bool(mods & pygame.KMOD_ALT)
        typing_text = self.active_tool == "Text"

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._running = False
            elif ev.type == pygame.WINDOWRESIZED:
                self._sync_window_size()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if typing_text:
                        self._select_tool("Spawn")
                    else:
                        self._running = False
                elif typing_text:
                    if ev.key == pygame.K_BACKSPACE:
                        self.text_input = self.text_input[:-1]
                    elif ev.key == pygame.K_RETURN:
                        self._stamp_text_center()
                    elif ev.unicode and len(self.text_input) < 28 and ev.unicode.isprintable():
                        self.text_input += ev.unicode
                else:
                    if ev.key == pygame.K_SPACE:
                        self._toggle_pause()
                    elif ev.key == pygame.K_c:
                        self._clear_all()
                    elif ev.key == pygame.K_r:
                        self._reset_demo()
                    elif ev.key == pygame.K_g:
                        self._toggle_zero_g()
                    elif ev.key == pygame.K_v:
                        self._cycle_style()
                    elif ev.key == pygame.K_s:
                        self._trigger_shake()
                    elif ev.key == pygame.K_t:
                        self._select_tool("Text")
                    elif ev.key == pygame.K_j:
                        self._select_tool("Jet")
                    elif ev.key == pygame.K_p and self.active_tool == "Jet":
                        self.jet_pinned = not self.jet_pinned
                        if self.jet_pinned:
                            mx, my = self._mouse_pos
                            if self.TOP_BAR_H <= my < (self.win_h - self.BOT_BAR_H):
                                self.jet_pin_local = (float(mx), float(my - self.TOP_BAR_H))
                    elif ev.key == pygame.K_q and self.active_tool == "Jet":
                        self.jet_angle -= math.radians(12.0)
                    elif ev.key == pygame.K_e and self.active_tool == "Jet":
                        self.jet_angle += math.radians(12.0)
            elif ev.type == pygame.MOUSEWHEEL:
                if self.active_tool == "Jet" and not (mods & pygame.KMOD_SHIFT):
                    self.jet_angle += math.radians(ev.y * 8.0)
                else:
                    new_b = max(14.0, min(90.0, self.slider_brush.value + ev.y * 4.0))
                    self.slider_brush.value = new_b
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                self._mouse_pos = (mx, my)
                if ev.button == 1:
                    edge = self._detect_resize_edge(mx, my)
                    if edge:
                        self._resizing_edge = edge
                        self._resize_start_global = get_global_cursor_pos()
                        self._resize_start_size = (self.win_w, self.win_h)
                        continue

                    clicked = False
                    check_btns = self.win_buttons + self.preset_buttons + self.tool_buttons + self.action_buttons
                    if self.active_tool == "Text":
                        check_btns = check_btns + self.text_bar_buttons
                    for btn in check_btns:
                        if btn.rect.collidepoint(mx, my):
                            btn.on_click()
                            clicked = True
                            break
                    if clicked:
                        continue

                    if self.active_tool == "Text" and self.TOP_BAR_H <= my < self.TOP_BAR_H + self.TEXT_BAR_H + 8:
                        continue

                    for s in self.sliders:
                        hit = s.rect.inflate(10, 10)
                        if hit.collidepoint(mx, my):
                            self._active_slider = s
                            s.set_from_x(mx)
                            if s is self.slider_grav:
                                self.btn_zero_g.active = abs(s.value) <= 0.05
                            clicked = True
                            break
                    if clicked:
                        continue

                    if my < self.TOP_BAR_H or alt_held:
                        if self.is_maximized:
                            self._toggle_maximize()
                        self._dragging_window = True
                        self._drag_start_global = get_global_cursor_pos()
                        self._drag_start_win = self.window.position
                        continue

                    if self.TOP_BAR_H <= my < (self.win_h - self.BOT_BAR_H):
                        if self.active_tool == "Text":
                            self._stamp_text_at(float(mx), float(my - self.TOP_BAR_H))
                        else:
                            self._mouse_btn = 1
                            self._prev_mouse_tank = (float(mx), float(my - self.TOP_BAR_H))
                elif ev.button in (2, 3):
                    if self.TOP_BAR_H <= my < (self.win_h - self.BOT_BAR_H):
                        self._mouse_btn = ev.button
                        self._prev_mouse_tank = (float(mx), float(my - self.TOP_BAR_H))
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    self._dragging_window = False
                    self._resizing_edge = ""
                    self._active_slider = None
                    if self._mouse_btn == 1:
                        self._mouse_btn = 0
                elif ev.button == self._mouse_btn:
                    self._mouse_btn = 0
            elif ev.type == pygame.MOUSEMOTION:
                self._mouse_pos = ev.pos
                if self._active_slider is not None:
                    self._active_slider.set_from_x(ev.pos[0])
                    if self._active_slider is self.slider_grav:
                        self.btn_zero_g.active = abs(self._active_slider.value) <= 0.05

        if self._dragging_window:
            gx, gy = get_global_cursor_pos()
            dx = gx - self._drag_start_global[0]
            dy = gy - self._drag_start_global[1]
            self.window.position = (self._drag_start_win[0] + dx, self._drag_start_win[1] + dy)
        elif self._resizing_edge:
            gx, gy = get_global_cursor_pos()
            dx = gx - self._resize_start_global[0]
            dy = gy - self._resize_start_global[1]
            nw = max(1020, self._resize_start_size[0] + (dx if "r" in self._resizing_edge else 0))
            nh = max(580, self._resize_start_size[1] + (dy if "b" in self._resizing_edge else 0))
            if nw != self.win_w or nh != self.win_h:
                self.window.size = (nw, nh)
                self._sync_window_size()

        self._update_cursor(*self._mouse_pos)

    def _apply_programmatic_shake(self, dt: float) -> None:
        if self._shake_remaining <= 0.0:
            return
        if self._dragging_window or self._resizing_edge:
            self._shake_remaining = 0.0
            return
        self._shake_remaining = max(0.0, self._shake_remaining - dt)
        elapsed = 0.42 - self._shake_remaining
        decay = self._shake_remaining / 0.42
        ox = int(round(40.0 * math.sin(elapsed * 32.0) * decay))
        oy = int(round(14.0 * math.cos(elapsed * 44.0) * decay))
        dx = ox - self._shake_prev_ox
        dy = oy - self._shake_prev_oy
        self._shake_prev_ox = ox
        self._shake_prev_oy = oy
        if dx != 0 or dy != 0:
            wx, wy = self.window.position
            self.window.position = (wx + dx, wy + dy)

    def _process_tank_pointer(self, bounds: Bounds, dt: float) -> Optional[Tuple[float, float]]:
        mx, my = self._mouse_pos
        in_tank = (
            not self._dragging_window
            and not self._resizing_edge
            and self._active_slider is None
            and self.TOP_BAR_H <= my < (self.win_h - self.BOT_BAR_H)
            and 0 <= mx < self.win_w
        )
        if not in_tank:
            self._prev_mouse_tank = None
            return None
        if self.active_tool == "Text" and my < self.TOP_BAR_H + self.TEXT_BAR_H + 8:
            self._prev_mouse_tank = None
            return None

        tx = float(mx)
        ty = float(my - self.TOP_BAR_H)
        wx = bounds[0] + tx
        wy = bounds[1] + ty

        if self._prev_mouse_tank is not None and dt > 0.0:
            vx_m = (tx - self._prev_mouse_tank[0]) / dt
            vy_m = (ty - self._prev_mouse_tank[1]) / dt
        else:
            vx_m = 0.0
            vy_m = 0.0
        self._prev_mouse_tank = (tx, ty)

        shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        if self.active_tool == "Jet" and self.jet_pinned:
            # Aim pinned emitter 360 deg toward mouse pointer.
            adx = tx - self.jet_pin_local[0]
            ady = ty - self.jet_pin_local[1]
            if adx * adx + ady * ady > 25.0:
                self.jet_angle = math.atan2(ady, adx)
        elif self.active_tool == "Jet" and self._mouse_btn == 1 and shift_held:
            # Shift+Drag steers jet angle dynamically.
            if abs(vx_m) + abs(vy_m) > 40.0:
                self.jet_angle = math.atan2(vy_m, vx_m)

        if self._mouse_btn != 0:
            radius = self.slider_brush.value
            if self._mouse_btn == 3:
                self.solver.apply_brush_force(
                    self.particles, wx, wy, vx_m, vy_m, radius * 1.4, "push", dt
                )
            elif self._mouse_btn == 2 or (self._mouse_btn == 1 and shift_held and self.active_tool != "Jet"):
                self.solver.apply_brush_force(
                    self.particles, wx, wy, vx_m, vy_m, radius * 1.5, "vortex", dt
                )
            elif self._mouse_btn == 1:
                tool = self.active_tool
                if tool == "Spawn":
                    self.particles.spawn_disk(
                        wx,
                        wy,
                        radius=radius * 0.80,
                        density=self.slider_density.value,
                        viscosity=self.slider_visc.value,
                        cohesion=self.selected_cohesion,
                        color_rgb=self.selected_color_rgb,
                        bounds=bounds,
                        init_vel=(
                            max(-420.0, min(420.0, vx_m * 0.35)),
                            max(-420.0, min(420.0, vy_m * 0.35)),
                        ),
                    )
                elif tool == "Jet":
                    ex = bounds[0] + (self.jet_pin_local[0] if self.jet_pinned else tx)
                    ey = bounds[1] + (self.jet_pin_local[1] if self.jet_pinned else ty)
                    self.particles.spawn_jet(
                        ex,
                        ey,
                        angle_rad=self.jet_angle,
                        speed=self.slider_force.value,
                        stream_radius=radius,
                        density=self.slider_density.value,
                        viscosity=self.slider_visc.value,
                        cohesion=self.selected_cohesion,
                        color_rgb=self.selected_color_rgb,
                        bounds=bounds,
                        dt=dt,
                    )
                elif tool == "Push":
                    self.solver.apply_brush_force(
                        self.particles, wx, wy, vx_m, vy_m, radius * 1.4, "push", dt
                    )
                elif tool == "Vortex":
                    self.solver.apply_brush_force(
                        self.particles, wx, wy, vx_m, vy_m, radius * 1.5, "vortex", dt
                    )
                elif tool == "Erase":
                    self.particles.remove_in_radius(wx, wy, radius)

        return (tx, ty)

    def _draw_pill(self, surf: pygame.Surface, btn: PillButton, hovered: bool, y_off: int = 0) -> None:
        r = pygame.Rect(btn.rect.x, btn.rect.y - y_off, btn.rect.width, btn.rect.height)
        if btn.active:
            bg = (
                min(255, int(btn.accent[0] * 0.30 + 20)),
                min(255, int(btn.accent[1] * 0.30 + 26)),
                min(255, int(btn.accent[2] * 0.30 + 38)),
            )
            border = btn.accent
            fg = (248, 250, 252)
        elif hovered:
            bg = (30, 41, 59)
            border = (71, 85, 105)
            fg = (241, 245, 249)
        else:
            bg = (19, 25, 38)
            border = (34, 44, 64)
            fg = (190, 202, 220)

        pygame.draw.rect(surf, bg, r, border_radius=7)
        pygame.draw.rect(surf, border, r, width=1, border_radius=7)

        txt = self.font_ui.render(btn.label, True, fg)
        if btn.dot_color is not None:
            dx = r.x + 11
            dy = r.centery
            pygame.draw.aacircle(surf, btn.dot_color, (dx, dy), 5)
            surf.blit(txt, (dx + 9, r.centery - txt.get_height() // 2))
        else:
            surf.blit(
                txt,
                (r.centerx - txt.get_width() // 2, r.centery - txt.get_height() // 2),
            )

    def _draw_slider(self, surf: pygame.Surface, s: SliderWidget, hovered: bool, y_off: int) -> None:
        sr = pygame.Rect(s.rect.x, s.rect.y - y_off, s.rect.width, s.rect.height)
        lbl = self.font_sub.render(s.label, True, (148, 163, 184))
        surf.blit(lbl, (sr.x - lbl.get_width() - 6, sr.centery - lbl.get_height() // 2))

        track_rect = pygame.Rect(sr.x, sr.centery - 3, sr.width, 6)
        pygame.draw.rect(surf, (25, 33, 50), track_rect, border_radius=3)

        fill_w = max(4, int(round(s.frac * sr.width)))
        fill_rect = pygame.Rect(sr.x, sr.centery - 3, fill_w, 6)
        pygame.draw.rect(surf, s.accent, fill_rect, border_radius=3)

        tx = sr.x + int(round(s.frac * sr.width))
        ty = sr.centery
        r_thumb = 7 if (hovered or self._active_slider is s) else 6
        pygame.draw.aacircle(surf, (248, 250, 252), (tx, ty), r_thumb)
        pygame.draw.aacircle(surf, s.accent, (tx, ty), r_thumb - 2)

        val_txt = self.font_mono.render(s.fmt.format(s.value), True, (226, 232, 240))
        surf.blit(val_txt, (sr.right + 6, sr.centery - val_txt.get_height() // 2))

    def _draw_chrome_gpu(self, mouse_tank: Optional[Tuple[float, float]]) -> None:
        w, h = self.win_w, self.win_h
        mx, my = self._mouse_pos

        # --- Tank overlays (solid text obstacles + brush/jet/text cursor previews) ---
        if self.obstacles.tex is not None:
            self.obstacles.tex.draw(dstrect=(0, self.TOP_BAR_H, w, self._tank_height()))

        if mouse_tank is not None:
            sx = int(round(mouse_tank[0]))
            sy = int(round(mouse_tank[1] + self.TOP_BAR_H))
            col_u8 = rgb_to_u8(self.selected_color_rgb)
            if self.active_tool == "Jet":
                ex = int(round(self.jet_pin_local[0])) if self.jet_pinned else sx
                ey = int(round((self.jet_pin_local[1] if self.jet_pinned else mouse_tank[1]) + self.TOP_BAR_H))
                ux = math.cos(self.jet_angle)
                uy = math.sin(self.jet_angle)
                # Parabolic trajectory preview dots
                self.gpu.draw_color = (col_u8[0], col_u8[1], col_u8[2], 210)
                v0 = self.slider_force.value
                g_px = self.slider_grav.value * 720.0
                prev_pt = (ex, ey)
                for step_k in range(1, 7):
                    t_p = step_k * 0.045
                    px_p = int(round(ex + v0 * ux * t_p))
                    py_p = int(round(ey + v0 * uy * t_p + 0.5 * g_px * t_p * t_p))
                    self.gpu.draw_line(prev_pt, (px_p, py_p))
                    prev_pt = (px_p, py_p)
                self.gpu.draw_color = (248, 250, 252, 240)
                self.gpu.draw_rect(pygame.Rect(ex - 4, ey - 4, 8, 8))
            elif self.active_tool == "Text":
                txt = self.text_input.strip()
                if txt:
                    _, font_px = self.TEXT_SIZES[self.text_size_idx]
                    gs, _ = SolidTextObstacles.render_text_mask(txt, font_px)
                    gw, gh = gs.get_size()
                    gx = max(4, min(w - gw - 4, sx - gw // 2))
                    gy = max(self.TOP_BAR_H + 4, min(h - self.BOT_BAR_H - gh - 4, sy - gh // 2))
                    self.gpu.draw_color = (56, 189, 248, 170) if self.text_solid_mode else (*col_u8, 170)
                    self.gpu.draw_rect(pygame.Rect(gx - 4, gy - 4, gw + 8, gh + 8))
            else:
                rad = max(6.0, self.slider_brush.value)
                self.gpu.draw_color = (col_u8[0], col_u8[1], col_u8[2], 190)
                steps = 28
                pts = [
                    (
                        int(round(sx + rad * math.cos(2.0 * math.pi * k / steps))),
                        int(round(sy + rad * math.sin(2.0 * math.pi * k / steps))),
                    )
                    for k in range(steps + 1)
                ]
                for k in range(steps):
                    self.gpu.draw_line(pts[k], pts[k + 1])

        # --- Inline Text Input Bar (only when Text tool active) ---
        if self.active_tool == "Text":
            self._tbar_surf.fill((0, 0, 0, 0))
            bar_r = pygame.Rect(10, 0, w - 20, self.TEXT_BAR_H - 4)
            pygame.draw.rect(self._tbar_surf, (15, 21, 34, 232), bar_r, border_radius=8)
            pygame.draw.rect(self._tbar_surf, (56, 189, 248, 180), bar_r, width=1, border_radius=8)

            cursor_ch = "|" if int(time.perf_counter() * 2.2) % 2 == 0 else ""
            prompt = f"Type text:  {self.text_input}{cursor_ch}"
            pt_s = self.font_ui.render(prompt, True, (241, 245, 249))
            self._tbar_surf.blit(pt_s, (24, 9))

            hint_t = self.font_sub.render("Click tank to stamp · Enter centers", True, (148, 163, 184))
            if self.text_bar_buttons[0].rect.x - (24 + pt_s.get_width()) > 230:
                self._tbar_surf.blit(hint_t, (self.text_bar_buttons[0].rect.x - hint_t.get_width() - 16, 10))

            tb_y = self.TOP_BAR_H + 6
            for btn in self.text_bar_buttons:
                self._draw_pill(self._tbar_surf, btn, btn.rect.collidepoint(mx, my), y_off=tb_y)

            self._tbar_tex.update(self._tbar_surf)
            self._tbar_tex.draw(dstrect=(0, tb_y, w, self.TEXT_BAR_H))

        # --- Top Titlebar ---
        top = self._top_surf
        top.fill((13, 18, 28))
        pygame.draw.line(top, (27, 36, 52), (0, self.TOP_BAR_H - 1), (w, self.TOP_BAR_H - 1))

        pygame.draw.aacircle(top, (56, 189, 248), (18, 19), 6)
        pygame.draw.aacircle(top, (186, 230, 253), (16, 17), 2)
        title_s = self.font_title.render("FLUID LAB", True, (241, 245, 249))
        top.blit(title_s, (31, 10))

        hint_txt = (
            "Jet: Scroll/Q/E aim · P pin · Shift+Drag aim"
            if self.active_tool == "Jet"
            else "Drag bar / Alt+Drag to move window · R-Drag stir · Scroll size"
        )
        hint_s = self.font_sub.render(hint_txt, True, (100, 116, 139))
        if w > 980:
            top.blit(hint_s, (118, 12))

        stats_text = (
            f"{self._fps_ema:4.0f} FPS  ·  "
            f"{self.particles.n}/{self.particles.max_particles} pts  ·  "
            f"Sim {self.solver.last_physics_ms:4.1f}ms  ·  "
            f"Win {self.solver.last_win_speed:4.0f}px/s"
        )
        stats_s = self.font_mono.render(stats_text, True, (56, 189, 248))
        badge_x = self.win_buttons[0].rect.x - stats_s.get_width() - 22
        if badge_x > 420:
            badge_rect = pygame.Rect(badge_x - 10, 7, stats_s.get_width() + 20, 24)
            pygame.draw.rect(top, (18, 25, 38), badge_rect, border_radius=6)
            pygame.draw.rect(top, (30, 41, 59), badge_rect, width=1, border_radius=6)
            top.blit(stats_s, (badge_x, 12))

        for btn in self.win_buttons:
            hov = btn.rect.collidepoint(mx, my)
            if btn is self.btn_close and hov:
                bg, fg = (185, 28, 28), (255, 255, 255)
            elif hov:
                bg, fg = (36, 47, 68), (248, 250, 252)
            else:
                bg, fg = (18, 24, 36), (148, 163, 184)
            pygame.draw.rect(top, bg, btn.rect, border_radius=6)
            lbl = self.font_ui.render(btn.label, True, fg)
            top.blit(lbl, (btn.rect.centerx - lbl.get_width() // 2, btn.rect.centery - lbl.get_height() // 2))

        self._top_tex.update(top)
        self._top_tex.draw(dstrect=(0, 0, w, self.TOP_BAR_H))

        # --- Bottom Dock ---
        bot = self._bot_surf
        dock_y = h - self.BOT_BAR_H
        bot.fill((13, 18, 28))
        pygame.draw.line(bot, (27, 36, 52), (0, 0), (w, 0))
        pygame.draw.line(bot, (21, 29, 43), (12, 40), (w - 12, 40))

        for btn in self.preset_buttons + self.tool_buttons + self.action_buttons:
            self._draw_pill(bot, btn, btn.rect.collidepoint(mx, my), y_off=dock_y)

        for s in self.sliders:
            self._draw_slider(bot, s, s.rect.inflate(10, 10).collidepoint(mx, my), y_off=dock_y)

        self._bot_tex.update(bot)
        self._bot_tex.draw(dstrect=(0, dock_y, w, self.BOT_BAR_H))

        # Crisp 1px window border + resize grip on GPU.
        self.gpu.draw_color = (30, 41, 59, 255)
        self.gpu.draw_rect(pygame.Rect(0, 0, w, h))
        self.gpu.draw_color = (71, 85, 105, 255)
        for k in range(3):
            off = 4 + k * 4
            self.gpu.draw_line((w - off, h - 2), (w - 2, h - off))

    def step_frame(self) -> None:
        now = time.perf_counter()
        raw_dt = now - self._last_time
        self._last_time = now
        dt = min(0.030, max(0.002, raw_dt))
        if raw_dt > 1e-4:
            self._fps_ema = 0.90 * self._fps_ema + 0.10 * min(300.0, 1.0 / raw_dt)

        self._handle_events()
        if IS_WINDOWS and _user32.IsIconic(self.window.handle):
            time.sleep(0.03)
            return
        self._apply_programmatic_shake(dt)

        curr_bounds = self._get_tank_bounds()
        mouse_tank = self._process_tank_pointer(curr_bounds, dt)

        if not self.paused:
            self.solver.step(
                self.particles,
                curr_bounds=curr_bounds,
                prev_bounds=self._prev_bounds,
                dt=dt,
                gravity=self.slider_grav.value * 720.0,
                sdf=self.obstacles.sdf,
                dragging=self._dragging_window or bool(self._resizing_edge),
            )
        self._prev_bounds = curr_bounds

        self.gpu.draw_color = (8, 11, 17, 255)
        self.gpu.clear()
        self.renderer.draw_fluid(self.TOP_BAR_H, self.particles, curr_bounds[0], curr_bounds[1])
        self._draw_chrome_gpu(mouse_tank)
        self.gpu.present()

    def run(self) -> None:
        clock = pygame.time.Clock()
        while self._running:
            self.step_frame()
            clock.tick(120)
        self.window.destroy()
        pygame.quit()


def main() -> None:
    app = FluidSimulatorApp()
    app.run()


if __name__ == "__main__":
    main()
