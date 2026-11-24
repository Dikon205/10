"""Particle SoA buffers, multi-density fluid presets, and spatial spawning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class FluidPreset:
    """Physical and visual preset for a spawnable fluid phase."""

    name: str
    density: float
    viscosity: float
    cohesion: float
    color_rgb: Tuple[float, float, float]
    rgb_u8: Tuple[int, int, int]


FLUID_PRESETS: Tuple[FluidPreset, ...] = (
    FluidPreset(
        name="Foam",
        density=0.45,
        viscosity=0.065,
        cohesion=0.02,
        color_rgb=(0.99, 0.88, 0.28),
        rgb_u8=(253, 224, 71),
    ),
    FluidPreset(
        name="Oil",
        density=0.72,
        viscosity=0.045,
        cohesion=0.06,
        color_rgb=(0.96, 0.62, 0.04),
        rgb_u8=(245, 158, 11),
    ),
    FluidPreset(
        name="Water",
        density=1.00,
        viscosity=0.018,
        cohesion=0.07,
        color_rgb=(0.22, 0.74, 0.97),
        rgb_u8=(56, 189, 248),
    ),
    FluidPreset(
        name="Honey",
        density=1.45,
        viscosity=0.180,
        cohesion=0.12,
        color_rgb=(0.92, 0.35, 0.05),
        rgb_u8=(234, 88, 12),
    ),
    FluidPreset(
        name="Brine",
        density=2.20,
        viscosity=0.030,
        cohesion=0.07,
        color_rgb=(0.66, 0.33, 0.97),
        rgb_u8=(168, 85, 247),
    ),
    FluidPreset(
        name="Mercury",
        density=4.00,
        viscosity=0.012,
        cohesion=0.22,
        color_rgb=(0.86, 0.90, 0.95),
        rgb_u8=(219, 230, 242),
    ),
)

_DENSITY_COLOR_STOPS = np.array(
    [
        [0.20, 0.99, 0.94, 0.54],
        [0.50, 0.98, 0.80, 0.14],
        [0.80, 0.20, 0.83, 0.60],
        [1.00, 0.22, 0.74, 0.97],
        [1.50, 0.95, 0.40, 0.12],
        [2.40, 0.68, 0.33, 0.97],
        [4.00, 0.86, 0.90, 0.95],
        [6.00, 0.96, 0.34, 0.44],
    ],
    dtype=np.float64,
)


def color_for_density(density: float) -> Tuple[float, float, float]:
    """Return an interpolated RGB tuple in [0, 1] for an arbitrary fluid density."""
    stops_d = _DENSITY_COLOR_STOPS[:, 0]
    r = float(np.interp(density, stops_d, _DENSITY_COLOR_STOPS[:, 1]))
    g = float(np.interp(density, stops_d, _DENSITY_COLOR_STOPS[:, 2]))
    b = float(np.interp(density, stops_d, _DENSITY_COLOR_STOPS[:, 3]))
    return (r, g, b)


def rgb_to_u8(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert normalized [0, 1] RGB tuple to 8-bit (R, G, B) integers."""
    return (
        max(0, min(255, int(round(rgb[0] * 255.0)))),
        max(0, min(255, int(round(rgb[1] * 255.0)))),
        max(0, min(255, int(round(rgb[2] * 255.0)))),
    )


class ParticleSystem:
    """Structure-of-Arrays (SoA) container for multi-density fluid particles."""

    DEFAULT_SPACING = 6.2

    def __init__(self, initial_capacity: int = 4096, max_particles: int = 10000) -> None:
        self.max_particles = max_particles
        self._cap = max(256, initial_capacity)
        self.n = 0

        self._pos = np.zeros((self._cap, 2), dtype=np.float64)
        self._vel = np.zeros((self._cap, 2), dtype=np.float64)
        self._rho0 = np.ones(self._cap, dtype=np.float64)
        self._inv_m = np.ones(self._cap, dtype=np.float64)
        self._visc = np.zeros(self._cap, dtype=np.float64)
        self._cohesion = np.zeros(self._cap, dtype=np.float64)
        self._color = np.zeros((self._cap, 3), dtype=np.float32)
        self._rng = np.random.default_rng(1337)

    @property
    def pos(self) -> np.ndarray:
        return self._pos[: self.n]

    @property
    def vel(self) -> np.ndarray:
        return self._vel[: self.n]

    @property
    def rho0(self) -> np.ndarray:
        return self._rho0[: self.n]

    @property
    def inv_m(self) -> np.ndarray:
        return self._inv_m[: self.n]

    @property
    def visc(self) -> np.ndarray:
        return self._visc[: self.n]

    @property
    def cohesion(self) -> np.ndarray:
        return self._cohesion[: self.n]

    @property
    def color(self) -> np.ndarray:
        return self._color[: self.n]

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._cap:
            return
        new_cap = self._cap
        while new_cap < needed:
            new_cap *= 2
        new_cap = min(new_cap, self.max_particles)
        if new_cap <= self._cap:
            return

        def _grow(arr: np.ndarray) -> np.ndarray:
            out = np.zeros((new_cap,) + arr.shape[1:], dtype=arr.dtype)
            out[: self.n] = arr[: self.n]
            return out

        self._pos = _grow(self._pos)
        self._vel = _grow(self._vel)
        self._rho0 = _grow(self._rho0)
        self._inv_m = _grow(self._inv_m)
        self._visc = _grow(self._visc)
        self._cohesion = _grow(self._cohesion)
        self._color = _grow(self._color)
        self._cap = new_cap

    def clear(self) -> None:
        """Remove all particles."""
        self.n = 0

    def add_particles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        density: float,
        viscosity: float,
        cohesion: float,
        color_rgb: Tuple[float, float, float],
    ) -> int:
        """Append particles up to max_particles; returns count added."""
        count = len(positions)
        if count == 0 or self.n >= self.max_particles:
            return 0
        allowed = min(count, self.max_particles - self.n)
        self._ensure_capacity(self.n + allowed)
        start = self.n
        end = start + allowed

        rho_clamped = max(0.1, float(density))
        self._pos[start:end] = positions[:allowed]
        self._vel[start:end] = velocities[:allowed]
        self._rho0[start:end] = rho_clamped
        self._inv_m[start:end] = 1.0 / rho_clamped
        self._visc[start:end] = float(viscosity)
        self._cohesion[start:end] = float(cohesion)
        self._color[start:end] = np.asarray(color_rgb, dtype=np.float32)
        self.n = end
        return allowed

    def _filter_overlaps(self, cand: np.ndarray, min_dist: float) -> np.ndarray:
        """Remove candidates closer than min_dist to existing particles."""
        if self.n == 0 or len(cand) == 0:
            return cand
        x_lo = float(np.min(cand[:, 0])) - min_dist * 2.0
        x_hi = float(np.max(cand[:, 0])) + min_dist * 2.0
        y_lo = float(np.min(cand[:, 1])) - min_dist * 2.0
        y_hi = float(np.max(cand[:, 1])) + min_dist * 2.0
        pos_cur = self.pos
        near_mask = (
            (pos_cur[:, 0] >= x_lo)
            & (pos_cur[:, 0] <= x_hi)
            & (pos_cur[:, 1] >= y_lo)
            & (pos_cur[:, 1] <= y_hi)
        )
        if not np.any(near_mask):
            return cand
        near = pos_cur[near_mask]
        min_d2 = min_dist * min_dist
        keep = np.ones(len(cand), dtype=bool)
        for s in range(0, len(cand), 256):
            blk = cand[s:s + 256]
            ddx = blk[:, 0:1] - near[:, 0][None, :]
            ddy = blk[:, 1:2] - near[:, 1][None, :]
            keep[s:s + 256] = np.min(ddx * ddx + ddy * ddy, axis=1) >= min_d2
        return cand[keep]

    def spawn_disk(
        self,
        cx: float,
        cy: float,
        radius: float,
        density: float,
        viscosity: float,
        cohesion: float,
        color_rgb: Tuple[float, float, float],
        bounds: Tuple[float, float, float, float],
        spacing: float = DEFAULT_SPACING,
        init_vel: Tuple[float, float] = (0.0, 0.0),
        max_new: int = 128,
    ) -> int:
        """Spawn non-overlapping fluid dots inside a circular brush region."""
        if self.n >= self.max_particles:
            return 0

        left, top, right, bottom = bounds
        margin = 6.0
        r_eff = max(spacing * 0.65, float(radius))

        xs = np.arange(cx - r_eff, cx + r_eff + 0.5 * spacing, spacing)
        ys = np.arange(cy - r_eff, cy + r_eff + 0.5 * spacing, spacing)
        gx, gy = np.meshgrid(xs, ys)
        cand = np.column_stack([gx.ravel(), gy.ravel()])
        cand += self._rng.uniform(-0.18 * spacing, 0.18 * spacing, size=cand.shape)

        dx = cand[:, 0] - cx
        dy = cand[:, 1] - cy
        in_disk = (dx * dx + dy * dy) <= (r_eff * r_eff)
        in_bounds = (
            (cand[:, 0] >= left + margin)
            & (cand[:, 0] <= right - margin)
            & (cand[:, 1] >= top + margin)
            & (cand[:, 1] <= bottom - margin)
        )
        cand = self._filter_overlaps(cand[in_disk & in_bounds], spacing * 0.70)
        if len(cand) == 0:
            return 0

        if len(cand) > max_new:
            idx = self._rng.choice(len(cand), size=max_new, replace=False)
            cand = cand[idx]

        vel = np.empty_like(cand)
        vel[:, 0] = init_vel[0]
        vel[:, 1] = init_vel[1]
        return self.add_particles(cand, vel, density, viscosity, cohesion, color_rgb)

    def spawn_jet(
        self,
        ox: float,
        oy: float,
        angle_rad: float,
        speed: float,
        stream_radius: float,
        density: float,
        viscosity: float,
        cohesion: float,
        color_rgb: Tuple[float, float, float],
        bounds: Tuple[float, float, float, float],
        dt: float,
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Emit a continuous pressurized laminar/parabolic liquid stream along angle_rad."""
        if self.n >= self.max_particles or dt <= 0.0:
            return 0

        ux = float(np.cos(angle_rad))
        uy = float(np.sin(angle_rad))
        px = -uy
        py = ux

        step_len = min(46.0, max(spacing * 0.9, speed * dt))
        step_ax = spacing * 0.88
        n_ax = max(1, int(np.ceil(step_len / step_ax)))
        half_w = max(0.0, min(14.0, (stream_radius - 14.0) * 0.18))
        n_lat = 1 if half_w < 2.8 else (2 if half_w < 6.5 else 3)

        ax_offs = np.linspace(0.0, step_len - step_ax * 0.5, n_ax)
        lat_offs = np.array([0.0]) if n_lat == 1 else np.linspace(-half_w, half_w, n_lat)
        ga, gl = np.meshgrid(ax_offs, lat_offs)
        ga_f = ga.ravel()
        gl_f = gl.ravel() + self._rng.uniform(-0.8, 0.8, size=ga.size)

        cand = np.column_stack([ox + ga_f * ux + gl_f * px, oy + ga_f * uy + gl_f * py])
        left, top, right, bottom = bounds
        margin = 5.5
        in_bounds = (
            (cand[:, 0] >= left + margin)
            & (cand[:, 0] <= right - margin)
            & (cand[:, 1] >= top + margin)
            & (cand[:, 1] <= bottom - margin)
        )
        cand = self._filter_overlaps(cand[in_bounds], spacing * 0.62)
        if len(cand) == 0:
            return 0

        spread = self._rng.normal(0.0, max(12.0, speed * 0.025), size=len(cand))
        vel = np.empty_like(cand)
        vel[:, 0] = speed * ux + spread * px
        vel[:, 1] = speed * uy + spread * py
        return self.add_particles(cand, vel, density, viscosity, cohesion, color_rgb)

    def spawn_from_mask(
        self,
        mask: np.ndarray,
        local_x0: float,
        local_y0: float,
        density: float,
        viscosity: float,
        cohesion: float,
        color_rgb: Tuple[float, float, float],
        bounds: Tuple[float, float, float, float],
        spacing: float = DEFAULT_SPACING,
    ) -> int:
        """Spawn fluid particles shaped like a 2D boolean pixel mask (e.g. text glyphs)."""
        if self.n >= self.max_particles or mask.size == 0:
            return 0
        mw, mh = mask.shape
        step = max(4, int(round(spacing)))
        xs = np.arange(0, mw, step)
        ys = np.arange(0, mh, step)
        if len(xs) == 0 or len(ys) == 0:
            return 0
        gx, gy = np.meshgrid(xs, ys)
        gx_f = gx.ravel()
        gy_f = gy.ravel()
        inside = mask[gx_f, gy_f]
        if not np.any(inside):
            return 0

        gx_f = gx_f[inside].astype(np.float64) + self._rng.uniform(-0.8, 0.8, size=int(np.count_nonzero(inside)))
        gy_f = gy_f[inside].astype(np.float64) + self._rng.uniform(-0.8, 0.8, size=len(gx_f))
        cand = np.column_stack([bounds[0] + local_x0 + gx_f, bounds[1] + local_y0 + gy_f])

        left, top, right, bottom = bounds
        margin = 5.5
        in_bounds = (
            (cand[:, 0] >= left + margin)
            & (cand[:, 0] <= right - margin)
            & (cand[:, 1] >= top + margin)
            & (cand[:, 1] <= bottom - margin)
        )
        cand = self._filter_overlaps(cand[in_bounds], spacing * 0.68)
        if len(cand) == 0:
            return 0

        vel = np.zeros_like(cand)
        return self.add_particles(cand, vel, density, viscosity, cohesion, color_rgb)

    def spawn_box(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        density: float,
        viscosity: float,
        cohesion: float,
        color_rgb: Tuple[float, float, float],
        spacing: float = DEFAULT_SPACING,
        init_vel: Tuple[float, float] = (0.0, 0.0),
    ) -> int:
        """Spawn a rectangular block of particles with slight jitter."""
        if x1 <= x0 or y1 <= y0 or self.n >= self.max_particles:
            return 0
        xs = np.arange(x0, x1, spacing)
        ys = np.arange(y0, y1, spacing)
        if len(xs) == 0 or len(ys) == 0:
            return 0
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pts += self._rng.uniform(-0.14 * spacing, 0.14 * spacing, size=pts.shape)
        vel = np.empty_like(pts)
        vel[:, 0] = init_vel[0]
        vel[:, 1] = init_vel[1]
        return self.add_particles(pts, vel, density, viscosity, cohesion, color_rgb)

    def spawn_demo_layers(self, bounds: Tuple[float, float, float, float]) -> None:
        """Populate an active multi-density showcase scene inside bounds."""
        self.clear()
        left, top, right, bottom = bounds
        w = max(140.0, right - left)
        h = max(140.0, bottom - top)

        foam = FLUID_PRESETS[0]
        oil = FLUID_PRESETS[1]
        water = FLUID_PRESETS[2]
        mercury = FLUID_PRESETS[5]

        # Water pool across lower ~16% of tank for dense real-fluid look & high FPS.
        self.spawn_box(
            left + 16.0,
            top + h * 0.83,
            right - 16.0,
            bottom - 8.0,
            water.density,
            water.viscosity,
            water.cohesion,
            water.color_rgb,
            spacing=6.4,
        )

        # Light oil cluster above left side.
        self.spawn_box(
            left + 24.0,
            top + h * 0.65,
            left + w * 0.34,
            top + h * 0.79,
            oil.density,
            oil.viscosity,
            oil.cohesion,
            oil.color_rgb,
            spacing=6.4,
        )

        # Dense mercury droplet falling from upper right.
        self.spawn_disk(
            left + w * 0.72,
            top + h * 0.42,
            min(w, h) * 0.075,
            mercury.density,
            mercury.viscosity,
            mercury.cohesion,
            mercury.color_rgb,
            bounds=bounds,
            spacing=6.2,
            max_new=360,
        )

        # Buried foam bubble near lower center that rises cleanly through water.
        foam_cx = left + w * 0.50
        foam_cy = bottom - min(48.0, h * 0.09)
        foam_r = min(w, h) * 0.065
        self.remove_in_radius(foam_cx, foam_cy, foam_r * 1.05)
        self.spawn_disk(
            foam_cx,
            foam_cy,
            foam_r,
            foam.density,
            foam.viscosity,
            foam.cohesion,
            foam.color_rgb,
            bounds=bounds,
            spacing=6.2,
            max_new=220,
        )

    def remove_in_radius(self, cx: float, cy: float, radius: float) -> int:
        """Erase particles within radius of (cx, cy); returns count removed."""
        if self.n == 0:
            return 0
        dx = self._pos[: self.n, 0] - cx
        dy = self._pos[: self.n, 1] - cy
        keep = (dx * dx + dy * dy) > (radius * radius)
        kept = int(np.count_nonzero(keep))
        removed = self.n - kept
        if removed == 0:
            return 0
        if kept > 0:
            self._pos[:kept] = self._pos[: self.n][keep]
            self._vel[:kept] = self._vel[: self.n][keep]
            self._rho0[:kept] = self._rho0[: self.n][keep]
            self._inv_m[:kept] = self._inv_m[: self.n][keep]
            self._visc[:kept] = self._visc[: self.n][keep]
            self._cohesion[:kept] = self._cohesion[: self.n][keep]
            self._color[:kept] = self._color[: self.n][keep]
        self.n = kept
        return removed

    def translate(self, dx: float, dy: float) -> None:
        """Translate all particles uniformly (used on window snap/teleport)."""
        if self.n == 0:
            return
        self._pos[: self.n, 0] += dx
        self._pos[: self.n, 1] += dy

    def clamp_to_bounds(
        self, left: float, top: float, right: float, bottom: float, margin: float = 5.0
    ) -> None:
        """Clamp all particles inside container bounds."""
        if self.n == 0:
            return
        x0 = min(left + margin, right - margin)
        x1 = max(left + margin, right - margin)
        y0 = min(top + margin, bottom - margin)
        y1 = max(top + margin, bottom - margin)
        np.clip(self._pos[: self.n, 0], x0, x1, out=self._pos[: self.n, 0])
        np.clip(self._pos[: self.n, 1], y0, y1, out=self._pos[: self.n, 1])
