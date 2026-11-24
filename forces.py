"""Numba-accelerated Multi-Density Position-Based Fluid (PBF) solver."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

from particles import ParticleSystem

Bounds = Tuple[float, float, float, float]


def _calibrate_rest_number_density(h: float, spacing: float) -> float:
    """Compute interior rest number density theta0 for (1 - u^2)^3 kernel."""
    xs = np.arange(0.0, 140.1, spacing)
    ys = np.arange(0.0, 140.1, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    dx = pts[:, 0:1] - pts[:, 0][None, :]
    dy = pts[:, 1:2] - pts[:, 1][None, :]
    r2 = dx * dx + dy * dy
    u2 = r2 / (h * h)
    w = np.where(u2 < 1.0, (1.0 - u2) ** 3, 0.0)
    theta = np.sum(w, axis=1)
    # 80% of interior lattice packing keeps hydrostatic pressure smoothly continuous.
    return float(np.median(theta)) * 0.80


@njit(fastmath=True, inline="always")
def _push_out_of_sdf(
    px: float,
    py: float,
    bx0: float,
    by0: float,
    margin: float,
    has_sdf: bool,
    sdf: np.ndarray,
    sdf_w: int,
    sdf_h: int,
) -> Tuple[float, float]:
    """Push particle out of solid SDF obstacles along local surface normal."""
    if not has_sdf:
        return px, py
    lx = int(px - bx0)
    ly = int(py - by0)
    if 1 <= lx < sdf_w - 1 and 1 <= ly < sdf_h - 1:
        d = sdf[lx, ly]
        if d < margin:
            gx = sdf[lx + 1, ly] - sdf[lx - 1, ly]
            gy = sdf[lx, ly + 1] - sdf[lx, ly - 1]
            glen = math.sqrt(gx * gx + gy * gy)
            if glen > 1e-5:
                push = margin - d + 0.12
                px += (gx / glen) * push
                py += (gy / glen) * push
    return px, py


@njit(fastmath=True, parallel=True, cache=True)
def _predict_step(
    n: int,
    pos: np.ndarray,
    prev_pos: np.ndarray,
    vel: np.ndarray,
    bx0: float,
    by0: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    margin: float,
    has_sdf: bool,
    sdf: np.ndarray,
    sdf_w: int,
    sdf_h: int,
    gravity: float,
    dt: float,
) -> None:
    max_v2 = 3000.0 * 3000.0
    for i in prange(n):
        vx = vel[i, 0]
        vy = vel[i, 1] + gravity * dt
        v2 = vx * vx + vy * vy
        if v2 > max_v2:
            sc = 3000.0 / math.sqrt(v2)
            vx *= sc
            vy *= sc
        vel[i, 0] = vx
        vel[i, 1] = vy

        px = pos[i, 0]
        py = pos[i, 1]
        prev_pos[i, 0] = px
        prev_pos[i, 1] = py
        px += vx * dt
        py += vy * dt
        px, py = _push_out_of_sdf(px, py, bx0, by0, margin, has_sdf, sdf, sdf_w, sdf_h)
        if px < x0:
            px = x0
        elif px > x1:
            px = x1
        if py < y0:
            py = y0
        elif py > y1:
            py = y1
        pos[i, 0] = px
        pos[i, 1] = py


@njit(fastmath=True, cache=True)
def _build_spatial_grid(
    n: int,
    pos: np.ndarray,
    bx0: float,
    by0: float,
    inv_h: float,
    gx: int,
    gy: int,
    cell_count: np.ndarray,
    cell_start: np.ndarray,
    sorted_idx: np.ndarray,
) -> None:
    ncells = gx * gy
    for c in range(ncells):
        cell_count[c] = 0

    for i in range(n):
        cx = int((pos[i, 0] - bx0) * inv_h)
        cy = int((pos[i, 1] - by0) * inv_h)
        if cx < 0:
            cx = 0
        elif cx >= gx:
            cx = gx - 1
        if cy < 0:
            cy = 0
        elif cy >= gy:
            cy = gy - 1
        cell_count[cy * gx + cx] += 1

    acc = 0
    for c in range(ncells):
        cell_start[c] = acc
        acc += cell_count[c]
        cell_count[c] = 0

    for i in range(n):
        cx = int((pos[i, 0] - bx0) * inv_h)
        cy = int((pos[i, 1] - by0) * inv_h)
        if cx < 0:
            cx = 0
        elif cx >= gx:
            cx = gx - 1
        if cy < 0:
            cy = 0
        elif cy >= gy:
            cy = gy - 1
        c = cy * gx + cx
        sorted_idx[cell_start[c] + cell_count[c]] = i
        cell_count[c] += 1


@njit(fastmath=True, parallel=True, cache=True)
def _collect_neighbors(
    n: int,
    pos: np.ndarray,
    bx0: float,
    by0: float,
    inv_h: float,
    gx: int,
    gy: int,
    search_r2: float,
    cell_count: np.ndarray,
    cell_start: np.ndarray,
    sorted_idx: np.ndarray,
    neigh: np.ndarray,
    neigh_cnt: np.ndarray,
) -> None:
    max_neigh = neigh.shape[1]
    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]
        cx = int((px - bx0) * inv_h)
        cy = int((py - by0) * inv_h)
        if cx < 0:
            cx = 0
        elif cx >= gx:
            cx = gx - 1
        if cy < 0:
            cy = 0
        elif cy >= gy:
            cy = gy - 1

        y_lo = cy - 1 if cy > 0 else 0
        y_hi = cy + 1 if cy + 1 < gy else gy - 1
        x_lo = cx - 1 if cx > 0 else 0
        x_hi = cx + 1 if cx + 1 < gx else gx - 1

        cnt = 0
        for ny_c in range(y_lo, y_hi + 1):
            row = ny_c * gx
            for nx_c in range(x_lo, x_hi + 1):
                c = row + nx_c
                st = cell_start[c]
                en = st + cell_count[c]
                for idx in range(st, en):
                    j = sorted_idx[idx]
                    if j != i:
                        dx = px - pos[j, 0]
                        dy = py - pos[j, 1]
                        if dx * dx + dy * dy < search_r2:
                            neigh[i, cnt] = j
                            cnt += 1
                            if cnt == max_neigh:
                                break
                if cnt == max_neigh:
                    break
            if cnt == max_neigh:
                break
        neigh_cnt[i] = cnt


@njit(fastmath=True, parallel=True, cache=True)
def _pbf_compute_lambda(
    n: int,
    pos: np.ndarray,
    inv_m: np.ndarray,
    neigh: np.ndarray,
    neigh_cnt: np.ndarray,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    h2: float,
    inv_h: float,
    inv_theta0: float,
    spiky_coeff: float,
    inv_wall_band: float,
    has_sdf: bool,
    sdf: np.ndarray,
    sdf_w: int,
    sdf_h: int,
    theta_arr: np.ndarray,
    lam_arr: np.ndarray,
) -> None:
    eps = 1e-4
    wall_band = 1.0 / inv_wall_band
    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]

        d_bot = max(0.0, 1.0 - (by1 - py) * inv_wall_band)
        d_top = max(0.0, 1.0 - (py - by0) * inv_wall_band)
        d_left = max(0.0, 1.0 - (px - bx0) * inv_wall_band)
        d_right = max(0.0, 1.0 - (bx1 - px) * inv_wall_band)
        theta = 1.0 + 0.72 * (d_bot * d_bot + d_top * d_top + d_left * d_left + d_right * d_right)

        if has_sdf:
            lx = int(px - bx0)
            ly = int(py - by0)
            if 0 <= lx < sdf_w and 0 <= ly < sdf_h:
                d_s = sdf[lx, ly]
                if d_s < wall_band:
                    dw = 1.0 - max(0.0, d_s) * inv_wall_band
                    theta += 0.72 * dw * dw

        sum_gx = 0.0
        sum_gy = 0.0
        sum_g2 = 0.0
        cnt = neigh_cnt[i]
        for k in range(cnt):
            j = neigh[i, k]
            dx = px - pos[j, 0]
            dy = py - pos[j, 1]
            r2 = dx * dx + dy * dy
            if r2 < h2:
                r = math.sqrt(r2)
                if r < 1e-4:
                    dx = 1e-3
                    dy = 1e-3
                    r = 1.4142e-3
                u = r * inv_h
                om_u2 = 1.0 - u * u
                theta += om_u2 * om_u2 * om_u2
                q = 1.0 - u
                sp = spiky_coeff * (q * q) / r
                gx_ij = sp * dx
                gy_ij = sp * dy
                sum_gx += gx_ij
                sum_gy += gy_ij
                sum_g2 += inv_m[j] * (gx_ij * gx_ij + gy_ij * gy_ij)

        theta_arr[i] = theta
        C = theta * inv_theta0 - 1.0
        if C <= 0.0:
            lam_arr[i] = 0.0
        else:
            denom = inv_m[i] * (sum_gx * sum_gx + sum_gy * sum_gy) + sum_g2 + eps
            lam_arr[i] = -C / denom


@njit(fastmath=True, parallel=True, cache=True)
def _pbf_compute_delta(
    n: int,
    pos: np.ndarray,
    inv_m: np.ndarray,
    neigh: np.ndarray,
    neigh_cnt: np.ndarray,
    lam_arr: np.ndarray,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    h2: float,
    inv_h: float,
    spiky_coeff: float,
    w_dq_inv: float,
    inv_wall_band: float,
    dp_x: np.ndarray,
    dp_y: np.ndarray,
) -> None:
    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]
        lam_i = lam_arr[i]
        dpx = 0.0
        dpy = 0.0
        cnt = neigh_cnt[i]
        for k in range(cnt):
            j = neigh[i, k]
            dx = px - pos[j, 0]
            dy = py - pos[j, 1]
            r2 = dx * dx + dy * dy
            if r2 < h2:
                r = math.sqrt(r2)
                if r < 1e-4:
                    dx = 1e-3
                    dy = 1e-3
                    r = 1.4142e-3
                u = r * inv_h
                om_u2 = 1.0 - u * u
                w_val = om_u2 * om_u2 * om_u2
                wr = w_val * w_dq_inv
                wr2 = wr * wr
                scorr = -0.002 * (wr2 * wr2)
                q = 1.0 - u
                sp = spiky_coeff * (q * q) / r
                ls = lam_i + lam_arr[j] + scorr
                dpx += ls * sp * dx
                dpy += ls * sp * dy

        wi = inv_m[i]
        d_bot = max(0.0, 1.0 - (by1 - py) * inv_wall_band)
        d_top = max(0.0, 1.0 - (py - by0) * inv_wall_band)
        d_left = max(0.0, 1.0 - (px - bx0) * inv_wall_band)
        d_right = max(0.0, 1.0 - (bx1 - px) * inv_wall_band)
        wp = wi * (-lam_i if lam_i < 0.0 else 0.0) * 0.14
        dp_x[i] = 0.5 * wi * dpx + wp * (d_left - d_right)
        dp_y[i] = 0.5 * wi * dpy + wp * (d_top - d_bot)


@njit(fastmath=True, parallel=True, cache=True)
def _pbf_apply_delta(
    n: int,
    pos: np.ndarray,
    dp_x: np.ndarray,
    dp_y: np.ndarray,
    bx0: float,
    by0: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    margin: float,
    has_sdf: bool,
    sdf: np.ndarray,
    sdf_w: int,
    sdf_h: int,
) -> None:
    for i in prange(n):
        nx = pos[i, 0] + dp_x[i]
        ny = pos[i, 1] + dp_y[i]
        nx, ny = _push_out_of_sdf(nx, ny, bx0, by0, margin, has_sdf, sdf, sdf_w, sdf_h)
        if nx < x0:
            nx = x0
        elif nx > x1:
            nx = x1
        if ny < y0:
            ny = y0
        elif ny > y1:
            ny = y1
        pos[i, 0] = nx
        pos[i, 1] = ny


@njit(fastmath=True, parallel=True, cache=True)
def _update_vel_buoy_coh_xsph(
    n: int,
    pos: np.ndarray,
    prev_pos: np.ndarray,
    vel: np.ndarray,
    rho0: np.ndarray,
    inv_m: np.ndarray,
    visc: np.ndarray,
    coh: np.ndarray,
    neigh: np.ndarray,
    neigh_cnt: np.ndarray,
    theta_arr: np.ndarray,
    h2: float,
    inv_h: float,
    theta0: float,
    g_eff_x: float,
    g_eff_y: float,
    dt: float,
    damping: float,
    out_vx: np.ndarray,
    out_vy: np.ndarray,
) -> None:
    inv_dt = 1.0 / dt
    for i in prange(n):
        vel[i, 0] = (pos[i, 0] - prev_pos[i, 0]) * inv_dt
        vel[i, 1] = (pos[i, 1] - prev_pos[i, 1]) * inv_dt

    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]
        vx_i = vel[i, 0]
        vy_i = vel[i, 1]
        rho_i = rho0[i]
        wi = inv_m[i]
        visc_i = visc[i]
        coh_i = coh[i]
        theta_i = theta_arr[i] if theta_arr[i] > 1.0 else 1.0

        w_neigh = 0.0
        rho_neigh = 0.0
        dvx_sum = 0.0
        dvy_sum = 0.0
        fx_coh = 0.0
        fy_coh = 0.0

        cnt = neigh_cnt[i]
        for k in range(cnt):
            j = neigh[i, k]
            dx = px - pos[j, 0]
            dy = py - pos[j, 1]
            r2 = dx * dx + dy * dy
            if r2 < h2 and r2 > 1e-8:
                r = math.sqrt(r2)
                u = r * inv_h
                om_u2 = 1.0 - u * u
                w_val = om_u2 * om_u2 * om_u2
                rho_j = rho0[j]
                w_neigh += w_val
                rho_neigh += w_val * rho_j

                vp = 0.5 * (visc_i + visc[j]) * w_val / theta_i
                dvx_sum += vp * (vel[j, 0] - vx_i)
                dvy_sum += vp * (vel[j, 1] - vy_i)

                if 0.55 < u < 0.98:
                    bell = (u - 0.55) * (0.98 - u) * 21.0
                    rel_d = (rho_i - rho_j) / (0.28 * (rho_i + rho_j))
                    same_p = 1.0 / (1.0 + rel_d * rel_d)
                    cm = (1100.0 * dt) * 0.5 * (coh_i + coh[j]) * same_p * bell / r
                    fx_coh -= cm * dx
                    fy_coh -= cm * dy

        ax_b = 0.0
        ay_b = 0.0
        if w_neigh > 0.25:
            rho_bar = rho_neigh / w_neigh
            imm = (w_neigh - 0.20) / (0.50 * theta0)
            if imm > 1.0:
                imm = 1.0
            elif imm < 0.0:
                imm = 0.0
            buoy = (rho_i - rho_bar) / rho_i
            if buoy > 1.6:
                buoy = 1.6
            elif buoy < -1.6:
                buoy = -1.6
            b_fac = buoy * 0.45 * imm * dt
            ax_b = b_fac * g_eff_x
            ay_b = b_fac * g_eff_y

        out_vx[i] = (vx_i + dvx_sum + wi * fx_coh + ax_b) * damping
        out_vy[i] = (vy_i + dvy_sum + wi * fy_coh + ay_b) * damping

    for i in prange(n):
        vel[i, 0] = out_vx[i]
        vel[i, 1] = out_vy[i]


class FluidPhysicsSolver:
    """High-speed Numba spatial-grid PBF solver with moving container bounds."""

    MAX_NEIGHBORS = 48

    def __init__(self, smoothing_radius: float = 12.2, spacing: float = 6.2) -> None:
        self.h = float(smoothing_radius)
        self.h2 = self.h * self.h
        self.inv_h = 1.0 / self.h
        self.search_r2 = (self.h * 1.12) ** 2
        self.spacing = float(spacing)
        self.wall_margin = 4.5

        self.theta0 = _calibrate_rest_number_density(self.h, self.spacing)
        self.inv_theta0 = 1.0 / self.theta0
        self.spiky_coeff = -2.5 * self.inv_h * self.inv_theta0
        self.w_dq_inv = 1.0 / ((1.0 - 0.25 * 0.25) ** 3)
        self.inv_wall_band = 1.0 / (0.6 * self.h)

        self._win_vel = np.zeros(2, dtype=np.float64)
        self._win_acc = np.zeros(2, dtype=np.float64)

        # Preallocated scratch buffers (never allocate during simulation frames).
        cap_p = 10000
        self._prev_pos = np.zeros((cap_p, 2), dtype=np.float64)
        self._sorted_idx = np.zeros(cap_p, dtype=np.int32)
        self._neigh = np.zeros((cap_p, self.MAX_NEIGHBORS), dtype=np.int32)
        self._neigh_cnt = np.zeros(cap_p, dtype=np.int32)
        self._theta = np.zeros(cap_p, dtype=np.float64)
        self._lam = np.zeros(cap_p, dtype=np.float64)
        self._dp_x = np.zeros(cap_p, dtype=np.float64)
        self._dp_y = np.zeros(cap_p, dtype=np.float64)

        self._cell_cap = 32768
        self._cell_count = np.zeros(self._cell_cap, dtype=np.int32)
        self._cell_start = np.zeros(self._cell_cap, dtype=np.int32)
        self._dummy_sdf = np.zeros((1, 1), dtype=np.float32)

        self.last_physics_ms = 0.0
        self.last_substeps = 1
        self.last_win_speed = 0.0

    def _ensure_particle_scratch(self, n: int) -> None:
        if n <= len(self._sorted_idx):
            return
        cap = max(n, len(self._sorted_idx) * 2)
        self._prev_pos = np.zeros((cap, 2), dtype=np.float64)
        self._sorted_idx = np.zeros(cap, dtype=np.int32)
        self._neigh = np.zeros((cap, self.MAX_NEIGHBORS), dtype=np.int32)
        self._neigh_cnt = np.zeros(cap, dtype=np.int32)
        self._theta = np.zeros(cap, dtype=np.float64)
        self._lam = np.zeros(cap, dtype=np.float64)
        self._dp_x = np.zeros(cap, dtype=np.float64)
        self._dp_y = np.zeros(cap, dtype=np.float64)

    def _ensure_grid_scratch(self, ncells: int) -> None:
        if ncells <= self._cell_cap:
            return
        cap = max(ncells, self._cell_cap * 2)
        self._cell_cap = cap
        self._cell_count = np.zeros(cap, dtype=np.int32)
        self._cell_start = np.zeros(cap, dtype=np.int32)

    def reset_kinematics(self) -> None:
        """Reset tracked window velocity and acceleration."""
        self._win_vel.fill(0.0)
        self._win_acc.fill(0.0)
        self.last_win_speed = 0.0

    def apply_brush_force(
        self,
        particles: ParticleSystem,
        wx: float,
        wy: float,
        vx_mouse: float,
        vy_mouse: float,
        radius: float,
        mode: str,
        dt: float,
    ) -> None:
        """Apply interactive push/stir or vortex/pull forces within a circular brush."""
        if particles.n == 0 or dt <= 0.0:
            return
        pos = particles.pos
        vel = particles.vel

        dx = pos[:, 0] - wx
        dy = pos[:, 1] - wy
        r2 = dx * dx + dy * dy
        r_max = max(12.0, float(radius))
        mask = r2 < (r_max * r_max)
        if not np.any(mask):
            return

        r = np.sqrt(r2[mask]) + 1e-6
        falloff = 1.0 - r / r_max
        mass_scale = np.clip(np.sqrt(particles.inv_m[mask]), 0.35, 1.8)

        if mode == "push":
            speed_m = float(np.hypot(vx_mouse, vy_mouse))
            if speed_m > 15.0:
                gain = np.clip(falloff * 14.0 * dt * mass_scale, 0.0, 0.85)
                vel[mask, 0] += (vx_mouse - vel[mask, 0]) * gain
                vel[mask, 1] += (vy_mouse - vel[mask, 1]) * gain
            else:
                impulse = 1600.0 * falloff * dt * mass_scale
                vel[mask, 0] += (dx[mask] / r) * impulse
                vel[mask, 1] += (dy[mask] / r) * impulse
        elif mode == "vortex":
            nx = dx[mask] / r
            ny = dy[mask] / r
            pull = 1500.0 * falloff * dt * mass_scale
            swirl = 1200.0 * falloff * dt * mass_scale
            vel[mask, 0] += -nx * pull - ny * swirl
            vel[mask, 1] += -ny * pull + nx * swirl

    def step(
        self,
        particles: ParticleSystem,
        curr_bounds: Bounds,
        prev_bounds: Bounds,
        dt: float,
        gravity: float = 720.0,
        damping: float = 0.9985,
        sdf: Optional[np.ndarray] = None,
        dragging: bool = False,
    ) -> None:
        """Advance fluid simulation by dt under moving/resizing bounds and optional SDF obstacles."""
        t_start = time.perf_counter()
        if dt <= 0.0:
            self.last_physics_ms = 0.0
            return

        cl, ct, cr, cb = curr_bounds
        pl, pt, pr, pb = prev_bounds
        cw = max(32.0, cr - cl)
        ch = max(32.0, cb - ct)
        pw = max(32.0, pr - pl)
        ph = max(32.0, pb - pt)

        dx_win = cl - pl
        dy_win = ct - pt
        dw = cw - pw
        dh = ch - ph

        # Guard against window maximize/restore/snap teleports without blocking fast dragging.
        resized_big = abs(dw) > 260.0 or abs(dh) > 260.0
        moved_teleport = (not dragging) and (abs(dx_win) > 360.0 or abs(dy_win) > 360.0)
        if resized_big or moved_teleport:
            if particles.n > 0:
                if resized_big:
                    pos = particles.pos
                    pos[:, 0] = cl + ((pos[:, 0] - pl) / pw) * cw
                    pos[:, 1] = ct + ((pos[:, 1] - pt) / ph) * ch
                else:
                    particles.translate(dx_win, dy_win)
                particles.clamp_to_bounds(cl, ct, cr, cb, self.wall_margin)
            self.reset_kinematics()
            prev_bounds = curr_bounds
            pl, pt, pr, pb = prev_bounds
            dx_win = dy_win = 0.0

        inst_vx = dx_win / dt
        inst_vy = dy_win / dt
        raw_ax = (inst_vx - self._win_vel[0]) / dt
        raw_ay = (inst_vy - self._win_vel[1]) / dt
        self._win_vel[0] = inst_vx
        self._win_vel[1] = inst_vy
        self._win_acc[0] = 0.65 * self._win_acc[0] + 0.35 * max(-6000.0, min(6000.0, raw_ax))
        self._win_acc[1] = 0.65 * self._win_acc[1] + 0.35 * max(-6000.0, min(6000.0, raw_ay))
        self.last_win_speed = float(math.hypot(inst_vx, inst_vy))

        n = particles.n
        if n == 0:
            self.last_substeps = 1
            self.last_physics_ms = (time.perf_counter() - t_start) * 1000.0
            return

        self._ensure_particle_scratch(n)
        pos = particles.pos
        vel = particles.vel
        rho0 = particles.rho0
        inv_m = particles.inv_m
        visc = particles.visc
        coh = particles.cohesion

        wall_move = max(abs(cl - pl), abs(ct - pt), abs(cr - pr), abs(cb - pb))
        max_v2 = float(np.max(vel[:, 0] * vel[:, 0] + vel[:, 1] * vel[:, 1]))
        max_disp = max(wall_move, math.sqrt(max_v2) * dt)

        substeps = 1 if max_disp <= 9.5 else min(3, max(2, int(math.ceil(max_disp / 8.5))))
        solver_iters = 3 if substeps == 1 else 2
        sub_dt = dt / substeps

        g_eff_x = -self._win_acc[0] * 0.55
        g_eff_y = gravity - self._win_acc[1] * 0.55

        margin = self.wall_margin
        inv_h = self.inv_h
        has_sdf = sdf is not None
        sdf_arr = sdf if sdf is not None else self._dummy_sdf
        sdf_w = int(sdf_arr.shape[0])
        sdf_h = int(sdf_arr.shape[1])

        for sub in range(substeps):
            frac = (sub + 1) / substeps
            sl = pl + (cl - pl) * frac
            st = pt + (ct - pt) * frac
            sr = pr + (cr - pr) * frac
            sb = pb + (cb - pb) * frac

            x0 = min(sl + margin, sr - margin)
            x1 = max(sl + margin, sr - margin)
            y0 = min(st + margin, sb - margin)
            y1 = max(st + margin, sb - margin)

            _predict_step(
                n, pos, self._prev_pos, vel, sl, st, x0, x1, y0, y1,
                margin, has_sdf, sdf_arr, sdf_w, sdf_h, gravity, sub_dt,
            )

            gx = max(1, int((sr - sl) * inv_h) + 1)
            gy = max(1, int((sb - st) * inv_h) + 1)
            self._ensure_grid_scratch(gx * gy)

            _build_spatial_grid(
                n, pos, sl, st, inv_h, gx, gy, self._cell_count, self._cell_start, self._sorted_idx
            )
            _collect_neighbors(
                n,
                pos,
                sl,
                st,
                inv_h,
                gx,
                gy,
                self.search_r2,
                self._cell_count,
                self._cell_start,
                self._sorted_idx,
                self._neigh,
                self._neigh_cnt,
            )

            for _it in range(solver_iters):
                _pbf_compute_lambda(
                    n,
                    pos,
                    inv_m,
                    self._neigh,
                    self._neigh_cnt,
                    sl,
                    st,
                    sr,
                    sb,
                    self.h2,
                    inv_h,
                    self.inv_theta0,
                    self.spiky_coeff,
                    self.inv_wall_band,
                    has_sdf,
                    sdf_arr,
                    sdf_w,
                    sdf_h,
                    self._theta,
                    self._lam,
                )
                _pbf_compute_delta(
                    n,
                    pos,
                    inv_m,
                    self._neigh,
                    self._neigh_cnt,
                    self._lam,
                    sl,
                    st,
                    sr,
                    sb,
                    self.h2,
                    inv_h,
                    self.spiky_coeff,
                    self.w_dq_inv,
                    self.inv_wall_band,
                    self._dp_x,
                    self._dp_y,
                )
                _pbf_apply_delta(
                    n, pos, self._dp_x, self._dp_y, sl, st, x0, x1, y0, y1,
                    margin, has_sdf, sdf_arr, sdf_w, sdf_h,
                )

            _update_vel_buoy_coh_xsph(
                n,
                pos,
                self._prev_pos,
                vel,
                rho0,
                inv_m,
                visc,
                coh,
                self._neigh,
                self._neigh_cnt,
                self._theta,
                self.h2,
                inv_h,
                self.theta0,
                g_eff_x,
                g_eff_y,
                sub_dt,
                damping,
                self._dp_x,
                self._dp_y,
            )

        self.last_substeps = substeps
        self.last_physics_ms = (time.perf_counter() - t_start) * 1000.0
