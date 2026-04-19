"""
Week 5: ns_solver.py
Minimal educational 2D periodic vorticity-form Navier-Stokes solver.
Pseudo-spectral in space, RK4 in time, 2/3 de-aliasing.

Governing equation:
    dω/dt + u·∇ω = ν Δω
with periodic boundary conditions on [0, 2π)^2.

This is designed for dataset generation and learning, not production CFD.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class NSSolverConfig:
    resolution: int = 128
    domain_length: float = 2 * np.pi
    viscosity: float = 1e-3
    dt: float = 1e-3
    t_final: float = 1.0
    dealias: bool = True


class NavierStokes2DVorticitySolver:
    def __init__(self, config: NSSolverConfig):
        self.config = config
        self.n = config.resolution
        self.L = config.domain_length
        self.nu = config.viscosity
        self.dt = config.dt
        self.t_final = config.t_final
        self.steps = int(round(self.t_final / self.dt))

        dx = self.L / self.n
        k = 2 * np.pi * np.fft.fftfreq(self.n, d=dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.inv_k2 = np.zeros_like(self.k2)
        self.inv_k2[self.k2 > 0] = 1.0 / self.k2[self.k2 > 0]

        if config.dealias:
            idx = np.fft.fftfreq(self.n) * self.n
            ix, iy = np.meshgrid(idx, idx, indexing="ij")
            cut = self.n // 3
            self.dealias_mask = ((np.abs(ix) <= cut) & (np.abs(iy) <= cut)).astype(np.float64)
        else:
            self.dealias_mask = np.ones((self.n, self.n), dtype=np.float64)

    def _apply_dealias(self, field_hat: np.ndarray) -> np.ndarray:
        return field_hat * self.dealias_mask

    def _velocity_from_vorticity_hat(self, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = -w_hat * self.inv_k2
        psi_hat[0, 0] = 0.0
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        return u, v

    def _rhs(self, w: np.ndarray) -> np.ndarray:
        w_hat = np.fft.fft2(w)
        w_hat = self._apply_dealias(w_hat)

        u, v = self._velocity_from_vorticity_hat(w_hat)

        dw_dx = np.fft.ifft2(1j * self.kx * w_hat).real
        dw_dy = np.fft.ifft2(1j * self.ky * w_hat).real
        convection = u * dw_dx + v * dw_dy

        convection_hat = self._apply_dealias(np.fft.fft2(convection))
        diffusion_hat = -self.nu * self.k2 * w_hat
        rhs_hat = -convection_hat + diffusion_hat
        rhs = np.fft.ifft2(rhs_hat).real
        return rhs

    def step_rk4(self, w: np.ndarray) -> np.ndarray:
        dt = self.dt
        k1 = self._rhs(w)
        k2 = self._rhs(w + 0.5 * dt * k1)
        k3 = self._rhs(w + 0.5 * dt * k2)
        k4 = self._rhs(w + dt * k3)
        w_next = w + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return w_next.astype(np.float32)

    def solve(self, initial_vorticity: np.ndarray, return_trajectory: bool = False):
        w = np.asarray(initial_vorticity, dtype=np.float32).copy()
        if w.shape != (self.n, self.n):
            raise ValueError(f"Expected input shape {(self.n, self.n)}, got {w.shape}")

        if return_trajectory:
            traj = [w.copy()]
            for _ in range(self.steps):
                w = self.step_rk4(w)
                traj.append(w.copy())
            return np.stack(traj, axis=0)

        for _ in range(self.steps):
            w = self.step_rk4(w)
        return w
