import cupy as cp
import numpy as np
from core.configs import LBMConfig


class LBMCuPy:
    def __init__(self, config: LBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.omega = config.omega

        # D2Q9 Lattice constants
        self.w_np = np.array(
            [
                config.w_0,
                config.w_1,
                config.w_1,
                config.w_1,
                config.w_1,
                config.w_2,
                config.w_2,
                config.w_2,
                config.w_2,
            ]
        )
        self.cx_np = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
        self.cy_np = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

        self.dtype = np.float32 if config.precision == "f32" else np.float64

        self.w = cp.asarray(self.w_np, dtype=self.dtype)
        self.cx = cp.asarray(self.cx_np)
        self.cy = cp.asarray(self.cy_np)

        # Initialize distributions
        self.fin = cp.zeros((9, self.nx, self.ny), dtype=self.dtype)
        self.fout = cp.zeros((9, self.nx, self.ny), dtype=self.dtype)
        self.feq = cp.zeros((9, self.nx, self.ny), dtype=self.dtype)
        self.rho = cp.zeros((self.nx, self.ny), dtype=self.dtype)
        self.ux = cp.zeros((self.nx, self.ny), dtype=self.dtype)
        self.uy = cp.zeros((self.nx, self.ny), dtype=self.dtype)

        # Initial condition: slight perturbation
        # Use numpy for random generation to match seed easily, then transfer
        np.random.seed(42)
        # h_rho = 1.0 + 0.01 * np.random.randn(self.nx, self.ny).astype(self.dtype)
        self.rho = self.init_guassian_pulse()
        self.ux = cp.zeros((self.nx, self.ny), dtype=self.dtype)
        self.uy = cp.zeros((self.nx, self.ny), dtype=self.dtype)

        # Compute Equilibrium
        # u2 = self.ux**2 + self.uy**2
        # for i in range(9):
        #     cu = self.cx[i] * self.ux + self.cy[i] * self.uy
        #     self.fin[i, :, :] = (
        #         self.rho * self.w[i] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        #     )
        self.fin = self.equilibrium()

    def step(self):
        # Collision
        self.fout = self.fin - self.omega * (self.fin - self.equilibrium())

        # Streaming (Period Boundary Conditions)
        for i in range(9):
            self.fin[i, :, :] = cp.roll(
                self.fout[i, :, :], (int(self.cx[i]), int(self.cy[i])), axis=(0, 1)
            )

        # Macroscopic variables (Update after streaming)
        self.rho = cp.sum(self.fin, axis=0)
        self.ux = (
            cp.sum(self.fin * self.cx[:, cp.newaxis, cp.newaxis], axis=0) / self.rho
        )
        self.uy = (
            cp.sum(self.fin * self.cy[:, cp.newaxis, cp.newaxis], axis=0) / self.rho
        )

    def run(self):
        # Synchronize before starting timing (handled by profiler mainly, but good practice)
        cp.cuda.Stream.null.synchronize()
        for _ in range(self.cfg.iterations):
            self.step()
        cp.cuda.Stream.null.synchronize()

    def equilibrium(self):
        u2 = self.ux**2 + self.uy**2
        feq = cp.zeros((9, self.nx, self.ny))
        for i in range(9):
            cu = self.cx[i] * self.ux + self.cy[i] * self.uy
            feq[i, :, :] = self.rho * self.w[i] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    def init_guassian_pulse(self, fluct_density=1e-3):
        mshx = cp.linspace(-1, 1, self.nx, dtype=self.dtype)
        mshy = cp.linspace(-1, 1, self.ny, dtype=self.dtype)
        x, y = cp.meshgrid(mshx, mshy)
        rho = 1 + fluct_density * cp.exp(-100 * (x * x + y * y))
        return rho
