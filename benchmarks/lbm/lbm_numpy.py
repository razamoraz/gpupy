import numpy as np
from core.configs import LBMConfig


class LBMNumPy:
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

        self.w = self.w_np
        self.cx = self.cx_np
        self.cy = self.cy_np

        self.dtype = np.float32 if config.precision == "f32" else np.float64

        # Initialize distributions
        self.fin = np.zeros((9, self.nx, self.ny), dtype=self.dtype)
        self.fout = np.zeros((9, self.nx, self.ny), dtype=self.dtype)
        self.feq = np.zeros((9, self.nx, self.ny), dtype=self.dtype)

        # Initial condition: slight perturbation
        np.random.seed(42)
        # self.rho = (1.0 + 0.01 * np.random.randn(self.nx, self.ny)).astype(self.dtype)
        self.rho = self.init_guassian_pulse()
        self.ux = np.zeros((self.nx, self.ny), dtype=self.dtype)
        self.uy = np.zeros((self.nx, self.ny), dtype=self.dtype)

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
            self.fin[i, :, :] = np.roll(
                self.fout[i, :, :], (self.cx[i], self.cy[i]), axis=(0, 1)
            )

        # Macroscopic variables (Update after streaming)
        self.rho = np.sum(self.fin, axis=0)
        self.ux = (
            np.sum(self.fin * self.cx[:, np.newaxis, np.newaxis], axis=0) / self.rho
        )
        self.uy = (
            np.sum(self.fin * self.cy[:, np.newaxis, np.newaxis], axis=0) / self.rho
        )

    def run(self):
        for _ in range(self.cfg.iterations):
            self.step()

    def equilibrium(self):
        u2 = self.ux**2 + self.uy**2
        feq = np.zeros((9, self.nx, self.ny))
        for i in range(9):
            cu = self.cx[i] * self.ux + self.cy[i] * self.uy
            feq[i, :, :] = self.rho * self.w[i] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
        return feq

    def init_guassian_pulse(self, fluct_density=1e-3):
        mshx = np.linspace(-1, 1, self.nx, dtype=self.dtype)
        mshy = np.linspace(-1, 1, self.ny, dtype=self.dtype)
        x, y = np.meshgrid(mshx, mshy)
        rho = 1 + fluct_density * np.exp(-100 * (x * x + y * y))
        return rho
