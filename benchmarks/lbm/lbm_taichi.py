import taichi as ti
import numpy as np
from core.configs import LBMConfig

# Initialize Taichi
try:
    ti.init(arch=ti.gpu)
except:
    pass


@ti.kernel
def init_kernel(
    rho: ti.types.ndarray(),
    ux: ti.types.ndarray(),
    uy: ti.types.ndarray(),
    fin: ti.types.ndarray(),
    w: ti.types.ndarray(),
    cx: ti.types.ndarray(),
    cy: ti.types.ndarray(),
    dtype: ti.template(),
):
    for x, y in rho:
        rho_val = rho[x, y]
        ux_val = ux[x, y]
        uy_val = uy[x, y]

        u2 = ux_val * ux_val + uy_val * uy_val

        for i in range(9):
            cu = ti.cast(cx[i], dtype) * ux_val + ti.cast(cy[i], dtype) * uy_val
            fin[i, x, y] = rho_val * w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


@ti.kernel
def update_macro_kernel(
    fin: ti.types.ndarray(),
    rho: ti.types.ndarray(),
    ux: ti.types.ndarray(),
    uy: ti.types.ndarray(),
    cx: ti.types.ndarray(),
    cy: ti.types.ndarray(),
    dtype: ti.template(),
):
    for x, y in rho:
        rho_val = ti.cast(0.0, dtype)
        ux_val = ti.cast(0.0, dtype)
        uy_val = ti.cast(0.0, dtype)
        for i in range(9):
            f = fin[i, x, y]
            rho_val += f
            ux_val += f * ti.cast(cx[i], dtype)
            uy_val += f * ti.cast(cy[i], dtype)
        rho[x, y] = rho_val
        ux[x, y] = ux_val / rho_val
        uy[x, y] = uy_val / rho_val


@ti.kernel
def lbm_step_kernel(
    fin: ti.types.ndarray(),
    fout: ti.types.ndarray(),
    rho: ti.types.ndarray(),
    ux: ti.types.ndarray(),
    uy: ti.types.ndarray(),
    w: ti.types.ndarray(),
    cx: ti.types.ndarray(),
    cy: ti.types.ndarray(),
    omega: float,
    nx: int,
    ny: int,
    dtype: ti.template(),
):
    # Parallelize over grid
    for x, y in rho:
        # Macroscopic (Needed for feq)
        rho_val = ti.cast(0.0, dtype)
        ux_val = ti.cast(0.0, dtype)
        uy_val = ti.cast(0.0, dtype)

        for i in range(9):
            f = fin[i, x, y]
            rho_val += f
            ux_val += f * ti.cast(cx[i], dtype)
            uy_val += f * ti.cast(cy[i], dtype)

        ux_val /= rho_val
        uy_val /= rho_val
        u2 = ux_val * ux_val + uy_val * uy_val

        # Collision & Streaming
        for i in range(9):
            cu = ti.cast(cx[i], dtype) * ux_val + ti.cast(cy[i], dtype) * uy_val
            feq = rho_val * w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

            f_val = fin[i, x, y]
            f_post = f_val - omega * (f_val - feq)

            # Stream
            next_x = (x + cx[i] + nx) % nx
            next_y = (y + cy[i] + ny) % ny

            fout[i, next_x, next_y] = f_post


class LBMTaichi:
    def __init__(self, config: LBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.omega = config.omega

        self.dtype = ti.f32 if config.precision == "f32" else ti.f64
        self.np_dtype = np.float32 if config.precision == "f32" else np.float64

        # Data allocation using ti.ndarray (interop friendly, handles memory on GPU)
        self.fin = ti.ndarray(dtype=self.dtype, shape=(9, self.nx, self.ny))
        self.fout = ti.ndarray(dtype=self.dtype, shape=(9, self.nx, self.ny))
        self.rho = ti.ndarray(dtype=self.dtype, shape=(self.nx, self.ny))
        self.ux = ti.ndarray(dtype=self.dtype, shape=(self.nx, self.ny))
        self.uy = ti.ndarray(dtype=self.dtype, shape=(self.nx, self.ny))

        # Constants
        w_np = np.array(
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
            ],
            dtype=self.np_dtype,
        )
        cx_np = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
        cy_np = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

        self.w = ti.ndarray(dtype=self.dtype, shape=(9,))
        self.cx = ti.ndarray(dtype=ti.i32, shape=(9,))
        self.cy = ti.ndarray(dtype=ti.i32, shape=(9,))

        self.w.from_numpy(w_np)
        self.cx.from_numpy(cx_np)
        self.cy.from_numpy(cy_np)

        # Initialize
        self.init_distributions()

    def init_distributions(self):
        rho_np = self.init_guassian_pulse()
        self.rho.from_numpy(rho_np)
        # ux, uy are zero already
        init_kernel(
            self.rho, self.ux, self.uy, self.fin, self.w, self.cx, self.cy, self.dtype
        )

    def step(self):
        lbm_step_kernel(
            self.fin,
            self.fout,
            self.rho,
            self.ux,
            self.uy,
            self.w,
            self.cx,
            self.cy,
            self.omega,
            self.nx,
            self.ny,
            self.dtype,
        )
        # Swap
        self.fin, self.fout = self.fout, self.fin

        # Update macroscopic (Standardized state: time t+1)
        update_macro_kernel(
            self.fin, self.rho, self.ux, self.uy, self.cx, self.cy, self.dtype
        )

    def run(self):
        ti.sync()
        for _ in range(self.cfg.iterations):
            self.step()
        ti.sync()

    def init_guassian_pulse(self, fluct_density=1e-3):
        mshx = np.linspace(-1, 1, self.nx, dtype=self.np_dtype)
        mshy = np.linspace(-1, 1, self.ny, dtype=self.np_dtype)
        x, y = np.meshgrid(mshx, mshy)
        rho = 1 + fluct_density * np.exp(-100 * (x * x + y * y))
        return rho
