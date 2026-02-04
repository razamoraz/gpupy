import warp as wp
import numpy as np
from core.configs import LBMConfig

wp.init()


@wp.kernel
def init_kernel_f32(
    rho: wp.array(dtype=wp.float32, ndim=2),
    ux: wp.array(dtype=wp.float32, ndim=2),
    uy: wp.array(dtype=wp.float32, ndim=2),
    fin: wp.array(dtype=wp.float32, ndim=3),
    w: wp.array(dtype=wp.float32, ndim=1),
    cx: wp.array(dtype=int, ndim=1),
    cy: wp.array(dtype=int, ndim=1),
):
    x, y = wp.tid()
    rho_val = rho[x, y]
    ux_val = ux[x, y]
    uy_val = uy[x, y]
    u2 = ux_val * ux_val + uy_val * uy_val
    for i in range(9):
        cu = wp.float32(cx[i]) * ux_val + wp.float32(cy[i]) * uy_val
        fin[i, x, y] = rho_val * w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


@wp.kernel
def init_kernel_f64(
    rho: wp.array(dtype=wp.float64, ndim=2),
    ux: wp.array(dtype=wp.float64, ndim=2),
    uy: wp.array(dtype=wp.float64, ndim=2),
    fin: wp.array(dtype=wp.float64, ndim=3),
    w: wp.array(dtype=wp.float64, ndim=1),
    cx: wp.array(dtype=int, ndim=1),
    cy: wp.array(dtype=int, ndim=1),
):
    x, y = wp.tid()
    rho_val = rho[x, y]
    ux_val = ux[x, y]
    uy_val = uy[x, y]
    u2 = ux_val * ux_val + uy_val * uy_val
    for i in range(9):
        cu = wp.float64(cx[i]) * ux_val + wp.float64(cy[i]) * uy_val
        fin[i, x, y] = (
            rho_val
            * w[i]
            * (
                wp.float64(1.0)
                + wp.float64(3.0) * cu
                + wp.float64(4.5) * cu * cu
                - wp.float64(1.5) * u2
            )
        )


@wp.kernel
def update_macro_kernel_f32(
    fin: wp.array(dtype=wp.float32, ndim=3),
    rho: wp.array(dtype=wp.float32, ndim=2),
    ux: wp.array(dtype=wp.float32, ndim=2),
    uy: wp.array(dtype=wp.float32, ndim=2),
    cx: wp.array(dtype=int, ndim=1),
    cy: wp.array(dtype=int, ndim=1),
):
    x, y = wp.tid()
    rho_val = wp.float32(0.0)
    ux_val = wp.float32(0.0)
    uy_val = wp.float32(0.0)
    for i in range(9):
        f = fin[i, x, y]
        rho_val += f
        ux_val += f * wp.float32(cx[i])
        uy_val += f * wp.float32(cy[i])
    rho[x, y] = rho_val
    ux[x, y] = ux_val / rho_val
    uy[x, y] = uy_val / rho_val


@wp.kernel
def update_macro_kernel_f64(
    fin: wp.array(dtype=wp.float64, ndim=3),
    rho: wp.array(dtype=wp.float64, ndim=2),
    ux: wp.array(dtype=wp.float64, ndim=2),
    uy: wp.array(dtype=wp.float64, ndim=2),
    cx: wp.array(dtype=int, ndim=1),
    cy: wp.array(dtype=int, ndim=1),
):
    x, y = wp.tid()
    rho_val = wp.float64(0.0)
    ux_val = wp.float64(0.0)
    uy_val = wp.float64(0.0)
    for i in range(9):
        f = fin[i, x, y]
        rho_val += f
        ux_val += f * wp.float64(cx[i])
        uy_val += f * wp.float64(cy[i])
    rho[x, y] = rho_val
    ux[x, y] = ux_val / rho_val
    uy[x, y] = uy_val / rho_val


@wp.kernel
def lbm_step_kernel_f32(
    fin: wp.array(dtype=wp.float32, ndim=3),
    fout: wp.array(dtype=wp.float32, ndim=3),
    rho: wp.array(dtype=wp.float32, ndim=2),
    ux: wp.array(dtype=wp.float32, ndim=2),
    uy: wp.array(dtype=wp.float32, ndim=2),
    w: wp.array(dtype=wp.float32, ndim=1),
    cx: wp.array(dtype=int, ndim=1),
    cy: wp.array(dtype=int, ndim=1),
    omega: wp.float32,
    nx: int,
    ny: int,
):
    x, y = wp.tid()
    rho_val = wp.float32(0.0)
    ux_val = wp.float32(0.0)
    uy_val = wp.float32(0.0)
    for i in range(9):
        f = fin[i, x, y]
        rho_val += f
        ux_val += f * wp.float32(cx[i])
        uy_val += f * wp.float32(cy[i])
    ux_val = ux_val / rho_val
    uy_val = uy_val / rho_val
    u2 = ux_val * ux_val + uy_val * uy_val
    for i in range(9):
        cu = wp.float32(cx[i]) * ux_val + wp.float32(cy[i]) * uy_val
        feq = rho_val * w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        f_val = fin[i, x, y]
        f_post = f_val - omega * (f_val - feq)
        nx_coord = (x + cx[i] + nx) % nx
        ny_coord = (y + cy[i] + ny) % ny
        fout[i, nx_coord, ny_coord] = f_post


@wp.kernel
def lbm_step_kernel_f64(
    fin: wp.array(dtype=wp.float64, ndim=3),
    fout: wp.array(dtype=wp.float64, ndim=3),
    rho: wp.array(dtype=wp.float64, ndim=2),
    ux: wp.array(dtype=wp.float64, ndim=2),
    uy: wp.array(dtype=wp.float64, ndim=2),
    w: wp.array(dtype=wp.float64, ndim=1),
    cx: wp.array(dtype=int, ndim=1),
    cy: wp.array(dtype=int, ndim=1),
    omega: wp.float64,
    nx: int,
    ny: int,
):
    x, y = wp.tid()
    rho_val = wp.float64(0.0)
    ux_val = wp.float64(0.0)
    uy_val = wp.float64(0.0)
    for i in range(9):
        f = fin[i, x, y]
        rho_val += f
        ux_val += f * wp.float64(cx[i])
        uy_val += f * wp.float64(cy[i])
    ux_val = ux_val / rho_val
    uy_val = uy_val / rho_val
    u2 = ux_val * ux_val + uy_val * uy_val
    for i in range(9):
        cu = wp.float64(cx[i]) * ux_val + wp.float64(cy[i]) * uy_val
        feq = (
            rho_val
            * w[i]
            * (
                wp.float64(1.0)
                + wp.float64(3.0) * cu
                + wp.float64(4.5) * cu * cu
                - wp.float64(1.5) * u2
            )
        )
        f_val = fin[i, x, y]
        f_post = f_val - omega * (f_val - feq)
        nx_coord = (x + cx[i] + nx) % nx
        ny_coord = (y + cy[i] + ny) % ny
        fout[i, nx_coord, ny_coord] = f_post


class LBMWarp:
    def __init__(self, config: LBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.omega = config.omega

        self.dtype = wp.float32 if config.precision == "f32" else wp.float64
        self.np_dtype = np.float32 if config.precision == "f32" else np.float64

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
            ],
            dtype=self.np_dtype,
        )
        self.cx_np = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
        self.cy_np = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

        self.device = "cuda"

        self.w = wp.from_numpy(self.w_np, device=self.device)
        self.cx = wp.from_numpy(self.cx_np, device=self.device)
        self.cy = wp.from_numpy(self.cy_np, device=self.device)

        self.fin = wp.zeros((9, self.nx, self.ny), dtype=self.dtype, device=self.device)
        self.fout = wp.zeros(
            (9, self.nx, self.ny), dtype=self.dtype, device=self.device
        )
        self.rho = wp.zeros((self.nx, self.ny), dtype=self.dtype, device=self.device)
        self.ux = wp.zeros((self.nx, self.ny), dtype=self.dtype, device=self.device)
        self.uy = wp.zeros((self.nx, self.ny), dtype=self.dtype, device=self.device)

        if config.precision == "f32":
            self.init_kern = init_kernel_f32
            self.step_kern = lbm_step_kernel_f32
            self.macro_kern = update_macro_kernel_f32
        else:
            self.init_kern = init_kernel_f64
            self.step_kern = lbm_step_kernel_f64
            self.macro_kern = update_macro_kernel_f64

        # Init
        # np.random.seed(42)
        # h_rho = 1.0 + 0.01 * np.random.randn(self.nx, self.ny).astype(self.np_dtype)
        wp.copy(self.rho, wp.from_numpy(self.init_guassian_pulse(), device=self.device))

        wp.launch(
            kernel=self.init_kern,
            dim=(self.nx, self.ny),
            inputs=[self.rho, self.ux, self.uy, self.fin, self.w, self.cx, self.cy],
            device=self.device,
        )

        # Swap buffers logic: fin is current, fout is next

    def step(self):
        wp.launch(
            kernel=self.step_kern,
            dim=(self.nx, self.ny),
            inputs=[
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
            ],
            device=self.device,
        )
        # Swap
        self.fin, self.fout = self.fout, self.fin

        # Update macroscopic (Standardized state: time t+1)
        wp.launch(
            kernel=self.macro_kern,
            dim=(self.nx, self.ny),
            inputs=[self.fin, self.rho, self.ux, self.uy, self.cx, self.cy],
            device=self.device,
        )

    def run(self):
        wp.synchronize()
        for _ in range(self.cfg.iterations):
            self.step()
        wp.synchronize()

    def init_guassian_pulse(self, fluct_density=1e-3):
        mshx = np.linspace(-1, 1, self.nx, dtype=self.np_dtype)
        mshy = np.linspace(-1, 1, self.ny, dtype=self.np_dtype)
        x, y = np.meshgrid(mshx, mshy)
        rho = 1 + fluct_density * np.exp(-100 * (x * x + y * y))
        return rho
