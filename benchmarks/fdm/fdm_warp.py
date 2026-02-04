import warp as wp
import numpy as np
from core.configs import FDMConfig

@wp.kernel
def fdm_step_kernel_f32(
    u: wp.array(dtype=wp.float32, ndim=2),
    u_new: wp.array(dtype=wp.float32, ndim=2),
    alpha: wp.float32,
    dt: wp.float32,
    dx: wp.float32,
    nx: int,
    ny: int
):
    x, y = wp.tid()
    x_prev = (x - 1 + nx) % nx
    x_next = (x + 1 + nx) % nx
    y_prev = (y - 1 + ny) % ny
    y_next = (y + 1 + ny) % ny
    
    center = u[x, y]
    up = u[x, y_next]
    down = u[x, y_prev]
    left = u[x_prev, y]
    right = u[x_next, y]
    
    laplacian = (up + down + left + right - 4.0*center) / (dx*dx)
    u_new[x, y] = center + alpha * dt * laplacian

@wp.kernel
def fdm_step_kernel_f64(
    u: wp.array(dtype=wp.float64, ndim=2),
    u_new: wp.array(dtype=wp.float64, ndim=2),
    alpha: wp.float64,
    dt: wp.float64,
    dx: wp.float64,
    nx: int,
    ny: int
):
    x, y = wp.tid()
    x_prev = (x - 1 + nx) % nx
    x_next = (x + 1 + nx) % nx
    y_prev = (y - 1 + ny) % ny
    y_next = (y + 1 + ny) % ny
    
    center = u[x, y]
    up = u[x, y_next]
    down = u[x, y_prev]
    left = u[x_prev, y]
    right = u[x_next, y]
    
    laplacian = (up + down + left + right - wp.float64(4.0)*center) / (dx*dx)
    u_new[x, y] = center + alpha * dt * laplacian

class FDMWarp:
    def __init__(self, config: FDMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        
        if config.precision == 'f32':
             self.dtype = wp.float32
             self.np_dtype = np.float32
             self.kernel = fdm_step_kernel_f32
        else:
             self.dtype = wp.float64
             self.np_dtype = np.float64
             self.kernel = fdm_step_kernel_f64
        
        self.device = "cuda"
        
        self.u = wp.zeros((self.nx, self.ny), dtype=self.dtype, device=self.device)
        self.u_new = wp.zeros((self.nx, self.ny), dtype=self.dtype, device=self.device)
        
        # Init
        cx, cy = self.nx // 2, self.ny // 2
        r = min(self.nx, self.ny) // 10
        y, x = np.ogrid[-cx:self.nx-cx, -cy:self.ny-cy]
        mask = x*x + y*y <= r*r
        u_init = np.zeros((self.nx, self.ny), dtype=self.np_dtype)
        u_init[mask] = 1.0
        
        wp.copy(self.u, wp.from_numpy(u_init, device=self.device))
        
    def step(self):
        wp.launch(
            kernel=self.kernel,
            dim=(self.nx, self.ny),
            inputs=[self.u, self.u_new, self.cfg.alpha, self.cfg.dt, self.cfg.dx, self.nx, self.ny],
            device=self.device
        )
        self.u, self.u_new = self.u_new, self.u

    def run(self):
        wp.synchronize()
        for _ in range(self.cfg.iterations):
            self.step()
        wp.synchronize()
