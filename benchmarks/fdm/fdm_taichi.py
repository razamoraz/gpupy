import taichi as ti
import numpy as np
from core.configs import FDMConfig

# Assuming ti.init(arch=ti.gpu) is called by lbm/main beforehand or we check?
# ti.init is global. If lbm_taichi ran, it's init. If not, we might need to lazy init?
# But typically main.py will run one benchmark. 
# Safe to call ti.init multiple times? 
# Robust init
try:
    ti.init(arch=ti.gpu)
except:
    pass

@ti.kernel
def fdm_step_kernel(u: ti.types.ndarray(), u_new: ti.types.ndarray(), 
                    alpha: float, dt: float, dx: float, nx: int, ny: int):
    for x, y in u:
        center = u[x, y]
        
        # Periodic BC manually
        x_prev = (x - 1 + nx) % nx
        x_next = (x + 1 + nx) % nx
        y_prev = (y - 1 + ny) % ny
        y_next = (y + 1 + ny) % ny
        
        up = u[x, y_next]
        down = u[x, y_prev]
        left = u[x_prev, y]
        right = u[x_next, y]
        
        laplacian = (up + down + left + right - 4.0 * center) / (dx * dx)
        u_new[x, y] = center + alpha * dt * laplacian

class FDMTaichi:
    def __init__(self, config: FDMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        
        self.dtype = ti.f32 if config.precision == 'f32' else ti.f64
        self.np_dtype = np.float32 if config.precision == 'f32' else np.float64
        
        self.u = ti.ndarray(dtype=self.dtype, shape=(self.nx, self.ny))
        self.u_new = ti.ndarray(dtype=self.dtype, shape=(self.nx, self.ny))
        
        # Init
        cx, cy = self.nx // 2, self.ny // 2
        r = min(self.nx, self.ny) // 10
        y, x = np.ogrid[-cx:self.nx-cx, -cy:self.ny-cy]
        mask = x*x + y*y <= r*r
        u_init = np.zeros((self.nx, self.ny), dtype=self.np_dtype)
        u_init[mask] = 1.0
        
        self.u.from_numpy(u_init)

    def step(self):
        fdm_step_kernel(self.u, self.u_new, self.cfg.alpha, self.cfg.dt, self.cfg.dx, self.nx, self.ny)
        self.u, self.u_new = self.u_new, self.u

    def run(self):
        ti.sync()
        for _ in range(self.cfg.iterations):
            self.step()
        ti.sync()
