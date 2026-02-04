import cupy as cp
import numpy as np
from core.configs import FDMConfig

class FDMCuPy:
    def __init__(self, config: FDMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.dx
        self.dt = config.dt
        self.alpha = config.alpha
        
        self.dtype = np.float32 if config.precision == 'f32' else np.float64
        
        self.u = cp.zeros((self.nx, self.ny), dtype=self.dtype)
        self.u_new = cp.zeros((self.nx, self.ny), dtype=self.dtype)
        
        # Initial condition
        cx, cy = self.nx // 2, self.ny // 2
        r = min(self.nx, self.ny) // 10
        y, x = np.ogrid[-cx:self.nx-cx, -cy:self.ny-cy]
        mask = x*x + y*y <= r*r
        u_init = np.zeros((self.nx, self.ny), dtype=self.dtype)
        u_init[mask] = 1.0
        self.u = cp.asarray(u_init)

    def step(self):
        u = self.u
        laplacian = (
            cp.roll(u, -1, axis=0) + 
            cp.roll(u, 1, axis=0) + 
            cp.roll(u, -1, axis=1) + 
            cp.roll(u, 1, axis=1) - 
            4*u
        ) / (self.dx**2)
        
        self.u_new = u + self.alpha * self.dt * laplacian
        self.u[:] = self.u_new[:]

    def run(self):
        cp.cuda.Stream.null.synchronize()
        for _ in range(self.cfg.iterations):
            self.step()
        cp.cuda.Stream.null.synchronize()
