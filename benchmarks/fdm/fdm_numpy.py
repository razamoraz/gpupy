import numpy as np
from core.configs import FDMConfig

class FDMNumPy:
    def __init__(self, config: FDMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.dx
        self.dt = config.dt
        self.alpha = config.alpha
        
        self.dtype = np.float32 if config.precision == 'f32' else np.float64
        
        self.u = np.zeros((self.nx, self.ny), dtype=self.dtype)
        self.u_new = np.zeros((self.nx, self.ny), dtype=self.dtype)
        
        # Init
        cx, cy = self.nx // 2, self.ny // 2
        r = min(self.nx, self.ny) // 10
        y, x = np.ogrid[-cx:self.nx-cx, -cy:self.ny-cy]
        mask = x*x + y*y <= r*r
        self.u[mask] = 1.0

    def step(self):
        # Laplacian using slicing
        # u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j]
        
        u = self.u
        laplacian = (
            np.roll(u, -1, axis=0) + 
            np.roll(u, 1, axis=0) + 
            np.roll(u, -1, axis=1) + 
            np.roll(u, 1, axis=1) - 
            4*u
        ) / (self.dx**2)
        
        self.u_new = u + self.alpha * self.dt * laplacian
        self.u[:] = self.u_new[:]

    def run(self):
        for _ in range(self.cfg.iterations):
            self.step()
