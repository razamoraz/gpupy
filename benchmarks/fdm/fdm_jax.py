import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from core.configs import FDMConfig

class FDMJax:
    def __init__(self, config: FDMConfig):
        if config.precision == 'f64':
             jax.config.update("jax_enable_x64", True)
        else:
             jax.config.update("jax_enable_x64", False)
             
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.dx
        self.dt = config.dt
        self.alpha = config.alpha
        
        # Initial condition
        cx, cy = self.nx // 2, self.ny // 2
        r = min(self.nx, self.ny) // 10
        y, x = np.ogrid[-cx:self.nx-cx, -cy:self.ny-cy]
        mask = x*x + y*y <= r*r
        u_init = np.zeros((self.nx, self.ny))
        u_init[mask] = 1.0
        self.u = jnp.asarray(u_init)

    @staticmethod
    @jit
    def step_fn(u, params):
        alpha, dt, dx = params
        
        laplacian = (
            jnp.roll(u, -1, axis=0) + 
            jnp.roll(u, 1, axis=0) + 
            jnp.roll(u, -1, axis=1) + 
            jnp.roll(u, 1, axis=1) - 
            4.0*u
        ) / (dx**2)
        
        u_new = u + alpha * dt * laplacian
        return u_new, None

    def run(self):
        params = (self.alpha, self.dt, self.dx)
        
        # JAX scan requires a carry and return
        def scan_body(carry, _):
            new_u, _ = FDMJax.step_fn(carry, params)
            return new_u, None

        final_u, _ = lax.scan(scan_body, self.u, None, length=self.cfg.iterations)
        final_u.block_until_ready()
        self.u = final_u

    def step(self):
        params = (self.alpha, self.dt, self.dx)
        u_new, _ = self.step_fn(self.u, params)
        self.u = u_new.block_until_ready()
