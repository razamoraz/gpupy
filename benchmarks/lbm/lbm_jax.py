import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from core.configs import LBMConfig

# Enable 64bit by default at the top


class LBMJax:
    def __init__(self, config: LBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.omega = config.omega
        self.dtype = jnp.float64 if config.precision == "f64" else jnp.float32

        self.w = jnp.array(
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
            dtype=self.dtype,
        )

        self.cx = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=jnp.int32)
        self.cy = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=jnp.int32)

        # Initial condition
        np.random.seed(42)
        # h_rho = 1.0 + 0.01 * np.random.randn(self.nx, self.ny)
        self._rho = jnp.asarray(self.init_guassian_pulse(), dtype=self.dtype)
        self.ux = jnp.zeros((self.nx, self.ny), dtype=self.dtype)
        self.uy = jnp.zeros((self.nx, self.ny), dtype=self.dtype)

        # Compute Initial Equilibrium
        u2 = self.ux**2 + self.uy**2
        self.fin = jnp.zeros((9, self.nx, self.ny), dtype=self.dtype)

        def init_eq_step(i, fin):
            cu = self.cx[i] * self.ux + self.cy[i] * self.uy
            val = self._rho * self.w[i] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
            return fin.at[i, :, :].set(val)

        self.fin = lax.fori_loop(0, 9, init_eq_step, self.fin)

    @staticmethod
    @jit
    def step_fn(carry, _):
        fin, w, cx, cy, omega, rho, ux, uy = carry

        # Equilibrium
        u2 = ux**2 + uy**2

        feq = jnp.zeros_like(fin)
        for i in range(9):
            cu = cx[i] * ux + cy[i] * uy
            val = rho * w[i] * (1.0 + 3.0 * cu + 4.5 * (cu**2) - 1.5 * u2)
            feq = feq.at[i].set(val)

        # Collision
        fout = fin - omega * (fin - feq)

        # Streaming using roll
        # We need static shifts for jnp.roll.
        CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
        CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]

        fin_next = jnp.zeros_like(fin)
        for i in range(9):
            fin_next = fin_next.at[i].set(
                jnp.roll(fout[i], (CX[i], CY[i]), axis=(0, 1))
            )

        # Update Macroscopic for the next step (Standardized state: time t+1)
        rho_next = jnp.sum(fin_next, axis=0)
        ux_next = jnp.sum(fin_next * cx[:, None, None], axis=0) / rho_next
        uy_next = jnp.sum(fin_next * cy[:, None, None], axis=0) / rho_next

        return (fin_next, w, cx, cy, omega, rho_next, ux_next, uy_next), None

    def run(self):
        # Package state
        carry = (
            self.fin,
            self.w,
            self.cx,
            self.cy,
            self.omega,
            self.rho,
            self.ux,
            self.uy,
        )

        # Run loop
        final_carry, _ = lax.scan(self.step_fn, carry, None, length=self.cfg.iterations)

        # Block until ready to ensure timing is correct
        final_fin = final_carry[0]
        final_fin.block_until_ready()
        self.fin = final_fin
        self._rho = final_carry[5]
        self.ux = final_carry[6]
        self.uy = final_carry[7]

    def step(self):
        carry = (
            self.fin,
            self.w,
            self.cx,
            self.cy,
            self.omega,
            self.rho,
            self.ux,
            self.uy,
        )
        (fin_new, _, _, _, _, rho_new, ux_new, uy_new), _ = self.step_fn(carry, None)
        self.fin = fin_new.block_until_ready()
        self._rho = rho_new
        self.ux = ux_new
        self.uy = uy_new

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value

    def init_guassian_pulse(self, fluct_density=1e-3):
        dtype_np = np.float64 if self.cfg.precision == "f64" else np.float32
        mshx = np.linspace(-1, 1, self.nx, dtype=dtype_np)
        mshy = np.linspace(-1, 1, self.ny, dtype=dtype_np)
        x, y = np.meshgrid(mshx, mshy)
        rho = 1 + fluct_density * np.exp(-100 * (x * x + y * y))
        return rho
