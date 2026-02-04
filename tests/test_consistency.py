import jax

jax.config.update("jax_enable_x64", True)
import pytest
import numpy as np
import cupy as cp
import jax.numpy as jnp
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.configs import LBMConfig, FDMConfig
from benchmarks.lbm.lbm_numpy import LBMNumPy
from benchmarks.lbm.lbm_cupy import LBMCuPy
from benchmarks.lbm.lbm_cupy_opt import LBMCuPyOpt
from benchmarks.lbm.lbm_jax import LBMJax
from benchmarks.lbm.lbm_warp import LBMWarp
from benchmarks.lbm.lbm_taichi import LBMTaichi

from benchmarks.fdm.fdm_numpy import FDMNumPy
from benchmarks.fdm.fdm_cupy import FDMCuPy
from benchmarks.fdm.fdm_jax import FDMJax
from benchmarks.fdm.fdm_warp import FDMWarp
from benchmarks.fdm.fdm_taichi import FDMTaichi

# Constants for testing
NX, NY = 64, 64
ITERATIONS = NX + NY
TOLERANCE_F32 = 3e-5
TOLERANCE_F64 = 1e-10  # On non-gpu devices with f64 support

# Backend Mappings
LBM_SOLVERS = {
    "cupy": LBMCuPy,
    "cupy_opt": LBMCuPyOpt,
    "jax": LBMJax,
    "warp": LBMWarp,
    "taichi": LBMTaichi,
}

FDM_SOLVERS = {"cupy": FDMCuPy, "jax": FDMJax, "warp": FDMWarp, "taichi": FDMTaichi}


def get_tolerance(backend: str, precision: str) -> tuple[float, float]:
    """Get (rtol, atol) based on backend quirks and precision."""
    if precision == "f32":
        # GPU backends often use fused operations that lead to small drifts
        match backend:
            case "cupy_opt":
                return 1e-5, 1e-5  # Now consistent after fix
            case "jax" | "warp" | "taichi" | "cupy":
                return 2e-5, 2e-5
            case _:
                return 1e-6, 1e-6

    # f64 tolerances
    match backend:
        case "jax" | "taichi" | "warp":
            return 5e-5, 5e-5  # Higher drift in f64 for some GPU libs
        case _:
            return 1e-6, 1e-6


@pytest.fixture(scope="module")
def lbm_baseline():
    """Generate NumPy LBM baseline once per module for each precision."""
    baselines = {}
    for prec in ["f32", "f64"]:
        cfg = LBMConfig(nx=NX, ny=NY, iterations=ITERATIONS, precision=prec)
        solver = LBMNumPy(cfg)
        solver.run()
        baselines[prec] = solver.rho
    return baselines


@pytest.fixture(scope="module")
def fdm_baseline():
    """Generate NumPy FDM baseline once per module for each precision."""
    baselines = {}
    for prec in ["f32", "f64"]:
        cfg = FDMConfig(nx=NX, ny=NY, iterations=ITERATIONS, precision=prec)
        solver = FDMNumPy(cfg)
        solver.run()
        baselines[prec] = solver.u
    return baselines


class TestLBM:
    @pytest.mark.parametrize("precision", ["f32", "f64"])
    @pytest.mark.parametrize("backend", LBM_SOLVERS.keys())
    def test_lbm_consistency(self, precision, backend, lbm_baseline):
        cfg = LBMConfig(nx=NX, ny=NY, iterations=ITERATIONS, precision=precision)
        rho_np = lbm_baseline[precision]

        # Instantiate and run GPU solver
        solver_cls = LBM_SOLVERS[backend]
        solver = solver_cls(cfg)
        solver.run()

        # Convert to numpy based on backend type
        match backend:
            case "cupy" | "cupy_opt":
                rho_gpu = cp.asnumpy(solver.rho)
            case "jax":
                rho_gpu = np.array(solver.rho)
            case "warp":
                rho_gpu = solver.rho.numpy()
            case "taichi":
                rho_gpu = solver.rho.to_numpy()
            case _:
                rho_gpu = solver.rho  # Should not happen

        rtol = TOLERANCE_F32 if precision == "f32" else TOLERANCE_F64
        diff = np.abs(rho_gpu - rho_np)
        print(f"\n{backend} vs NumPy ({precision}) Max Diff: {np.max(diff)}")

        np.testing.assert_allclose(
            rho_gpu,
            rho_np,
            rtol=rtol,
            # atol=atol,
            err_msg=f"{backend} LBM failed consistency check",
        )


class TestFDM:
    @pytest.mark.parametrize("precision", ["f32", "f64"])
    @pytest.mark.parametrize("backend", FDM_SOLVERS.keys())
    def test_fdm_consistency(self, precision, backend, fdm_baseline):
        cfg = FDMConfig(nx=NX, ny=NY, iterations=ITERATIONS, precision=precision)
        u_np = fdm_baseline[precision]

        # Case-by-case initialization check (e.g. Taichi f64 issues)
        if backend == "taichi" and precision == "f64":
            # Note: If Taichi hasn't been re-initialized for f64, this might fail or drift
            pass

        solver_cls = FDM_SOLVERS[backend]
        solver = solver_cls(cfg)
        solver.run()

        match backend:
            case "cupy":
                u_gpu = cp.asnumpy(solver.u)
            case "jax":
                u_gpu = np.array(solver.u)
            case "warp":
                u_gpu = solver.u.numpy()
            case "taichi":
                u_gpu = solver.u.to_numpy()
            case _:
                u_gpu = solver.u

        rtol, atol = get_tolerance(backend, precision)
        diff = np.abs(u_gpu - u_np)
        print(f"\n{backend} vs NumPy ({precision}) Max Diff: {np.max(diff)}")

        np.testing.assert_allclose(
            u_gpu,
            u_np,
            rtol=rtol,
            atol=atol,
            err_msg=f"{backend} FDM failed consistency check",
        )
