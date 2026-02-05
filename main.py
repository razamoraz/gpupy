import argparse
import sys
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import json
from datetime import datetime, timezone
from core.configs import LBMConfig, FDMConfig
from core.profiler import Profiler

# Import Benchmarks
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


def run_benchmark(args):
    print(
        f"Running Benchmark: {args.benchmark} | Backend: {args.backend} | Iterations: {args.iterations}"
    )

    match args.benchmark:
        case "lbm":
            cfg = LBMConfig(
                nx=args.nx,
                ny=args.ny,
                iterations=args.iterations,
                precision=args.precision,
            )
            match args.backend:
                case "numpy":
                    solver = LBMNumPy(cfg)
                case "cupy":
                    solver = LBMCuPy(cfg)
                case "cupy_opt":
                    solver = LBMCuPyOpt(cfg)
                case "jax":
                    solver = LBMJax(cfg)
                case "warp":
                    solver = LBMWarp(cfg)
                case "taichi":
                    solver = LBMTaichi(cfg)
                case _:
                    raise ValueError(f"Unknown LBM backend: {args.backend}")

        case "fdm":
            cfg = FDMConfig(
                nx=args.nx,
                ny=args.ny,
                iterations=args.iterations,
                precision=args.precision,
            )
            match args.backend:
                case "numpy":
                    solver = FDMNumPy(cfg)
                case "cupy":
                    solver = FDMCuPy(cfg)
                case "jax":
                    solver = FDMJax(cfg)
                case "warp":
                    solver = FDMWarp(cfg)
                case "taichi":
                    solver = FDMTaichi(cfg)
                case _:
                    raise ValueError(f"Unknown FDM backend: {args.backend}")

        case _:
            raise ValueError(f"Unknown benchmark: {args.benchmark}")

    # Warmup (optional, generally good for JIT)
    print("Initializing...")
    # Some solvers init in constructor, but JIT compilation happens on first run/step usually or we can force it.
    # We will just run 1 step as warmup if it's not numpy (numpy doesn't need warmup)
    if args.backend != "numpy":
        solver.step()

    print("Starting benchmark...")

    with Profiler(f"{args.benchmark}_{args.backend}") as p:
        solver.run()

    print(f"Completed in {p.duration:.4f} seconds")

    # Calculate Metrics
    # LBM: Lattice Updates Per Second (LUPS) = (nx * ny * iterations) / time
    total_cells = args.nx * args.ny
    total_ops = total_cells * args.iterations

    if args.benchmark == "lbm":
        mlups = (total_ops / p.duration) / 1e6
        perf_value = mlups
        perf_unit = "MLUPS"
        print(f"Performance: {perf_value:.2f} {perf_unit}")
    elif args.benchmark == "fdm":
        mpts = (total_ops / p.duration) / 1e6  # Million Points Per Second
        perf_value = mpts
        perf_unit = "Mpts/s"
        print(f"Performance: {perf_value:.2f} {perf_unit}")

    # Output JSON if requested
    if args.output_json:
        results = {
            "benchmark": args.benchmark,
            "backend": args.backend,
            "nx": args.nx,
            "ny": args.ny,
            "iterations": args.iterations,
            "precision": args.precision,
            "duration": p.duration,
            "performance_metric": perf_value,
            "performance_unit": perf_unit,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_json}")


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmarking Suite")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["lbm", "fdm"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["numpy", "cupy", "cupy_opt", "jax", "warp", "taichi"],
        help="Backend to use",
    )
    parser.add_argument("--nx", type=int, default=512, help="Grid width")
    parser.add_argument("--ny", type=int, default=512, help="Grid height")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="f64",
        choices=["f32", "f64"],
        help="Precision (f32 or f64)",
    )
    parser.add_argument(
        "--output-json", type=str, help="Path to save results in JSON format"
    )

    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
