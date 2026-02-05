# BenchGPUPy: Python GPU Benchmarking Suite

A comprehensive benchmarking suite to compare the performance of Python libraries (NumPy, CuPy, JAX, Warp) for scientific computing tasks: Lattice Boltzmann Method (LBM) and Finite Difference Method (FDM).

## Requirements
- Python 3.10+
- CUDA Toolkit 12.x
- Dependencies: `pip install -r requirements.txt`

## Usage

Run benchmarks using `main.py`:

```bash
python main.py --benchmark lbm --backend cupy --nx 1024 --ny 1024 --iterations 2000 --output-json result.json
```

### Options
- `--benchmark`: `lbm` or `fdm`
- `--backend`: `numpy`, `cupy`, `cupy_opt`, `jax`, `warp`, `taichi`
- `--nx`, `--ny`: Grid dimensions (default: 512)
- `--iterations`: Number of simulation steps (default: 1000)
- `--precision`: `f32` or `f64` (default: `f64`)
- `--output-json`: (Optional) Path to save results in JSON format for scaling analysis.

## Automated Benchmarking

Use the provided shell script to automate performance measurements across all backends:

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh [NX] [NY] [ITERATIONS] [PRECISION]
```

> [!TIP]
> The script is flexible with its 3rd and 4th arguments. You can pass a precision string (e.g., `f32`) as the third argument to use default iterations, or specify both:
> - `./run_benchmarks.sh 512 512 f32` (Uses 1024 iterations at f32)
> - `./run_benchmarks.sh 128 128 100 f64` (Uses 100 iterations at f64)

### Structured Results
The script generates a session-based directory in `results/bench_YYYYMMDD_HHMMSS/` containing:
- Individual `.json` files for each backend/benchmark combination.
- Verbose `.log` files for each run (capturing backend-specific initialization and errors).
- A consolidated `summary.json` containing an array of all results, ideal for scaling analysis and automated plotting.

Example:
```bash
./run_benchmarks.sh 1024 1024 2000 f64
```

## Backend Consistency
All backends (NumPy, CuPy, JAX, Warp, Taichi) are standardized to update macroscopic variables (`rho`, `ux`, `uy`) at the end of each simulation step ($t+1$), ensuring numerical consistency when comparing results across different implementations.

## Profiling with NVIDIA Nsight Systems
For deep performance analysis, use `nsys`:
```bash
nsys profile -o profile_output python main.py --benchmark lbm --backend warp
```
The codebase includes NVTX ranges to help identify specific sections in the timeline.
