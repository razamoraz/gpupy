# BenchGPUPy: Python GPU Benchmarking Suite

A comprehensive benchmarking suite to compare the performance of Python libraries (NumPy, CuPy, JAX, Warp) for scientific computing tasks: Lattice Boltzmann Method (LBM) and Finite Difference Method (FDM).

## Requirements
- Python 3.10+
- CUDA Toolkit 12.x
- Dependencies: `pip install -r requirements.txt`

## Usage

Run benchmarks using `main.py`:

```bash
python main.py --benchmark lbm --backend cupy --nx 1024 --ny 1024 --iterations 2000
```

### Options
- `--benchmark`: `lbm` or `fdm`
- `--backend`: `numpy`, `cupy`, `cupy_opt`, `jax`, `warp`, `taichi`
- `--nx`, `--ny`: Grid dimensions (default: 512)
- `--iterations`: Number of simulation steps (default: 1000)
- `--precision`: `f32` or `f64` (default: `f64`)

## Automated Benchmarking

Use the provided shell script to automate performance measurements across all backends:

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh [NX] [NY] [ITERATIONS] [PRECISION]
```

Example:
```bash
./run_benchmarks.sh 1024 1024 2000 f64
```

Results will be displayed in the terminal and saved to a timestamped log file.

## Backend Consistency
All backends (NumPy, CuPy, JAX, Warp, Taichi) are standardized to update macroscopic variables (`rho`, `ux`, `uy`) at the end of each simulation step ($t+1$), ensuring numerical consistency when comparing results across different implementations.

## Profiling with NVIDIA Nsight Systems
<truncated 527 bytes>
