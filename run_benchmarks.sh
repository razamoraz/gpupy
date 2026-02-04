#!/bin/bash

# Configuration
NX=${1:-512}
NY=${2:-512}
ITERATIONS=${3:-1024}
PRECISION=${4:-"f64"}
PYTHON_EXEC="./.venv/bin/python3"
LOG_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).log"

# Benchmark arrays
BENCHMARKS=("lbm" "fdm")
LBM_BACKENDS=("numpy" "cupy" "cupy_opt" "jax" "warp" "taichi")
FDM_BACKENDS=("numpy" "cupy" "jax" "warp" "taichi")

echo "====================================================" | tee -a "$LOG_FILE"
echo "GPU Benchmarking Suite - Automated Profiling" | tee -a "$LOG_FILE"
echo "Config: Grid=${NX}x${NY}, Iterations=${ITERATIONS}, Precision=${PRECISION}" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "====================================================" | tee -a "$LOG_FILE"

run_bench() {
    local bench=$1
    local backend=$2
    echo "----------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running: Benchmark=$bench | Backend=$backend" | tee -a "$LOG_FILE"
    
    # Run the benchmark and capture output
    $PYTHON_EXEC main.py --benchmark "$bench" --backend "$backend" --nx "$NX" --ny "$NY" --iterations "$ITERATIONS" --precision "$PRECISION" 2>&1 | tee -a "$LOG_FILE"
}

# Run LBM Benchmarks
echo -e "\n>>> Starting LBM Benchmarks" | tee -a "$LOG_FILE"
for backend in "${LBM_BACKENDS[@]}"; do
    run_bench "lbm" "$backend"
done

# Run FDM Benchmarks
echo -e "\n>>> Starting FDM Benchmarks" | tee -a "$LOG_FILE"
for backend in "${FDM_BACKENDS[@]}"; do
    run_bench "fdm" "$backend"
done

echo -e "\n====================================================" | tee -a "$LOG_FILE"
echo "Benchmarking Complete. Results saved to $LOG_FILE" | tee -a "$LOG_FILE"
echo "====================================================" | tee -a "$LOG_FILE"
