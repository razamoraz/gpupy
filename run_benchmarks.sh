#!/bin/bash

# Configuration
NX=${1:-512}
NY=${2:-512}
# Handle optional positional arguments more flexibly
ITERATIONS=1024
PRECISION="f64"

# Check if 3rd arg is precision or iterations
if [[ "$3" == "f32" || "$3" == "f64" ]]; then
    PRECISION="$3"
    ITERATIONS=${4:-1024}
elif [[ -n "$3" ]]; then
    ITERATIONS="$3"
    PRECISION=${4:-"f64"}
fi

PYTHON_EXEC="python3"

# Session setup
SESSION_ID=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/bench_${SESSION_ID}"
mkdir -p "$RESULTS_DIR"

# Benchmark arrays
BENCHMARKS=("lbm" "fdm")
LBM_BACKENDS=("numpy" "cupy" "cupy_opt" "jax" "warp" "taichi")
FDM_BACKENDS=("numpy" "cupy" "jax" "warp" "taichi")

echo "===================================================="
echo "GPU Benchmarking Suite - Structured Profiling"
echo "Config: Grid=${NX}x${NY}, Iterations=${ITERATIONS}, Precision=${PRECISION}"
echo "Session: ${SESSION_ID}"
echo "Results Dir: ${RESULTS_DIR}"
echo "===================================================="

run_bench() {
    local bench=$1
    local backend=$2
    local json_out="${RESULTS_DIR}/${bench}_${backend}.json"
    
    echo -n "Running ${bench} | ${backend} ... "
    
    # Run the benchmark and capture output to a log file, but use --output-json for metrics
    $PYTHON_EXEC main.py --benchmark "$bench" --backend "$backend" --nx "$NX" --ny "$NY" --iterations "$ITERATIONS" --precision "$PRECISION" --output-json "$json_out" > "${json_out}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "DONE"
    else
        echo "FAILED (see ${json_out}.log)"
    fi
}

# Run LBM Benchmarks
for backend in "${LBM_BACKENDS[@]}"; do
    run_bench "lbm" "$backend"
done

# Run FDM Benchmarks
for backend in "${FDM_BACKENDS[@]}"; do
    run_bench "fdm" "$backend"
done

# Consolidate results into a single summary.json
SUMMARY_FILE="${RESULTS_DIR}/summary.json"
echo "[" > "$SUMMARY_FILE"
FIRST=1
for f in "${RESULTS_DIR}"/*.json; do
    if [[ "$f" == *"summary.json" ]]; then continue; fi
    if [ $FIRST -ne 1 ]; then echo "," >> "$SUMMARY_FILE"; fi
    cat "$f" >> "$SUMMARY_FILE"
    FIRST=0
done
echo "]" >> "$SUMMARY_FILE"

echo -e "\n===================================================="
echo "Benchmark Summary (Precision: ${PRECISION})"
echo -e "Backend\t\tBenchmark\tPerformance"
echo "----------------------------------------------------"

# Extract and display results from individual JSONs (subset for display)
for f in "${RESULTS_DIR}"/*.json; do
    if [[ "$f" == *"summary.json" ]]; then continue; fi
    BACKEND=$(grep '"backend"' "$f" | cut -d'"' -f4)
    BENCH=$(grep '"benchmark"' "$f" | cut -d'"' -f4)
    PERF=$(grep '"performance_metric"' "$f" | cut -d':' -f2 | sed 's/ //g' | sed 's/,//g')
    UNIT=$(grep '"performance_unit"' "$f" | cut -d'"' -f4)
    printf "%-15s %-10s %10.2f %s\n" "$BACKEND" "$BENCH" "$PERF" "$UNIT"
done

echo "===================================================="
echo "Benchmarking Complete."
echo "Full summary: $SUMMARY_FILE"

# Auto-generate scaling plots using core/analysis.py
echo -e "\nGenerating scaling plots..."
$PYTHON_EXEC core/analysis.py --output scaling_latest

echo "===================================================="
