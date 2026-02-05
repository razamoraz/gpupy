import json
import glob
import os
import argparse
import matplotlib.pyplot as plt


def load_benchmarks(directory="results"):
    """
    Search for all benchmark JSON files in the specified directory and aggregate the results.
    """
    all_results = []
    pattern = os.path.join(directory, "**/*.json")

    found_summary_dirs = set()
    for summary_path in glob.glob(
        os.path.join(directory, "**/summary.json"), recursive=True
    ):
        found_summary_dirs.add(os.path.dirname(summary_path))
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
        except Exception as e:
            print(f"Error loading summary {summary_path}: {e}")

    for result_path in glob.glob(pattern, recursive=True):
        dir_name = os.path.dirname(result_path)
        if dir_name in found_summary_dirs:
            continue
        if os.path.basename(result_path) == "summary.json":
            continue

        try:
            with open(result_path, "r") as f:
                data = json.load(f)
                if (
                    isinstance(data, dict)
                    and "backend" in data
                    and "performance_metric" in data
                ):
                    all_results.append(data)
        except Exception:
            continue

    for res in all_results:
        if "nx" in res and "ny" in res:
            res["total_cells"] = res["nx"] * res["ny"]

    return all_results


def filter_results(results, benchmark=None, backend=None, precision=None):
    filtered = results
    if benchmark:
        filtered = [r for r in filtered if r.get("benchmark") == benchmark]
    if backend:
        filtered = [r for r in filtered if r.get("backend") == backend]
    if precision:
        filtered = [r for r in filtered if r.get("precision") == precision]
    return filtered


def plot_scaling_robust(
    results, benchmark_name, unit, output_path=None, precision=None
):
    bench_results = filter_results(
        results, benchmark=benchmark_name, precision=precision
    )
    if not bench_results:
        print(f"No results found for benchmark: {benchmark_name}")
        return

    plt.figure(figsize=(12, 7))
    groups = {}
    for r in bench_results:
        key = (r["backend"], r.get("precision", "unknown"))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (backend, prec), data in sorted(groups.items()):
        data = sorted(data, key=lambda x: x["total_cells"])
        sizes = sorted(list(set(r["total_cells"] for r in data)))
        x_points = [r["total_cells"] for r in data]
        y_points = [r["performance_metric"] for r in data]

        x_mean = []
        y_mean = []
        for s in sizes:
            matches = [r["performance_metric"] for r in data if r["total_cells"] == s]
            x_mean.append(s)
            y_mean.append(sum(matches) / len(matches))

        label = f"{backend} ({prec})"
        marker = "o" if prec == "f32" else "s"
        line = plt.plot(x_mean, y_mean, marker + "-", label=label, alpha=0.8)
        plt.scatter(x_points, y_points, color=line[0].get_color(), alpha=0.5, s=20)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Total Cells (NX * NY)")
    plt.ylabel(f"Performance ({unit})")
    title = f"Scaling Analysis: {benchmark_name.upper()}"
    if precision:
        title += f" ({precision})"
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        base, ext = os.path.splitext(output_path)
        if not ext:
            ext = ".png"
        out = f"{base}_{benchmark_name}{ext}"
        plt.savefig(out)
        print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Benchmarking Scaling Analysis CLI")
    parser.add_argument(
        "--dir", type=str, default="results", help="Directory containing JSON results"
    )
    parser.add_argument("--output", type=str, help="Output image filename (prefix)")
    parser.add_argument(
        "--precision", type=str, choices=["f32", "f64"], help="Filter by precision"
    )

    args = parser.parse_args()

    results = load_benchmarks(args.dir)
    print(f"Loaded {len(results)} benchmark entries from {args.dir}")

    if not results:
        return

    # Generate plots for both LBM and FDM if data exists
    plot_scaling_robust(
        results, "lbm", "MLUPS", output_path=args.output, precision=args.precision
    )
    plot_scaling_robust(
        results, "fdm", "Mpts/s", output_path=args.output, precision=args.precision
    )


if __name__ == "__main__":
    main()
