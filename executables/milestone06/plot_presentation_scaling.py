import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "plots_presentation"


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append({key: float(value) for key, value in raw.items()})
    rows.sort(key=lambda row: row["np"])
    return rows


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cpu = load_rows(ROOT / "outputs/scaling_metrics.csv")
    gpu_n1 = load_rows(ROOT / "outputs_gpu_n1/scaling_metrics.csv")
    gpu_same_2000 = load_rows(ROOT / "outputs_gpu_np2_same_node_fix3/scaling_metrics.csv")
    gpu_same_4000 = load_rows(ROOT / "outputs_gpu_np2_same_node_4000/scaling_metrics.csv")
    gpu_same_8000 = load_rows(ROOT / "outputs_gpu_np2_same_node_8000/scaling_metrics.csv")
    gpu_two_node_2000 = load_rows(ROOT / "outputs_gpu_np2_n2/scaling_metrics.csv")

    cpu_by_np = {int(row["np"]): row for row in cpu}
    gpu_same_2000_by_np = {int(row["np"]): row for row in gpu_same_2000}
    gpu_two_node_2000_by_np = {int(row["np"]): row for row in gpu_two_node_2000}

    # 1) Runtime bar chart for CPU vs GPU on the 2000x2000 case.
    labels = [
        "CPU np=1",
        "CPU np=2",
        "GPU np=1",
        "GPU np=2\nsame node",
        "GPU np=2\ntwo nodes",
    ]
    values = [
        cpu_by_np[1]["time_s"],
        cpu_by_np[2]["time_s"],
        gpu_n1[0]["time_s"],
        gpu_same_2000_by_np[2]["time_s"],
        gpu_two_node_2000_by_np[2]["time_s"],
    ]
    colors = ["#7f8c8d", "#95a5a6", "#2e86de", "#1f618d", "#c0392b"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("2000x2000 Runtime Comparison: CPU vs GPU")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "runtime_cpu_vs_gpu_2000.png", dpi=200)
    plt.close(fig)

    # 2) Speedup vs number of processes for CPU and GPU 2000x2000.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cpu_nps = [row["np"] for row in cpu]
    cpu_speedup = [row["speedup"] for row in cpu]
    gpu_same_nps = [row["np"] for row in gpu_same_2000]
    gpu_same_speedup = [row["speedup"] for row in gpu_same_2000]
    gpu_two_speedup = [row["speedup"] for row in gpu_two_node_2000]
    ax.plot(cpu_nps, cpu_speedup, marker="o", linewidth=2, label="CPU")
    ax.plot(gpu_same_nps, gpu_same_speedup, marker="o", linewidth=2, label="GPU same node")
    ax.plot(gpu_same_nps, gpu_two_speedup, marker="o", linewidth=2, label="GPU two nodes")
    ax.plot(cpu_nps, cpu_nps, linestyle="--", color="black", alpha=0.6, label="Ideal")
    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Number of Processes")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "speedup_vs_np_cpu_gpu_2000.png", dpi=200)
    plt.close(fig)

    # 3) Efficiency vs number of processes for CPU and GPU 2000x2000.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cpu_eff = [row["efficiency"] for row in cpu]
    gpu_same_eff = [row["efficiency"] for row in gpu_same_2000]
    gpu_two_eff = [row["efficiency"] for row in gpu_two_node_2000]
    ax.plot(cpu_nps, cpu_eff, marker="o", linewidth=2, label="CPU")
    ax.plot(gpu_same_nps, gpu_same_eff, marker="o", linewidth=2, label="GPU same node")
    ax.plot(gpu_same_nps, gpu_two_eff, marker="o", linewidth=2, label="GPU two nodes")
    ax.axhline(1.0, linestyle="--", color="black", alpha=0.6, label="Ideal")
    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency vs Number of Processes")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "efficiency_vs_np_cpu_gpu_2000.png", dpi=200)
    plt.close(fig)

    # 4) GPU same-node speedup by grid size.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for rows, label, color in [
        (gpu_same_2000, "2000x2000", "#1f77b4"),
        (gpu_same_4000, "4000x4000", "#ff7f0e"),
        (gpu_same_8000, "8000x8000", "#2ca02c"),
    ]:
        ax.plot([row["np"] for row in rows], [row["speedup"] for row in rows], marker="o", linewidth=2, label=label, color=color)
    ax.plot([1, 2], [1, 2], linestyle="--", color="black", alpha=0.6, label="Ideal")
    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Speedup")
    ax.set_title("GPU Same-Node Strong Scaling: Speedup by Grid Size")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gpu_same_node_speedup_by_grid.png", dpi=200)
    plt.close(fig)

    # 5) GPU same-node efficiency by grid size.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for rows, label, color in [
        (gpu_same_2000, "2000x2000", "#1f77b4"),
        (gpu_same_4000, "4000x4000", "#ff7f0e"),
        (gpu_same_8000, "8000x8000", "#2ca02c"),
    ]:
        ax.plot([row["np"] for row in rows], [row["efficiency"] for row in rows], marker="o", linewidth=2, label=label, color=color)
    ax.axhline(1.0, linestyle="--", color="black", alpha=0.6, label="Ideal")
    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Efficiency")
    ax.set_title("GPU Same-Node Strong Scaling: Efficiency by Grid Size")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gpu_same_node_efficiency_by_grid.png", dpi=200)
    plt.close(fig)

    # 6) GPU same-node throughput by grid size.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for rows, label, color in [
        (gpu_same_2000, "2000x2000", "#1f77b4"),
        (gpu_same_4000, "4000x4000", "#ff7f0e"),
        (gpu_same_8000, "8000x8000", "#2ca02c"),
    ]:
        ax.plot([row["np"] for row in rows], [row["mlups"] for row in rows], marker="o", linewidth=2, label=label, color=color)
    ax.set_xlabel("Number of processes")
    ax.set_ylabel("MLUPS")
    ax.set_title("GPU Same-Node Throughput: MLUPS by Grid Size")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gpu_same_node_mlups_by_grid.png", dpi=200)
    plt.close(fig)

    # 7) Direct np=2 runtime comparison, same node vs two nodes.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = ["GPU np=2\nsame node", "GPU np=2\ntwo nodes"]
    values = [gpu_same_2000_by_np[2]["time_s"], gpu_two_node_2000_by_np[2]["time_s"]]
    ax.bar(labels, values, color=["#1f618d", "#c0392b"])
    ax.set_ylabel("Runtime (s)")
    ax.set_title("2000x2000 GPU np=2: Same Node vs Two Nodes")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gpu_np2_same_vs_two_node_runtime.png", dpi=200)
    plt.close(fig)

    print(f"Wrote plots to {OUTDIR}")


if __name__ == "__main__":
    main()
