import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            rows.append(
                {
                    "np": float(raw["np"]),
                    "time_s": float(raw["time_s"]),
                    "mlups": float(raw["mlups"]),
                    "speedup": float(raw["speedup"]),
                    "efficiency": float(raw["efficiency"]),
                }
            )
    rows.sort(key=lambda row: row["np"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strong scaling metrics from CSV.")
    parser.add_argument("--csv", default="outputs/scaling_metrics.csv", help="Input CSV path")
    parser.add_argument("--outdir", default="outputs", help="Output directory for plot images")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    nps = [row["np"] for row in rows]
    times = [row["time_s"] for row in rows]
    mlups = [row["mlups"] for row in rows]
    speedup = [row["speedup"] for row in rows]
    efficiency = [row["efficiency"] for row in rows]

    # Runtime vs MPI processes
    plt.figure()
    plt.plot(nps, times, marker="o", label="Measured")
    plt.xlabel("MPI processes")
    plt.ylabel("Runtime (s)")
    plt.title("Strong Scaling: Runtime vs MPI Processes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_np.png")
    plt.close()

    # Speedup vs MPI processes
    plt.figure()
    plt.plot(nps, speedup, marker="o", label="Measured")
    plt.plot(nps, nps, linestyle="--", label="Ideal")
    plt.xlabel("MPI processes")
    plt.ylabel("Speedup")
    plt.title("Strong Scaling: Speedup vs MPI Processes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "speedup_vs_np.png")
    plt.close()

    # Efficiency vs MPI processes
    plt.figure()
    plt.plot(nps, efficiency, marker="o", label="Measured")
    plt.plot(nps, [1.0 for _ in nps], linestyle="--", label="Ideal")
    plt.xlabel("MPI processes")
    plt.ylabel("Efficiency")
    plt.title("Strong Scaling: Efficiency vs MPI Processes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "efficiency_vs_np.png")
    plt.close()

    # Optional MLUPS plot
    plt.figure()
    plt.plot(nps, mlups, marker="o")
    plt.xlabel("MPI processes")
    plt.ylabel("MLUPS")
    plt.title("Strong Scaling: MLUPS vs MPI Processes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "mlups_vs_np.png")
    plt.close()

    print(f"Wrote {outdir / 'runtime_vs_np.png'}")
    print(f"Wrote {outdir / 'speedup_vs_np.png'}")
    print(f"Wrote {outdir / 'efficiency_vs_np.png'}")
    print(f"Wrote {outdir / 'mlups_vs_np.png'}")


if __name__ == "__main__":
    main()
