import argparse
import csv
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_nprocs(text: str) -> list[int]:
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one process count must be provided")
    return values


def extract_runtime(stdout: str) -> float:
    match = re.search(r"Total runtime \(s\):\s*([0-9eE+\-.]+)", stdout)
    if not match:
        raise RuntimeError("Could not find runtime line in milestone06 output")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build_milestone", help="Meson build directory")
    parser.add_argument("--output-dir", default="build_milestone/outputs_m6_scaling", help="Output directory for scaling CSV/plots")
    parser.add_argument("--nprocs", default="1,2,4", help="Comma-separated process counts")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--lid-ux", type=float, default=0.05)
    parser.add_argument("--lid-uy", type=float, default=0.0)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exe = build_dir / "executables" / "milestone06" / "milestone06"
    if not exe.exists():
        raise SystemExit(f"milestone06 executable not found: {exe}")

    nprocs = parse_nprocs(args.nprocs)
    timings: list[tuple[int, float]] = []

    for np_value in nprocs:
        run_dir = output_dir / f"run_np{np_value}"
        cmd = [
            "mpirun",
            "-np",
            str(np_value),
            str(exe),
            "--nx",
            str(args.nx),
            "--ny",
            str(args.ny),
            "--steps",
            str(args.steps),
            "--omega",
            str(args.omega),
            "--rho0",
            str(args.rho0),
            "--lid-ux",
            str(args.lid_ux),
            "--lid-uy",
            str(args.lid_uy),
            "--write-field-every",
            str(args.steps + 1),
            "--residual-every",
            str(max(1, args.steps // 4)),
            "--output-dir",
            str(run_dir),
        ]

        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        runtime = extract_runtime(completed.stdout)
        timings.append((np_value, runtime))
        print(f"np={np_value}: time={runtime:.6f} s")

    base_time = timings[0][1]
    rows = []
    for np_value, runtime in timings:
        speedup = base_time / runtime if runtime > 0.0 else 0.0
        efficiency = speedup / float(np_value) if np_value > 0 else 0.0
        rows.append((np_value, runtime, speedup, efficiency))

    csv_path = output_dir / "scaling.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["np", "time_s", "speedup", "efficiency"])
        writer.writerows(rows)

    plt.figure()
    plt.plot([row[0] for row in rows], [row[2] for row in rows], marker="o", label="Measured")
    ideal_x = [row[0] for row in rows]
    ideal_y = [float(value) / float(ideal_x[0]) for value in ideal_x]
    plt.plot(ideal_x, ideal_y, linestyle="--", label="Ideal")
    plt.xlabel("MPI ranks")
    plt.ylabel("Speedup")
    plt.title("Milestone 06 Strong Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_vs_np.png")
    plt.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'speedup_vs_np.png'}")


if __name__ == "__main__":
    main()
