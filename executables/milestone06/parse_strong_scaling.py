import argparse
import csv
import re
from pathlib import Path


def parse_nprocs(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one process count must be provided")
    return values


def extract_runtime_seconds(log_text: str) -> float:
    match = re.search(r"Total runtime \(s\):\s*([0-9eE+\-.]+)", log_text)
    if not match:
        raise RuntimeError("Could not find 'Total runtime (s):' in log")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse strong scaling logs and compute MLUPS/speedup/efficiency.")
    parser.add_argument("--log-dir", default="outputs", help="Directory containing run_np*.log files")
    parser.add_argument("--nprocs", default="1,2,4,8,16,32", help="Comma-separated MPI process counts")
    parser.add_argument("--nx", type=int, default=2000, help="Fixed grid size in x")
    parser.add_argument("--ny", type=int, default=2000, help="Fixed grid size in y")
    parser.add_argument("--steps", type=int, default=2000, help="Fixed number of time steps")
    parser.add_argument("--output-csv", default="outputs/scaling_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    nprocs = parse_nprocs(args.nprocs)

    runtimes: dict[int, float] = {}
    for np_value in nprocs:
        log_path = log_dir / f"run_np{np_value}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")
        runtime = extract_runtime_seconds(log_path.read_text(encoding="utf-8", errors="replace"))
        runtimes[np_value] = runtime

    if 1 not in runtimes:
        raise RuntimeError("np=1 runtime is required to compute speedup and efficiency")

    t1 = runtimes[1]
    updates = float(args.nx) * float(args.ny) * float(args.steps)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["np", "time_s", "mlups", "speedup", "efficiency"])
        for np_value in sorted(nprocs):
            time_s = runtimes[np_value]
            mlups = updates / (time_s * 1.0e6)
            speedup = t1 / time_s
            efficiency = speedup / float(np_value)
            writer.writerow([np_value, time_s, mlups, speedup, efficiency])

    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
