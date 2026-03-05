import argparse
import csv
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_omegas(raw: str) -> list[float]:
    omegas: list[float] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        omegas.append(float(value))
    if not omegas:
        raise ValueError("No omega values were provided.")
    return omegas


def theoretical_viscosity(omega: float) -> float:
    return (1.0 / 3.0) * (1.0 / omega - 0.5)


def fit_viscosity_from_decay(decay_file: Path, ny: int) -> tuple[float, float]:
    decay = np.genfromtxt(decay_file, delimiter=",", names=True)
    times = np.asarray(decay["time"], dtype=float)
    amplitudes = np.asarray(decay["a_sim"], dtype=float)

    valid = amplitudes > 0.0
    times = times[valid]
    amplitudes = amplitudes[valid]
    if times.size < 3:
        raise RuntimeError(f"Not enough positive amplitude points in {decay_file}")

    log_a = np.log(amplitudes)
    slope, intercept = np.polyfit(times, log_a, 1)
    predicted = slope * times + intercept

    residual = np.sum((log_a - predicted) ** 2)
    total = np.sum((log_a - np.mean(log_a)) ** 2)
    r2 = 1.0 if total == 0.0 else 1.0 - residual / total

    k = 2.0 * math.pi / float(ny)
    nu_measured = -slope / (k * k)
    return float(nu_measured), float(r2)


def run_benchmark(
    executable: Path,
    output_dir: Path,
    nx: int,
    ny: int,
    steps: int,
    omega: float,
    rho0: float,
    amplitude: float,
    write_field_every: int) -> None:
    cmd = [
        str(executable),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--steps",
        str(steps),
        "--omega",
        str(omega),
        "--rho0",
        str(rho0),
        "--amplitude",
        str(amplitude),
        "--output-dir",
        str(output_dir),
        "--write-field-every",
        str(write_field_every),
    ]
    subprocess.run(cmd, check=True)


def plot_viscosity_dependency(csv_path: Path, output_path: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    omegas = data["omega"]
    nu_measured = data["nu_measured"]
    nu_theory = data["nu_theory"]

    plt.figure()
    plt.plot(omegas, nu_theory, label="Analytical", marker="o")
    plt.plot(omegas, nu_measured, label="Measured", marker="s")
    plt.xlabel("omega")
    plt.ylabel("kinematic viscosity")
    plt.title("Measured vs analytical viscosity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_decay_curves_vs_omega(output_dir: Path, omegas: list[float], output_path: Path) -> None:
    plt.figure()
    for omega in omegas:
        decay_file = output_dir / f"omega_{omega:.3f}" / "shear_decay.csv"
        decay = np.genfromtxt(decay_file, delimiter=",", names=True)
        plt.plot(decay["time"], decay["a_sim"], label=f"Experimental decay for omega = {omega:.1f}")

    plt.xlabel("Time step / 1")
    plt.ylabel("Wave amplitude")
    plt.title("Shear wave decay for different omega")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build_milestone")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rho0", type=float, default=0.2)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--omegas", default="0.6,0.8,1.0,1.2,1.4,1.6")
    parser.add_argument("--write-field-every", type=int, default=20)
    parser.add_argument("--output-dir", default="build_milestone/outputs_m4_sweep")
    args = parser.parse_args()

    omegas = parse_omegas(args.omegas)
    build_dir = Path(args.build_dir)
    executable = build_dir / "executables" / "milestone04" / "milestone04"
    if not executable.exists():
        raise SystemExit(f"milestone04 executable not found at {executable}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "viscosity_vs_omega.csv"
    rows = []

    for omega in omegas:
        if not (0.0 < omega < 2.0):
            raise SystemExit(f"Invalid omega {omega}: expected 0 < omega < 2")

        run_output = output_dir / f"omega_{omega:.3f}"
        run_output.mkdir(parents=True, exist_ok=True)

        print(f"Running omega={omega:.3f} ...")
        run_benchmark(
            executable=executable,
            output_dir=run_output,
            nx=args.nx,
            ny=args.ny,
            steps=args.steps,
            omega=omega,
            rho0=args.rho0,
            amplitude=args.amplitude,
            write_field_every=args.write_field_every,
        )

        decay_file = run_output / "shear_decay.csv"
        nu_measured, fit_r2 = fit_viscosity_from_decay(decay_file, args.ny)
        nu_theory = theoretical_viscosity(omega)
        relative_error = abs(nu_measured - nu_theory) / max(abs(nu_theory), 1e-14)
        rows.append((omega, nu_measured, nu_theory, relative_error, fit_r2))

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "nu_measured", "nu_theory", "relative_error", "fit_r2"])
        writer.writerows(rows)

    plot_path = output_dir / "viscosity_vs_omega.png"
    plot_viscosity_dependency(csv_path, plot_path)
    decay_plot_path = output_dir / "shear_decay_vs_omega.png"
    plot_decay_curves_vs_omega(output_dir, omegas, decay_plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {decay_plot_path}")


if __name__ == "__main__":
    main()
