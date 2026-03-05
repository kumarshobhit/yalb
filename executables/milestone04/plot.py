import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def step_from_name(path: Path) -> str:
    return path.stem.split("_")[-1]


def plot_decay(decay_file: Path, plots_dir: Path) -> None:
    decay = np.genfromtxt(decay_file, delimiter=",", names=True)

    plt.figure()
    plt.plot(decay["time"], decay["a_sim"], label="Simulated amplitude")
    plt.plot(decay["time"], decay["a_theory"], label="Analytical amplitude")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Shear-Wave Decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "shear_decay.png")
    plt.close()

    plt.figure()
    plt.plot(decay["time"], decay["relative_error"])
    plt.xlabel("Time")
    plt.ylabel("Relative error")
    plt.title("Shear-Wave Relative Error")
    plt.tight_layout()
    plt.savefig(plots_dir / "relative_error.png")
    plt.close()


def plot_velocity_profiles(input_dir: Path, plots_dir: Path, reference_amplitude: float) -> None:
    ux_files = sorted(Path(path) for path in glob.glob(str(input_dir / "ux_step_*.csv")))
    if not ux_files:
        return

    max_seen = 0.0
    profiles: list[tuple[str, np.ndarray]] = []

    for ux_path in ux_files:
        ux = load_csv(ux_path)
        profile = ux.mean(axis=1)
        step = step_from_name(ux_path)
        profiles.append((step, profile))
        max_seen = max(max_seen, float(np.max(np.abs(profile))))

    y = np.arange(profiles[0][1].shape[0])
    y_limit = 1.1 * max(reference_amplitude, max_seen, 1e-12)

    for step, profile in profiles:
        plt.figure()
        plt.plot(y, profile)
        plt.xlabel("y")
        plt.ylabel("u_x averaged over x")
        plt.title(f"Velocity profile step {step}")
        plt.ylim(-y_limit, y_limit)
        plt.tight_layout()
        plt.savefig(plots_dir / f"velocity_profile_step_{step}.png")
        plt.close()

    plt.figure()
    for step, profile in profiles:
        plt.plot(y, profile, alpha=0.6, label=f"step {step}")
    plt.xlabel("y")
    plt.ylabel("u_x averaged over x")
    plt.title("Velocity profile overlay")
    plt.ylim(-y_limit, y_limit)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "velocity_profile_overlay.png")
    plt.close()

    # Keep legacy filenames for compatibility with existing references.
    plt.figure()
    for step, profile in profiles:
        plt.plot(y, profile, alpha=0.6, label=f"step {step}")
    plt.xlabel("y")
    plt.ylabel("u_x averaged over x")
    plt.title("Shear profile overlay")
    plt.ylim(-y_limit, y_limit)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "profile_overlay.png")
    plt.close()


def plot_density_profiles(input_dir: Path, plots_dir: Path) -> None:
    rho_files = sorted(Path(path) for path in glob.glob(str(input_dir / "rho_step_*.csv")))
    if not rho_files:
        return

    min_seen = float("inf")
    max_seen = float("-inf")
    profiles: list[tuple[str, np.ndarray]] = []

    for rho_path in rho_files:
        rho = load_csv(rho_path)
        profile = rho.mean(axis=1)
        step = step_from_name(rho_path)
        profiles.append((step, profile))
        min_seen = min(min_seen, float(np.min(profile)))
        max_seen = max(max_seen, float(np.max(profile)))

    if min_seen == max_seen:
        padding = max(1e-6, 0.05 * abs(min_seen))
        y_min = min_seen - padding
        y_max = max_seen + padding
    else:
        padding = 0.1 * (max_seen - min_seen)
        y_min = min_seen - padding
        y_max = max_seen + padding

    y = np.arange(profiles[0][1].shape[0])
    for step, profile in profiles:
        plt.figure()
        plt.plot(y, profile)
        plt.xlabel("y")
        plt.ylabel("rho averaged over x")
        plt.title(f"Density profile step {step}")
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig(plots_dir / f"density_profile_step_{step}.png")
        plt.close()

    plt.figure()
    for step, profile in profiles:
        plt.plot(y, profile, alpha=0.6, label=f"step {step}")
    plt.xlabel("y")
    plt.ylabel("rho averaged over x")
    plt.title("Density profile overlay")
    plt.ylim(y_min, y_max)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "density_profile_overlay.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs_m4", help="Directory containing Milestone 4 outputs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"No input directory found: {input_dir}")

    decay_file = input_dir / "shear_decay.csv"
    if not decay_file.exists():
        raise SystemExit(f"No shear_decay.csv found in {input_dir}")

    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    decay = np.genfromtxt(decay_file, delimiter=",", names=True)
    reference_amplitude = float(np.max(np.abs(decay["a_sim"])))

    plot_decay(decay_file, plots_dir)
    plot_velocity_profiles(input_dir, plots_dir, reference_amplitude)
    plot_density_profiles(input_dir, plots_dir)
    print(f"Generated plots in {plots_dir}")


if __name__ == "__main__":
    main()
