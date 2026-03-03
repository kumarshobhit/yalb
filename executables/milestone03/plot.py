import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def step_from_name(path: Path) -> str:
    return path.stem.split("_")[-1]


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def plot_density(output_dir: Path, step: str, rho: np.ndarray) -> None:
    plt.figure()
    plt.imshow(rho, origin="lower", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Density")
    plt.title(f"Density step {step}")
    plt.tight_layout()
    plt.savefig(output_dir / f"rho_step_{step}.png")
    plt.close()


def plot_velocity(output_dir: Path, step: str, ux: np.ndarray, uy: np.ndarray) -> None:
    x = np.arange(ux.shape[1])
    y = np.arange(ux.shape[0])
    grid_x, grid_y = np.meshgrid(x, y)
    speed = np.sqrt(ux**2 + uy**2)

    plt.figure()
    plt.streamplot(grid_x, grid_y, ux, uy, color=speed, cmap="plasma", density=1.1)
    plt.colorbar(label="Speed")
    plt.xlim(0, ux.shape[1] - 1)
    plt.ylim(0, ux.shape[0] - 1)
    plt.title(f"Velocity step {step}")
    plt.tight_layout()
    plt.savefig(output_dir / f"velocity_step_{step}.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs_m3", help="Directory containing rho/ux/uy CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"No input directory found: {input_dir}")

    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rho_files = sorted(Path(path) for path in glob.glob(str(input_dir / "rho_step_*.csv")))
    if not rho_files:
        raise SystemExit(f"No rho_step_*.csv files found in {input_dir}")

    for rho_path in rho_files:
        step = step_from_name(rho_path)
        rho = load_csv(rho_path)
        plot_density(plots_dir, step, rho)

        ux_path = input_dir / f"ux_step_{step}.csv"
        uy_path = input_dir / f"uy_step_{step}.csv"
        if ux_path.exists() and uy_path.exists():
            plot_velocity(plots_dir, step, load_csv(ux_path), load_csv(uy_path))

        print(f"Generated plots for step {step}")


if __name__ == "__main__":
    main()
