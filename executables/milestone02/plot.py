import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"


def step_from_name(path: Path) -> str:
    return path.stem.split("_")[-1]


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def plot_density(step: str, rho: np.ndarray) -> None:
    plt.figure()
    plt.imshow(rho, origin="lower", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Density")
    plt.title(f"Density step {step}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"rho_step_{step}.png")
    plt.close()


def plot_velocity(step: str, ux: np.ndarray, uy: np.ndarray) -> None:
    x = np.arange(ux.shape[1])
    y = np.arange(ux.shape[0])
    grid_x, grid_y = np.meshgrid(x, y)
    speed = np.sqrt(ux**2 + uy**2)

    plt.figure()
    plt.streamplot(grid_x, grid_y, ux, uy, color=speed, cmap="plasma", density=1.0)
    plt.colorbar(label="Speed")
    plt.xlim(0, ux.shape[1] - 1)
    plt.ylim(0, ux.shape[0] - 1)
    plt.title(f"Velocity step {step}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"velocity_step_{step}.png")
    plt.close()


def main() -> None:
    if not OUTPUT_DIR.exists():
        raise SystemExit("No outputs directory found. Run milestone02 first.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rho_files = sorted(Path(path) for path in glob.glob(str(OUTPUT_DIR / "rho_step_*.csv")))
    if not rho_files:
        raise SystemExit("No rho_step_*.csv files found in outputs/.")

    for rho_path in rho_files:
        step = step_from_name(rho_path)
        rho = load_csv(rho_path)
        plot_density(step, rho)

        ux_path = OUTPUT_DIR / f"ux_step_{step}.csv"
        uy_path = OUTPUT_DIR / f"uy_step_{step}.csv"
        if ux_path.exists() and uy_path.exists():
            plot_velocity(step, load_csv(ux_path), load_csv(uy_path))

        print(f"Generated plots for step {step}")


if __name__ == "__main__":
    main()
