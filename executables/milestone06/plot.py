import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def step_from_name(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def collect_step_files(input_dir: Path, prefix: str) -> list[Path]:
    files = sorted(Path(path) for path in glob.glob(str(input_dir / f"{prefix}_step_*.csv")))
    if not files:
        raise SystemExit(f"No files found for pattern {prefix}_step_*.csv in {input_dir}")
    return files


def plot_quiver(ux: np.ndarray, uy: np.ndarray, output_path: Path) -> None:
    ny, nx = ux.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    stride = max(1, min(nx, ny) // 24)
    plt.figure(figsize=(7, 6))
    plt.quiver(xx[::stride, ::stride], yy[::stride, ::stride], ux[::stride, ::stride], uy[::stride, ::stride], scale=None)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity Quiver (Last Step)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_streamlines(ux: np.ndarray, uy: np.ndarray, output_path: Path) -> None:
    ny, nx = ux.shape
    x = np.arange(nx)
    y = np.arange(ny)
    speed = np.sqrt(ux**2 + uy**2)

    plt.figure(figsize=(7, 6))
    plt.streamplot(x, y, ux, uy, density=1.2, color=speed, cmap="viridis")
    plt.colorbar(label="|u|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Streamlines (Last Step)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_speed_contour(ux: np.ndarray, uy: np.ndarray, output_path: Path) -> None:
    speed = np.sqrt(ux**2 + uy**2)
    plt.figure(figsize=(7, 6))
    plt.contourf(speed, levels=30, cmap="magma")
    plt.colorbar(label="|u|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Speed Magnitude (Last Step)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_centerlines(ux: np.ndarray, uy: np.ndarray, plots_dir: Path) -> None:
    ny, nx = ux.shape
    center_x = nx // 2
    center_y = ny // 2

    plt.figure()
    plt.plot(np.arange(ny), ux[:, center_x])
    plt.xlabel("y")
    plt.ylabel("u_x(x=nx/2, y)")
    plt.title("Vertical Centerline Velocity")
    plt.tight_layout()
    plt.savefig(plots_dir / "u_centerline_vertical_last.png")
    plt.close()

    plt.figure()
    plt.plot(np.arange(nx), uy[center_y, :])
    plt.xlabel("x")
    plt.ylabel("u_y(x, y=ny/2)")
    plt.title("Horizontal Centerline Velocity")
    plt.tight_layout()
    plt.savefig(plots_dir / "v_centerline_horizontal_last.png")
    plt.close()


def plot_profile_overlays(rho_files: list[Path], ux_files: list[Path], plots_dir: Path) -> None:
    velocity_profiles: list[tuple[int, np.ndarray]] = []
    density_profiles: list[tuple[int, np.ndarray]] = []

    for ux_path in ux_files:
        velocity_profiles.append((step_from_name(ux_path), load_csv(ux_path).mean(axis=1)))
    for rho_path in rho_files:
        density_profiles.append((step_from_name(rho_path), load_csv(rho_path).mean(axis=1)))

    velocity_profiles.sort(key=lambda entry: entry[0])
    density_profiles.sort(key=lambda entry: entry[0])

    y_vel = np.arange(velocity_profiles[0][1].shape[0])
    vmax = max(float(np.max(np.abs(profile))) for _, profile in velocity_profiles)
    vlim = 1.1 * max(vmax, 1e-12)

    plt.figure()
    for step, profile in velocity_profiles:
        plt.plot(y_vel, profile, alpha=0.6, label=f"step {step}")
    plt.xlabel("y")
    plt.ylabel("u_x averaged over x")
    plt.title("Velocity Profile Overlay")
    plt.ylim(-vlim, vlim)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "velocity_profile_overlay.png")
    plt.close()

    y_rho = np.arange(density_profiles[0][1].shape[0])
    rho_min = min(float(np.min(profile)) for _, profile in density_profiles)
    rho_max = max(float(np.max(profile)) for _, profile in density_profiles)
    pad = max(1e-6, 0.1 * (rho_max - rho_min)) if rho_min != rho_max else max(1e-6, 0.01 * abs(rho_min))

    plt.figure()
    for step, profile in density_profiles:
        plt.plot(y_rho, profile, alpha=0.6, label=f"step {step}")
    plt.xlabel("y")
    plt.ylabel("rho averaged over x")
    plt.title("Density Profile Overlay")
    plt.ylim(rho_min - pad, rho_max + pad)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "density_profile_overlay.png")
    plt.close()


def plot_residual_history(input_dir: Path, plots_dir: Path) -> None:
    residual_path = input_dir / "residual_history.csv"
    if not residual_path.exists():
        raise SystemExit(f"Missing residual file: {residual_path}")

    residual = np.genfromtxt(residual_path, delimiter=",", names=True)
    if residual.shape == ():
        residual = np.array([residual], dtype=residual.dtype)

    plt.figure()
    values = residual["max_delta_u"]
    if np.all(values > 0.0):
        plt.semilogy(residual["step"], values)
    else:
        plt.plot(residual["step"], values)
    plt.xlabel("step")
    plt.ylabel("max_delta_u")
    plt.title("Residual History")
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_history.png")
    plt.close()

    if "global_mass" in residual.dtype.names:
        plt.figure()
        plt.plot(residual["step"], residual["global_mass"])
        plt.xlabel("step")
        plt.ylabel("global_mass")
        plt.title("Global Mass History")
        plt.tight_layout()
        plt.savefig(plots_dir / "global_mass_history.png")
        plt.close()

    if "global_ke" in residual.dtype.names:
        plt.figure()
        plt.plot(residual["step"], residual["global_ke"])
        plt.xlabel("step")
        plt.ylabel("global_ke")
        plt.title("Global Kinetic Energy History")
        plt.tight_layout()
        plt.savefig(plots_dir / "global_ke_history.png")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs_m6", help="Directory containing Milestone 06 output CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    ux_files = collect_step_files(input_dir, "ux")
    uy_files = collect_step_files(input_dir, "uy")
    rho_files = collect_step_files(input_dir, "rho")

    uy_by_step = {step_from_name(path): path for path in uy_files}
    last_ux_path = ux_files[-1]
    last_step = step_from_name(last_ux_path)

    if last_step not in uy_by_step:
        raise SystemExit(f"Missing uy file for last ux step {last_step}")

    ux_last = load_csv(last_ux_path)
    uy_last = load_csv(uy_by_step[last_step])

    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_quiver(ux_last, uy_last, plots_dir / "velocity_quiver_last.png")
    plot_streamlines(ux_last, uy_last, plots_dir / "streamlines_last.png")
    plot_speed_contour(ux_last, uy_last, plots_dir / "speed_contour_last.png")
    plot_centerlines(ux_last, uy_last, plots_dir)
    plot_profile_overlays(rho_files, ux_files, plots_dir)
    plot_residual_history(input_dir, plots_dir)

    print(f"Generated Milestone 06 plots in {plots_dir}")


if __name__ == "__main__":
    main()
