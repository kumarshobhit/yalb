import argparse
import csv
import math
from pathlib import Path

import numpy as np


def step_from_name(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def available_steps(input_dir: Path, prefix: str) -> set[int]:
    return {step_from_name(path) for path in input_dir.glob(f"{prefix}_step_*.csv")}


def load_field(input_dir: Path, prefix: str, step: int) -> np.ndarray:
    filename = input_dir / f"{prefix}_step_{step:05d}.csv"
    if not filename.exists():
        raise FileNotFoundError(f"Missing field file: {filename}")
    return np.loadtxt(filename, delimiter=",")


def l2_relative(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a)
    if den == 0.0:
        return float(num)
    return float(num / den)


def read_residuals(path: Path) -> dict[int, tuple[float, float | None, float | None]]:
    if not path.exists():
        return {}

    result: dict[int, tuple[float, float | None, float | None]] = {}
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            step = int(row["step"])
            max_delta = float(row["max_delta_u"])
            mass = float(row["global_mass"]) if "global_mass" in row and row["global_mass"] else None
            ke = float(row["global_ke"]) if "global_ke" in row and row["global_ke"] else None
            result[step] = (max_delta, mass, ke)
    return result


def rel_diff(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-14)
    return abs(a - b) / scale


def summarize(ref_dir: Path, cmp_dir: Path, step: int) -> dict[str, float]:
    rho_ref = load_field(ref_dir, "rho", step)
    ux_ref = load_field(ref_dir, "ux", step)
    uy_ref = load_field(ref_dir, "uy", step)

    rho_cmp = load_field(cmp_dir, "rho", step)
    ux_cmp = load_field(cmp_dir, "ux", step)
    uy_cmp = load_field(cmp_dir, "uy", step)

    if rho_ref.shape != rho_cmp.shape:
        raise RuntimeError(f"rho shapes differ: {rho_ref.shape} vs {rho_cmp.shape}")
    if ux_ref.shape != ux_cmp.shape or uy_ref.shape != uy_cmp.shape:
        raise RuntimeError("Velocity field shapes differ between compared runs")

    center_x = rho_ref.shape[1] // 2
    center_y = rho_ref.shape[0] // 2

    ux_center_ref = ux_ref[:, center_x]
    ux_center_cmp = ux_cmp[:, center_x]
    uy_center_ref = uy_ref[center_y, :]
    uy_center_cmp = uy_cmp[center_y, :]

    metrics: dict[str, float] = {
        "step": float(step),
        "rho_l2_rel": l2_relative(rho_ref, rho_cmp),
        "ux_l2_rel": l2_relative(ux_ref, ux_cmp),
        "uy_l2_rel": l2_relative(uy_ref, uy_cmp),
        "ux_centerline_l2_rel": l2_relative(ux_center_ref, ux_center_cmp),
        "uy_centerline_l2_rel": l2_relative(uy_center_ref, uy_center_cmp),
        "mass_ref": float(np.sum(rho_ref)),
        "mass_cmp": float(np.sum(rho_cmp)),
    }
    metrics["mass_rel_diff"] = rel_diff(metrics["mass_ref"], metrics["mass_cmp"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", required=True, help="Reference output directory")
    parser.add_argument("--cmp-dir", required=True, help="Compared output directory")
    parser.add_argument("--step", type=int, default=-1, help="Step to compare; default: latest common step")
    parser.add_argument("--out", default="", help="Optional output JSON-like text file for metrics")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    cmp_dir = Path(args.cmp_dir)

    common_steps = sorted(available_steps(ref_dir, "rho") & available_steps(cmp_dir, "rho"))
    if not common_steps:
        raise SystemExit("No common rho_step_#####.csv files between directories")

    if args.step >= 0:
        step = args.step
        if step not in common_steps:
            raise SystemExit(f"Requested step {step} not common between runs")
    else:
        step = common_steps[-1]

    metrics = summarize(ref_dir, cmp_dir, step)

    ref_residuals = read_residuals(ref_dir / "residual_history.csv")
    cmp_residuals = read_residuals(cmp_dir / "residual_history.csv")
    common_res_steps = sorted(set(ref_residuals.keys()) & set(cmp_residuals.keys()))

    if common_res_steps:
        final = common_res_steps[-1]
        ref_max_delta, ref_mass, ref_ke = ref_residuals[final]
        cmp_max_delta, cmp_mass, cmp_ke = cmp_residuals[final]
        metrics["residual_step"] = float(final)
        metrics["max_delta_rel_diff"] = rel_diff(ref_max_delta, cmp_max_delta)
        if ref_mass is not None and cmp_mass is not None:
            metrics["global_mass_rel_diff"] = rel_diff(ref_mass, cmp_mass)
        if ref_ke is not None and cmp_ke is not None:
            metrics["global_ke_rel_diff"] = rel_diff(ref_ke, cmp_ke)

    lines = [
        f"step={int(metrics['step'])}",
        f"rho_l2_rel={metrics['rho_l2_rel']:.6e}",
        f"ux_l2_rel={metrics['ux_l2_rel']:.6e}",
        f"uy_l2_rel={metrics['uy_l2_rel']:.6e}",
        f"ux_centerline_l2_rel={metrics['ux_centerline_l2_rel']:.6e}",
        f"uy_centerline_l2_rel={metrics['uy_centerline_l2_rel']:.6e}",
        f"mass_rel_diff={metrics['mass_rel_diff']:.6e}",
    ]

    if "max_delta_rel_diff" in metrics:
        lines.append(f"max_delta_rel_diff={metrics['max_delta_rel_diff']:.6e}")
    if "global_mass_rel_diff" in metrics:
        lines.append(f"global_mass_rel_diff={metrics['global_mass_rel_diff']:.6e}")
    if "global_ke_rel_diff" in metrics:
        lines.append(f"global_ke_rel_diff={metrics['global_ke_rel_diff']:.6e}")

    output = "\n".join(lines)
    print(output)

    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
