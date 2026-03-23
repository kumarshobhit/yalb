import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',')


def step_from_name(path: Path) -> int:
    return int(path.stem.split('_')[-1])


def collect_step_files(input_dir: Path, prefix: str) -> list[Path]:
    files = sorted(Path(path) for path in glob.glob(str(input_dir / f"{prefix}_step_*.csv")))
    if not files:
        raise SystemExit(f"No files found for pattern {prefix}_step_*.csv in {input_dir}")
    return files


def available_steps(input_dir: Path) -> list[int]:
    ux_steps = {step_from_name(path) for path in collect_step_files(input_dir, 'ux')}
    uy_steps = {step_from_name(path) for path in collect_step_files(input_dir, 'uy')}
    steps = sorted(ux_steps & uy_steps)
    if not steps:
        raise SystemExit(f'No common ux/uy steps in {input_dir}')
    return steps


def make_frame(ux: np.ndarray, uy: np.ndarray, step: int, out_path: Path, vmax: float, quiver_stride: int) -> None:
    speed = np.sqrt(ux**2 + uy**2)
    ny, nx = ux.shape
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=140)
    im = ax.imshow(speed, origin='lower', cmap='magma', vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, label='|u|')

    stride = max(1, quiver_stride)
    ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        ux[::stride, ::stride],
        uy[::stride, ::stride],
        color='white',
        pivot='mid',
        angles='xy',
        scale_units='xy',
        scale=0.35,
        alpha=0.85,
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Lid-driven cavity flow, step {step}')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directory containing ux_step_*.csv and uy_step_*.csv')
    parser.add_argument('--output', default='', help='Output GIF path (default: <input-dir>/plots/cavity_flow.gif)')
    parser.add_argument('--duration-ms', type=int, default=250, help='Frame duration in milliseconds')
    parser.add_argument('--quiver-stride', type=int, default=16, help='Subsampling stride for quiver arrows')
    parser.add_argument('--max-frames', type=int, default=0, help='If > 0, uniformly subsample to at most this many frames')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f'Input directory does not exist: {input_dir}')

    steps = available_steps(input_dir)
    if args.max_frames > 0 and len(steps) > args.max_frames:
        indices = np.linspace(0, len(steps) - 1, args.max_frames, dtype=int)
        steps = [steps[idx] for idx in indices]

    plots_dir = input_dir / 'plots'
    frames_dir = plots_dir / 'animation_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else plots_dir / 'cavity_flow.gif'
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = 0.0
    for step in steps:
        ux = load_csv(input_dir / f'ux_step_{step:05d}.csv')
        uy = load_csv(input_dir / f'uy_step_{step:05d}.csv')
        vmax = max(vmax, float(np.sqrt(ux**2 + uy**2).max()))
    vmax = max(vmax, 1e-12)

    frame_paths: list[Path] = []
    for step in steps:
        ux = load_csv(input_dir / f'ux_step_{step:05d}.csv')
        uy = load_csv(input_dir / f'uy_step_{step:05d}.csv')
        frame_path = frames_dir / f'frame_{step:05d}.png'
        make_frame(ux, uy, step, frame_path, vmax, args.quiver_stride)
        frame_paths.append(frame_path)

    images = [Image.open(path).convert('P', palette=Image.ADAPTIVE) for path in frame_paths]
    if not images:
        raise SystemExit('No frames generated')
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()

    print(f'Wrote animation to {output_path}')
    print(f'Wrote individual frames to {frames_dir}')


if __name__ == '__main__':
    main()
