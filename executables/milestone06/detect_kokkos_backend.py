#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Kokkos backend symbols in an executable.")
    parser.add_argument("--exe", required=True, help="Path to milestone executable")
    parser.add_argument(
        "--require",
        choices=["any", "serial", "cuda"],
        default="any",
        help="Fail unless the requested backend family is detected",
    )
    args = parser.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe}")

    try:
        strings_out = subprocess.check_output(["strings", str(exe)], text=True, errors="replace")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Failed to run strings on {exe}: {exc}") from exc

    has_cuda = any(token in strings_out for token in ["Kokkos::Cuda", "CudaSpace", "CudaUVMSpace"])
    has_serial = "Kokkos::Serial" in strings_out or "Host Serial Execution Space" in strings_out
    has_openmp = "Kokkos::OpenMP" in strings_out

    detected = []
    if has_cuda:
        detected.append("cuda")
    if has_openmp:
        detected.append("openmp")
    if has_serial:
        detected.append("serial")
    if not detected:
        detected.append("unknown")

    print("Detected backends:", ",".join(detected))

    if args.require == "cuda" and not has_cuda:
        raise SystemExit("ERROR: CUDA backend symbols not found in executable.")
    if args.require == "serial" and not has_serial:
        raise SystemExit("ERROR: SERIAL backend symbols not found in executable.")


if __name__ == "__main__":
    main()
