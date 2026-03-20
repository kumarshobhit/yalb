#!/bin/bash -l
#SBATCH --job-name=yalb-m6-gpu-sanity
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load compiler/gnu mpi/openmpi

# Build GPU release milestone06 (uses installed kokkoscore from module environment).
export CC=gcc
export CXX=g++
GPU_BUILD_DIR=${GPU_BUILD_DIR:-build_release_gpu_sanity}
CPU_BUILD_DIR=${CPU_BUILD_DIR:-build_release_cpu_sanity}
meson setup --wipe "${GPU_BUILD_DIR}" --buildtype=release
meson compile -C "${GPU_BUILD_DIR}" executables/milestone06/milestone06

# Build CPU release milestone05 reference if needed.
meson setup --wipe "${CPU_BUILD_DIR}" --buildtype=release
meson compile -C "${CPU_BUILD_DIR}" executables/milestone05/milestone05

CPU_OUT="${CPU_BUILD_DIR}/verify_gpu_ref_m5"
GPU_OUT="${GPU_BUILD_DIR}/verify_gpu_m6_np1"
CMP_OUT="${GPU_BUILD_DIR}/verify_gpu_vs_cpu.txt"

mkdir -p "${CPU_OUT}" "${GPU_OUT}"

./"${CPU_BUILD_DIR}"/executables/milestone05/milestone05 \
  --nx 96 --ny 96 --steps 1200 \
  --omega 1.0 --rho0 1.0 --lid-ux 0.05 --lid-uy 0.0 \
  --write-field-every 1200 --residual-every 300 \
  --output-dir "${CPU_OUT}"

srun --ntasks=1 --gpus-per-task=1 ./"${GPU_BUILD_DIR}"/executables/milestone06/milestone06 \
  --nx 96 --ny 96 --steps 1200 \
  --omega 1.0 --rho0 1.0 --lid-ux 0.05 --lid-uy 0.0 \
  --write-field-every 1200 --residual-every 300 \
  --output-dir "${GPU_OUT}"

python3 executables/milestone06/compare_fields.py \
  --ref-dir "${CPU_OUT}" \
  --cmp-dir "${GPU_OUT}" \
  --step 1200 \
  --out "${CMP_OUT}"

echo "Wrote ${CMP_OUT}"
