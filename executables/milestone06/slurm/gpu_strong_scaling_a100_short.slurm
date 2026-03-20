#!/bin/bash -l
#SBATCH --job-name=yalb-m6-gpu-strong
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load compiler/gnu mpi/openmpi

# Override by exporting before sbatch, e.g.:
# export NPROCS=1,2,4
# export NX=512 NY=512 STEPS=5000
NPROCS=${NPROCS:-1,2,4}
NX=${NX:-512}
NY=${NY:-512}
STEPS=${STEPS:-5000}
BUILD_DIR=${BUILD_DIR:-build_release_gpu_strong}
OUTDIR=${OUTDIR:-${BUILD_DIR}/outputs_m6_scaling_gpu}

export CC=gcc
export CXX=g++
meson setup --wipe "${BUILD_DIR}" --buildtype=release
meson compile -C "${BUILD_DIR}" executables/milestone06/milestone06

python3 executables/milestone06/measure_scaling.py \
  --build-dir "${BUILD_DIR}" \
  --output-dir "${OUTDIR}" \
  --nprocs "${NPROCS}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --steps "${STEPS}" \
  --omega 1.0 \
  --rho0 1.0 \
  --lid-ux 0.05 \
  --lid-uy 0.0

# Optional: generate cavity plots for the largest np run.
LAST_NP=$(echo "${NPROCS}" | awk -F, '{gsub(/ /, "", $NF); print $NF}')
python3 executables/milestone06/plot.py --input-dir "${OUTDIR}/run_np${LAST_NP}" || true

echo "Wrote ${OUTDIR}/scaling.csv"
echo "Wrote ${OUTDIR}/speedup_vs_np.png"
