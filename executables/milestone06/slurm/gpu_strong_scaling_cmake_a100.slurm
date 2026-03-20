#!/bin/bash -l
#SBATCH --job-name=lbm-gpu-strong-cmake
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load compiler/gnu mpi/openmpi devel/cuda

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

NX=${NX:-2000}
NY=${NY:-2000}
STEPS=${STEPS:-2000}
OMEGA=${OMEGA:-1.0}
RHO0=${RHO0:-1.0}
LID_UX=${LID_UX:-0.05}
LID_UY=${LID_UY:-0.0}

NPROCS_CSV=${NPROCS_CSV:-1,2,4}
IFS=',' read -r -a NPROCS <<< "${NPROCS_CSV}"

KOKKOS_CUDA_ARCH=${KOKKOS_CUDA_ARCH:-AMPERE80}
KOKKOS_SRC=${KOKKOS_SRC:-${SLURM_SUBMIT_DIR}/subprojects/kokkos}
KOKKOS_BUILD_DIR=${KOKKOS_BUILD_DIR:-build_kokkos_cuda}
KOKKOS_INSTALL_DIR=${KOKKOS_INSTALL_DIR:-${KOKKOS_BUILD_DIR}/install}
KOKKOS_NVCC_WRAPPER=${KOKKOS_NVCC_WRAPPER:-${KOKKOS_SRC}/bin/nvcc_wrapper}
KOKKOS_ROOT=${KOKKOS_ROOT:-}

BUILD_DIR=${BUILD_DIR:-build_release_gpu_strong_cmake}
OUTDIR=${OUTDIR:-outputs_gpu}
EXE="${BUILD_DIR}/executables/milestone06/milestone06"

echo "[gpu-strong-cmake] Kokkos src: ${KOKKOS_SRC}"
echo "[gpu-strong-cmake] Kokkos build: ${KOKKOS_BUILD_DIR}"
echo "[gpu-strong-cmake] Kokkos install: ${KOKKOS_INSTALL_DIR}"
echo "[gpu-strong-cmake] Build directory: ${BUILD_DIR}"
echo "[gpu-strong-cmake] Output directory: ${OUTDIR}"
echo "[gpu-strong-cmake] Fixed case: NX=${NX} NY=${NY} STEPS=${STEPS}"
echo "[gpu-strong-cmake] NPROCS: ${NPROCS[*]}"

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/run_np*.log

if [[ -n "${KOKKOS_ROOT}" ]]; then
  echo "[gpu-strong-cmake] Using existing Kokkos install at ${KOKKOS_ROOT}"
  KOKKOS_INSTALL_DIR="${KOKKOS_ROOT}"
else
  cmake -S "${KOKKOS_SRC}" -B "${KOKKOS_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_DIR}" \
    -DCMAKE_CXX_COMPILER="${KOKKOS_NVCC_WRAPPER}" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ARCH_${KOKKOS_CUDA_ARCH}=ON

  cmake --build "${KOKKOS_BUILD_DIR}" -j
  cmake --install "${KOKKOS_BUILD_DIR}"
fi

cmake -S executables/milestone06/cmake -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${KOKKOS_NVCC_WRAPPER}" \
  -DKokkos_DIR="${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos"

cmake --build "${BUILD_DIR}" -j

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: executable not found: ${EXE}" >&2
    exit 1
fi

python3 executables/milestone06/detect_kokkos_backend.py --exe "${EXE}" --require cuda

max_np="${NPROCS[${#NPROCS[@]}-1]}"
allocated_tasks="${SLURM_NTASKS:-0}"
if (( allocated_tasks > 0 && allocated_tasks < max_np )); then
    echo "ERROR: Allocation provides ${allocated_tasks} tasks but max np is ${max_np}." >&2
    echo "Increase --nodes/--ntasks-per-node or reduce NPROCS." >&2
    exit 2
fi

for np in "${NPROCS[@]}"; do
    log_file="${OUTDIR}/run_np${np}.log"
    run_dir="${OUTDIR}/fields_np${np}"
    mkdir -p "${run_dir}"

    echo "[gpu-strong-cmake] Running np=${np}, log=${log_file}"
    srun --ntasks="${np}" --gpus-per-task=1 --cpus-per-task=1 "${EXE}" \
      --nx "${NX}" \
      --ny "${NY}" \
      --steps "${STEPS}" \
      --omega "${OMEGA}" \
      --rho0 "${RHO0}" \
      --lid-ux "${LID_UX}" \
      --lid-uy "${LID_UY}" \
      --write-field-every "${STEPS}" \
      --residual-every 500 \
      --output-dir "${run_dir}" \
      > "${log_file}" 2>&1

    if ! grep -q "MPI ranks: ${np}," "${log_file}"; then
        echo "ERROR: np=${np} run did not report expected MPI rank count." >&2
        tail -n 60 "${log_file}" >&2 || true
        exit 3
    fi
done

python3 executables/milestone06/parse_strong_scaling.py \
  --log-dir "${OUTDIR}" \
  --nprocs "${NPROCS_CSV}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --steps "${STEPS}" \
  --output-csv "${OUTDIR}/scaling_metrics.csv"

python3 executables/milestone06/plot_strong_scaling.py \
  --csv "${OUTDIR}/scaling_metrics.csv" \
  --outdir "${OUTDIR}"

echo "[gpu-strong-cmake] Completed."
echo "[gpu-strong-cmake] Wrote ${OUTDIR}/scaling_metrics.csv"
