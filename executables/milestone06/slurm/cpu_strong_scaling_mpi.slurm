#!/bin/bash -l
#SBATCH --job-name=lbm-cpu-strong-scaling
#SBATCH --partition=${PARTITION:-<CPU_PARTITION>}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load compiler/gnu mpi/openmpi

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Fixed workload for strong scaling (override via environment if needed)
NX=${NX:-2000}
NY=${NY:-2000}
STEPS=${STEPS:-2000}
OMEGA=${OMEGA:-1.0}
RHO0=${RHO0:-1.0}
LID_UX=${LID_UX:-0.05}
LID_UY=${LID_UY:-0.0}

# MPI process counts to benchmark (override via NPROCS_CSV, e.g. 1,2,4,8)
NPROCS_CSV=${NPROCS_CSV:-1,2,4,8,16,32}
IFS=',' read -r -a NPROCS <<< "${NPROCS_CSV}"

BUILD_DIR=${BUILD_DIR:-build_release_cpu_strong}
OUTDIR=${OUTDIR:-outputs}
EXE="${BUILD_DIR}/executables/milestone06/milestone06"

echo "[strong-scaling] Build directory: ${BUILD_DIR}"
echo "[strong-scaling] Output directory: ${OUTDIR}"
echo "[strong-scaling] Fixed case: NX=${NX} NY=${NY} STEPS=${STEPS}"

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/run_np*.log

export CC=gcc
export CXX=g++
meson setup --wipe "${BUILD_DIR}" \
  --buildtype=release \
  -Dkokkos_backend=serial
meson compile -C "${BUILD_DIR}" executables/milestone06/milestone06

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: executable not found: ${EXE}" >&2
    exit 1
fi

max_np="${NPROCS[${#NPROCS[@]}-1]}"
allocated_tasks="${SLURM_NTASKS:-0}"
if (( allocated_tasks > 0 && allocated_tasks < max_np )); then
    echo "ERROR: Allocation provides ${allocated_tasks} MPI tasks but max np is ${max_np}." >&2
    echo "Increase --nodes/--ntasks-per-node or reduce NPROCS." >&2
    exit 2
fi

for np in "${NPROCS[@]}"; do
    log_file="${OUTDIR}/run_np${np}.log"
    echo "[strong-scaling] Running np=${np}, log=${log_file}"

    mpirun -np "${np}" "${EXE}" \
      --nx "${NX}" \
      --ny "${NY}" \
      --steps "${STEPS}" \
      --omega "${OMEGA}" \
      --rho0 "${RHO0}" \
      --lid-ux "${LID_UX}" \
      --lid-uy "${LID_UY}" \
      --write-field-every "${STEPS}" \
      --residual-every 500 \
      --output-dir "${OUTDIR}/fields_np${np}" \
      > "${log_file}" 2>&1

    if ! grep -q "MPI ranks: ${np}," "${log_file}"; then
        echo "ERROR: np=${np} run did not report expected MPI rank count." >&2
        echo "Last log lines:" >&2
        tail -n 40 "${log_file}" >&2
        exit 3
    fi
done

echo "[strong-scaling] Completed. Logs:"
ls -1 "${OUTDIR}"/run_np*.log
