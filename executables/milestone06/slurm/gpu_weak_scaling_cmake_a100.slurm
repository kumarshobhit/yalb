#!/bin/bash -l
#SBATCH --job-name=lbm-gpu-weak-cmake
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
cd "${SUBMIT_DIR}"

module purge
module load compiler/gnu mpi/openmpi devel/cuda

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

BASE_LOCAL_NX=${BASE_LOCAL_NX:-2000}
NY=${NY:-2000}
STEPS=${STEPS:-2000}
OMEGA=${OMEGA:-1.0}
RHO0=${RHO0:-1.0}
LID_UX=${LID_UX:-0.05}
LID_UY=${LID_UY:-0.0}

NPROCS_CSV=${NPROCS_CSV:-1,2}
IFS=',' read -r -a NPROCS <<< "${NPROCS_CSV}"
SRUN_MPI_FLAG=${SRUN_MPI_FLAG:-pmix}
SRUN_GPU_BIND=${SRUN_GPU_BIND:-single:1}

KOKKOS_BOOTSTRAP=${KOKKOS_BOOTSTRAP:-auto}
KOKKOS_VERSION=${KOKKOS_VERSION:-4.5.01}
KOKKOS_WORK_ROOT=${KOKKOS_WORK_ROOT:-${SUBMIT_DIR}/build_m6}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-${KOKKOS_WORK_ROOT}/kokkos-install}
KOKKOS_SRC=${KOKKOS_SRC:-${KOKKOS_WORK_ROOT}/kokkos-src}
KOKKOS_BUILD=${KOKKOS_BUILD:-${KOKKOS_WORK_ROOT}/kokkos-build}
KOKKOS_SOURCE_TARBALL=${KOKKOS_SOURCE_TARBALL:-}
ALLOW_GIT_FETCH=${ALLOW_GIT_FETCH:-0}
CUDA_ARCH=${CUDA_ARCH:-80}
KOKKOS_DIR=${KOKKOS_DIR:-}

BUILD_DIR=${BUILD_DIR:-build_release_gpu_weak_cmake}
OUTDIR=${OUTDIR:-outputs_gpu_weak}
EXE="${BUILD_DIR}/executables/milestone06/milestone06"

usage_hint() {
  echo "Example rerun:" >&2
  echo "  KOKKOS_SOURCE_TARBALL=/path/to/kokkos-${KOKKOS_VERSION}.tar.gz sbatch executables/milestone06/slurm/gpu_weak_scaling_cmake_a100.slurm" >&2
}

fail() {
  echo "ERROR: $*" >&2
  usage_hint
  exit 10
}

resolve_kokkos_dir_from_prefix() {
  local prefix="$1"
  if [[ -f "${prefix}/lib64/cmake/Kokkos/KokkosConfig.cmake" ]]; then
    echo "${prefix}/lib64/cmake/Kokkos"
    return 0
  fi
  if [[ -f "${prefix}/lib/cmake/Kokkos/KokkosConfig.cmake" ]]; then
    echo "${prefix}/lib/cmake/Kokkos"
    return 0
  fi
  return 1
}

map_cuda_arch_to_kokkos_flag() {
  case "$1" in
    80|AMPERE80|ampere80) echo "Kokkos_ARCH_AMPERE80" ;;
    86|AMPERE86|ampere86) echo "Kokkos_ARCH_AMPERE86" ;;
    70|VOLTA70|volta70) echo "Kokkos_ARCH_VOLTA70" ;;
    75|TURING75|turing75) echo "Kokkos_ARCH_TURING75" ;;
    *) return 1 ;;
  esac
}

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/run_np*.log

case "${KOKKOS_BOOTSTRAP}" in
  auto|never|force) ;;
  *) fail "KOKKOS_BOOTSTRAP must be one of: auto, never, force (got: ${KOKKOS_BOOTSTRAP})" ;;
esac

if ! KOKKOS_ARCH_FLAG="$(map_cuda_arch_to_kokkos_flag "${CUDA_ARCH}")"; then
  fail "Unsupported CUDA_ARCH='${CUDA_ARCH}'. Supported values: 80, 86, 70, 75 (or AMPERE80/AMPERE86/VOLTA70/TURING75)."
fi

mkdir -p "${KOKKOS_WORK_ROOT}"

if [[ "${KOKKOS_BOOTSTRAP}" != "force" ]]; then
  if [[ -n "${KOKKOS_DIR}" ]]; then
    if [[ -f "${KOKKOS_DIR}/KokkosConfig.cmake" ]]; then
      echo "[gpu-weak-cmake] Using user-provided KOKKOS_DIR=${KOKKOS_DIR}"
    else
      echo "[gpu-weak-cmake] Warning: ignoring invalid KOKKOS_DIR=${KOKKOS_DIR} (KokkosConfig.cmake missing)"
      KOKKOS_DIR=""
    fi
  fi

  if [[ -z "${KOKKOS_DIR}" ]]; then
    if resolved="$(resolve_kokkos_dir_from_prefix "${KOKKOS_PREFIX}")"; then
      KOKKOS_DIR="${resolved}"
      echo "[gpu-weak-cmake] Found Kokkos in KOKKOS_PREFIX: ${KOKKOS_DIR}"
    fi
  fi

  if [[ -z "${KOKKOS_DIR}" && -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    IFS=':' read -r -a cmake_prefixes <<< "${CMAKE_PREFIX_PATH}"
    for prefix in "${cmake_prefixes[@]}"; do
      if [[ -z "${prefix}" ]]; then
        continue
      fi
      if resolved="$(resolve_kokkos_dir_from_prefix "${prefix}")"; then
        KOKKOS_DIR="${resolved}"
        echo "[gpu-weak-cmake] Found Kokkos in CMAKE_PREFIX_PATH entry: ${KOKKOS_DIR}"
        break
      fi
    done
  fi
fi

need_bootstrap=0
if [[ "${KOKKOS_BOOTSTRAP}" == "force" ]]; then
  need_bootstrap=1
elif [[ -z "${KOKKOS_DIR}" ]]; then
  if [[ "${KOKKOS_BOOTSTRAP}" == "auto" ]]; then
    need_bootstrap=1
  else
    fail "Kokkos not found and KOKKOS_BOOTSTRAP=never. Set KOKKOS_DIR or KOKKOS_PREFIX to a valid install."
  fi
fi

if (( need_bootstrap )); then
  echo "[gpu-weak-cmake] Bootstrapping Kokkos (mode=${KOKKOS_BOOTSTRAP})"
  echo "[gpu-weak-cmake] Kokkos source: ${KOKKOS_SRC}"
  echo "[gpu-weak-cmake] Kokkos build: ${KOKKOS_BUILD}"
  echo "[gpu-weak-cmake] Kokkos install: ${KOKKOS_PREFIX}"

  if [[ -n "${KOKKOS_SOURCE_TARBALL}" ]]; then
    if [[ ! -f "${KOKKOS_SOURCE_TARBALL}" ]]; then
      fail "KOKKOS_SOURCE_TARBALL is set but file does not exist: ${KOKKOS_SOURCE_TARBALL}"
    fi
    rm -rf "${KOKKOS_SRC}"
    mkdir -p "${KOKKOS_SRC}"
    tar -xf "${KOKKOS_SOURCE_TARBALL}" -C "${KOKKOS_SRC}" --strip-components=1
  elif [[ "${ALLOW_GIT_FETCH}" == "1" ]]; then
    rm -rf "${KOKKOS_SRC}"
    git clone --branch "${KOKKOS_VERSION}" --depth 1 https://github.com/kokkos/kokkos.git "${KOKKOS_SRC}"
  elif [[ ! -f "${KOKKOS_SRC}/CMakeLists.txt" ]]; then
    fail "No Kokkos source available. Set KOKKOS_SOURCE_TARBALL or enable ALLOW_GIT_FETCH=1."
  fi

  rm -rf "${KOKKOS_BUILD}" "${KOKKOS_PREFIX}"
  cmake -S "${KOKKOS_SRC}" -B "${KOKKOS_BUILD}"     -DCMAKE_BUILD_TYPE=Release     -DCMAKE_INSTALL_PREFIX="${KOKKOS_PREFIX}"     -DCMAKE_CXX_COMPILER="${KOKKOS_SRC}/bin/nvcc_wrapper"     -DKokkos_ENABLE_CUDA=ON     -DKokkos_ENABLE_SERIAL=ON     -DKokkos_ENABLE_TESTS=OFF     "-D${KOKKOS_ARCH_FLAG}=ON"

  cmake --build "${KOKKOS_BUILD}" -j
  cmake --install "${KOKKOS_BUILD}"

  if ! KOKKOS_DIR="$(resolve_kokkos_dir_from_prefix "${KOKKOS_PREFIX}")"; then
    fail "Bootstrap completed but KokkosConfig.cmake not found under ${KOKKOS_PREFIX}/lib64 or ${KOKKOS_PREFIX}/lib."
  fi
fi

if [[ -z "${KOKKOS_DIR}" || ! -f "${KOKKOS_DIR}/KokkosConfig.cmake" ]]; then
  fail "Kokkos_DIR is invalid: ${KOKKOS_DIR:-<unset>}"
fi

KOKKOS_PREFIX_FROM_DIR=""
if [[ -n "${KOKKOS_DIR}" ]]; then
  KOKKOS_PREFIX_FROM_DIR="$(cd "${KOKKOS_DIR}/../../.." && pwd -P)"
fi

KOKKOS_NVCC_WRAPPER="${KOKKOS_PREFIX}/bin/nvcc_wrapper"
if [[ ! -x "${KOKKOS_NVCC_WRAPPER}" && -n "${KOKKOS_PREFIX_FROM_DIR}" ]]; then
  KOKKOS_NVCC_WRAPPER="${KOKKOS_PREFIX_FROM_DIR}/bin/nvcc_wrapper"
fi
if [[ ! -x "${KOKKOS_NVCC_WRAPPER}" ]]; then
  KOKKOS_NVCC_WRAPPER="${KOKKOS_SRC}/bin/nvcc_wrapper"
fi
if [[ ! -x "${KOKKOS_NVCC_WRAPPER}" ]]; then
  fail "Could not find executable nvcc_wrapper in ${KOKKOS_PREFIX}/bin, ${KOKKOS_PREFIX_FROM_DIR:-<derived-none>}/bin, or ${KOKKOS_SRC}/bin."
fi

echo "[gpu-weak-cmake] KOKKOS_BOOTSTRAP: ${KOKKOS_BOOTSTRAP}"
echo "[gpu-weak-cmake] KOKKOS_PREFIX: ${KOKKOS_PREFIX}"
echo "[gpu-weak-cmake] Resolved KOKKOS_DIR: ${KOKKOS_DIR}"
echo "[gpu-weak-cmake] Kokkos compiler wrapper: ${KOKKOS_NVCC_WRAPPER}"
echo "[gpu-weak-cmake] Build directory: ${BUILD_DIR}"
echo "[gpu-weak-cmake] Output directory: ${OUTDIR}"
echo "[gpu-weak-cmake] Weak case: BASE_LOCAL_NX=${BASE_LOCAL_NX} NY=${NY} STEPS=${STEPS}"
echo "[gpu-weak-cmake] NPROCS: ${NPROCS[*]}"

rm -rf "${BUILD_DIR}"
cmake -S executables/milestone06/cmake -B "${BUILD_DIR}"   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CXX_COMPILER="${KOKKOS_NVCC_WRAPPER}"   -DKokkos_DIR="${KOKKOS_DIR}"

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

allocated_nodes="${SLURM_JOB_NUM_NODES:-0}"
if (( allocated_nodes > 0 && allocated_nodes != 1 )); then
    echo "ERROR: This script is configured for a single-node run, but the allocation has ${allocated_nodes} nodes." >&2
    echo "Use a 1-node allocation or switch to the multi-node launcher." >&2
    exit 2
fi

for np in "${NPROCS[@]}"; do
    nx=$(( BASE_LOCAL_NX * np ))
    log_file="${OUTDIR}/run_np${np}.log"
    run_dir="${OUTDIR}/fields_np${np}"
    mkdir -p "${run_dir}"

    echo "[gpu-weak-cmake] Running np=${np}, nx=${nx}, ny=${NY}, log=${log_file}"
    srun --mpi="${SRUN_MPI_FLAG}" --nodes=1 --ntasks="${np}" --ntasks-per-node="${np}"       --gpus-per-task=1 --gpu-bind="${SRUN_GPU_BIND}" --cpus-per-task=1 "${EXE}"       --nx "${nx}"       --ny "${NY}"       --steps "${STEPS}"       --omega "${OMEGA}"       --rho0 "${RHO0}"       --lid-ux "${LID_UX}"       --lid-uy "${LID_UY}"       --write-field-every "${STEPS}"       --residual-every 500       --output-dir "${run_dir}"       > "${log_file}" 2>&1

    if ! grep -q "MPI ranks: ${np}," "${log_file}"; then
        echo "ERROR: np=${np} run did not report expected MPI rank count." >&2
        tail -n 60 "${log_file}" >&2 || true
        exit 3
    fi
done

python3 executables/milestone06/parse_weak_scaling.py   --log-dir "${OUTDIR}"   --nprocs "${NPROCS_CSV}"   --base-local-nx "${BASE_LOCAL_NX}"   --ny "${NY}"   --steps "${STEPS}"   --output-csv "${OUTDIR}/weak_scaling_metrics.csv"

python3 executables/milestone06/plot_weak_scaling.py   --csv "${OUTDIR}/weak_scaling_metrics.csv"   --outdir "${OUTDIR}"

echo "[gpu-weak-cmake] Completed."
echo "[gpu-weak-cmake] Wrote ${OUTDIR}/weak_scaling_metrics.csv"
