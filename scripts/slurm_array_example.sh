#!/usr/bin/env bash
# Example only. Adjust paths/modules for your server.
#
# Submit 80 packets of 500 simulations for a 40,000-sample manifest:
#
#   sbatch --array=0-79 scripts/slurm_array_example.sh

#SBATCH --job-name=febio_foot
#SBATCH --partition=cpu
#SBATCH --account=cai_leg
#SBATCH --array=0-79
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/logs/febio_%A_%a.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/logs/febio_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
set -x

# ---------- Paths ----------
PROJECT_DIR="/path/to/febio-foot-surrogate-pipeline"
DATASET_ID="${DATASET_ID:-simplefoot_v2}"
BASE_PRESET="${BASE_PRESET:-simplefoot}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-templates/base_models}"
BASE_PROFILES="${BASE_PROFILES:-${BASE_MODEL_DIR}/base_model_profiles.json}"
BASE_TEMPLATE_SOURCE="${BASE_TEMPLATE_SOURCE:-templates/simplefoot_stance_ligamented_base.feb}"
LOG_DIR="${PROJECT_DIR}/logs"
RUN_DIR="${PROJECT_DIR}/runs/${DATASET_ID}"
SHARD_DIR="${PROJECT_DIR}/shards/${DATASET_ID}"
DATASET_DIR="${PROJECT_DIR}/data/datasets/${DATASET_ID}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_pipeline_py312}"
PIP_CACHE="${PIP_CACHE:-/path/to/pip-cache}"
VENV_LOCK="${VENV_DIR}.lock"
VENV_READY="${VENV_DIR}/.ready"

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${SHARD_DIR}" "${DATASET_DIR}" "${PIP_CACHE}"

# ---------- Threads ----------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ---------- Modules ----------
module purge
module load python/3.12.4

# ---------- Reusable Python environment ----------
# Array jobs start independently, so only one task should create/install the venv.
# Other tasks wait for the .ready marker and then reuse the same environment.
if [[ ! -f "${VENV_READY}" ]]; then
  if mkdir "${VENV_LOCK}" 2>/dev/null; then
    echo "[SETUP] Creating/updating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
    unset PYTHONUSERBASE PIP_TARGET PIP_PREFIX
    export PIP_CONFIG_FILE=/dev/null
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    python -m pip install -U pip wheel setuptools --cache-dir "${PIP_CACHE}" --no-input
    python -m pip install -r "${PROJECT_DIR}/requirements-cluster.txt" --cache-dir "${PIP_CACHE}" --no-input
    python - <<'PY'
import sys, numpy
print("[PYTHON]", sys.executable)
print("[NUMPY ]", numpy.__version__)
PY
    touch "${VENV_READY}"
    rmdir "${VENV_LOCK}"
  else
    echo "[WAIT] Another array task is preparing ${VENV_DIR}"
    until [[ -f "${VENV_READY}" ]]; do
      sleep 10
    done
  fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

PACKET_SIZE="${PACKET_SIZE:-500}"
SIM_WORKERS="${SIM_WORKERS:-16}"
START=$((SLURM_ARRAY_TASK_ID * PACKET_SIZE))
MANIFEST="${MANIFEST:-data/datasets/${DATASET_ID}/manifest.jsonl}"
MANIFEST_COUNT="${MANIFEST_COUNT:-40000}"
MANIFEST_LOCK="${DATASET_DIR}/manifest.lock"
INCLUDE_HISTORY="${INCLUDE_HISTORY:-1}"

# Keep each FEBio process mostly single-threaded when running many simulations
# concurrently inside one SLURM task.
export OMP_NUM_THREADS="${FEBIO_THREADS_PER_SIM:-1}"
export MKL_NUM_THREADS="${FEBIO_THREADS_PER_SIM:-1}"

cd "${PROJECT_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  if mkdir "${MANIFEST_LOCK}" 2>/dev/null; then
    echo "[SETUP] Manifest not found; generating ${MANIFEST}"
    python scripts/generate_manifest.py \
      --count "${MANIFEST_COUNT}" \
      --dataset-id "${DATASET_ID}" \
      --out "${MANIFEST}" \
      --preset "${BASE_PRESET}" \
      --base-profiles "${BASE_PROFILES}"
    rmdir "${MANIFEST_LOCK}"
  else
    echo "[WAIT] Another array task is generating ${MANIFEST}"
    until [[ -f "${MANIFEST}" ]]; do
      sleep 5
    done
  fi
fi

if [[ ! -f "${BASE_MODEL_DIR}/simplefoot_base_00.feb" || ! -f "${BASE_MODEL_DIR}/simplefoot_base_11.feb" ]]; then
  echo "[SETUP] Generating explicit base FEB templates in ${BASE_MODEL_DIR}"
  python scripts/generate_base_templates.py \
    --source "${BASE_TEMPLATE_SOURCE}" \
    --out-dir "${BASE_MODEL_DIR}" \
    --metadata "${BASE_PROFILES}" \
    --preset "${BASE_PRESET}" \
    --prefix simplefoot_base
fi

python scripts/validate_pinn_dataset_contract.py \
  --dataset-id "${DATASET_ID}" \
  --manifest "${MANIFEST}" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --base-profiles "${BASE_PROFILES}"

HISTORY_FLAG=()
if [[ "${INCLUDE_HISTORY}" == "1" || "${INCLUDE_HISTORY}" == "true" ]]; then
  HISTORY_FLAG=(--include-history)
fi

if [[ -z "${FEBIO_EXE:-}" && -f "${PROJECT_DIR}/.febio_exe" ]]; then
  FEBIO_EXE="$(cat "${PROJECT_DIR}/.febio_exe")"
fi
if [[ -z "${FEBIO_EXE:-}" ]]; then
  FEBIO_EXE="$(find "${PROJECT_DIR}/third_party/febio" -type f -name febio4 -perm -u+x 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${FEBIO_EXE:-}" ]]; then
  FEBIO_EXE="febio4"
fi

if ! command -v "${FEBIO_EXE}" >/dev/null 2>&1; then
  echo "[ERROR] FEBio executable not found: ${FEBIO_EXE}" >&2
  echo "[HINT] Install FEBio once with:" >&2
  echo "       cd ${PROJECT_DIR}" >&2
  echo "       bash scripts/install_febio_cluster.sh" >&2
  echo "[HINT] Or submit with --export=ALL,FEBIO_EXE=/path/to/febio4" >&2
  exit 3
fi

python scripts/run_batch.py \
  --dataset-id "${DATASET_ID}" \
  --manifest "${MANIFEST}" \
  --start "${START}" \
  --count "${PACKET_SIZE}" \
  --template "${BASE_MODEL_DIR}" \
  --runs "${RUN_DIR}" \
  --febio "${FEBIO_EXE}" \
  --workers "${SIM_WORKERS}" \
  "${HISTORY_FLAG[@]}" \
  --pack \
  --pack-dir "${SHARD_DIR}" \
  --cleanup
