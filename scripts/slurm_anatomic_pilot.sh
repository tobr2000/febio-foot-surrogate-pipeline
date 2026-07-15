#!/usr/bin/env bash
# Pilot generator for the anatomical knee-down foot model.
#
# Submit the default 1000-sample pilot as 10 packets of 100:
#
#   sbatch --array=0-9 /path/to/febio-foot-surrogate-pipeline/scripts/slurm_anatomic_pilot.sh

#SBATCH --job-name=febio_anatomic
#SBATCH --partition=cpu
#SBATCH --account=cai_leg
#SBATCH --array=0-9
#SBATCH --cpus-per-task=80
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/logs/anatomic_%A_%a.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/logs/anatomic_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
set -x

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
DATASET_ID="${DATASET_ID:-anatomic_foot_v2_pilot}"
LOG_DIR="${PROJECT_DIR}/logs"
RUN_DIR="${PROJECT_DIR}/runs/${DATASET_ID}"
SHARD_DIR="${PROJECT_DIR}/shards/${DATASET_ID}"
DATASET_DIR="${PROJECT_DIR}/data/datasets/${DATASET_ID}"
BASE_MODEL_DIR="${PROJECT_DIR}/templates/base_models/${DATASET_ID}"
BASE_PROFILE_PATH="${BASE_MODEL_DIR}/base_model_profiles.json"
SOURCE_TEMPLATE="${SOURCE_TEMPLATE:-templates/anatomic_knee_down_foot_smooth_v6.feb}"
GENERATION_PRESET="${GENERATION_PRESET:-anatomic_pilot}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_pipeline_py312}"
PIP_CACHE="${PIP_CACHE:-/path/to/pip-cache}"
VENV_LOCK="${VENV_DIR}.lock"
VENV_READY="${VENV_DIR}/.ready"

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${SHARD_DIR}" "${DATASET_DIR}" "${BASE_MODEL_DIR}" "${PIP_CACHE}"

module purge
module load python/3.12.4

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

PACKET_SIZE="${PACKET_SIZE:-100}"
SIM_WORKERS="${SIM_WORKERS:-16}"
MANIFEST_COUNT="${MANIFEST_COUNT:-1000}"
MANIFEST="${MANIFEST:-data/datasets/${DATASET_ID}/manifest.jsonl}"
MANIFEST_LOCK="${DATASET_DIR}/manifest.lock"
BASE_LOCK="${BASE_MODEL_DIR}.lock"
TIME_STEPS="${TIME_STEPS:-120}"
STEP_SIZE="${STEP_SIZE:-0.008333333333333333}"
INCLUDE_HISTORY="${INCLUDE_HISTORY:-0}"
FORCE_REGENERATE_BASES="${FORCE_REGENERATE_BASES:-0}"
FORCE_REGENERATE_MANIFEST="${FORCE_REGENERATE_MANIFEST:-0}"
START=$((SLURM_ARRAY_TASK_ID * PACKET_SIZE))

export OMP_NUM_THREADS="${FEBIO_THREADS_PER_SIM:-1}"
export MKL_NUM_THREADS="${FEBIO_THREADS_PER_SIM:-1}"

cd "${PROJECT_DIR}"

echo "[CONFIG] packet_size=${PACKET_SIZE} sim_workers=${SIM_WORKERS} febio_threads_per_sim=${OMP_NUM_THREADS}"
echo "[CONFIG] time_steps=${TIME_STEPS} step_size=${STEP_SIZE} dataset_id=${DATASET_ID} preset=${GENERATION_PRESET}"
echo "[CONFIG] force_regenerate_bases=${FORCE_REGENERATE_BASES} force_regenerate_manifest=${FORCE_REGENERATE_MANIFEST}"

if [[ ! -f "${SOURCE_TEMPLATE}" ]]; then
  echo "[ERROR] Anatomical source template is missing: ${PROJECT_DIR}/${SOURCE_TEMPLATE}" >&2
  echo "[HINT] Upload/copy anatomic_knee_down_foot_smooth_v6.feb into ${PROJECT_DIR}/templates before submitting." >&2
  exit 4
fi

if [[ -d "${BASE_LOCK}" && ! -f "${BASE_MODEL_DIR}/simplefoot_base_00.feb" && ! -f "${BASE_MODEL_DIR}/simplefoot_base_11.feb" ]]; then
  if find "${BASE_LOCK}" -maxdepth 0 -mmin +30 | grep -q .; then
    echo "[WARN] Removing stale base-template lock older than 30 minutes: ${BASE_LOCK}"
    rmdir "${BASE_LOCK}" 2>/dev/null || true
  fi
fi

if [[ "${FORCE_REGENERATE_BASES}" == "1" || "${FORCE_REGENERATE_BASES}" == "true" || ! -f "${BASE_MODEL_DIR}/simplefoot_base_00.feb" || ! -f "${BASE_MODEL_DIR}/simplefoot_base_11.feb" ]]; then
  if mkdir "${BASE_LOCK}" 2>/dev/null; then
    cleanup_base_lock() {
      rmdir "${BASE_LOCK}" 2>/dev/null || true
    }
    trap cleanup_base_lock EXIT
    echo "[SETUP] Generating anatomical base FEB templates in ${BASE_MODEL_DIR}"
    python scripts/generate_base_templates.py \
      --preset "${GENERATION_PRESET}" \
      --source "${SOURCE_TEMPLATE}" \
      --out-dir "${BASE_MODEL_DIR}" \
      --metadata "${BASE_PROFILE_PATH}" \
      --prefix simplefoot_base
    rmdir "${BASE_LOCK}"
    trap - EXIT
  else
    echo "[WAIT] Another array task is generating ${BASE_MODEL_DIR}"
    WAIT_SECONDS=0
    until [[ -f "${BASE_MODEL_DIR}/simplefoot_base_00.feb" && -f "${BASE_MODEL_DIR}/simplefoot_base_11.feb" ]]; do
      if (( WAIT_SECONDS >= 1800 )); then
        echo "[ERROR] Timed out waiting for base templates after ${WAIT_SECONDS}s." >&2
        echo "[HINT] Check/remove stale lock directory: ${BASE_LOCK}" >&2
        exit 5
      fi
      sleep 10
      WAIT_SECONDS=$((WAIT_SECONDS + 10))
    done
  fi
fi

if [[ "${FORCE_REGENERATE_MANIFEST}" == "1" || "${FORCE_REGENERATE_MANIFEST}" == "true" || ! -f "${MANIFEST}" ]]; then
  if mkdir "${MANIFEST_LOCK}" 2>/dev/null; then
    echo "[SETUP] Manifest not found; generating ${MANIFEST}"
    python scripts/generate_manifest.py \
      --preset "${GENERATION_PRESET}" \
      --count "${MANIFEST_COUNT}" \
      --dataset-id "${DATASET_ID}" \
      --base-profiles "${BASE_PROFILE_PATH}" \
      --out "${MANIFEST}"
    rmdir "${MANIFEST_LOCK}"
  else
    echo "[WAIT] Another array task is generating ${MANIFEST}"
    WAIT_SECONDS=0
    until [[ -f "${MANIFEST}" ]]; do
      if (( WAIT_SECONDS >= 1800 )); then
        echo "[ERROR] Timed out waiting for manifest after ${WAIT_SECONDS}s." >&2
        echo "[HINT] Check/remove stale lock directory: ${MANIFEST_LOCK}" >&2
        exit 6
      fi
      sleep 5
      WAIT_SECONDS=$((WAIT_SECONDS + 5))
    done
  fi
fi

python scripts/validate_pinn_dataset_contract.py \
  --dataset-id "${DATASET_ID}" \
  --manifest "${MANIFEST}" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --base-profiles "${BASE_PROFILE_PATH}"

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
  --time-steps "${TIME_STEPS}" \
  --step-size "${STEP_SIZE}" \
  "${HISTORY_FLAG[@]}" \
  --pack \
  --pack-dir "${SHARD_DIR}" \
  --cleanup
