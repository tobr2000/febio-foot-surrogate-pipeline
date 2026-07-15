#!/usr/bin/env bash
# Recompute report-ready dataset quality and regression baselines for all FEBio
# foot/contact dataset lineages used in the VT2 report.

#SBATCH --job-name=report_reg_all
#SBATCH --partition=cpu
#SBATCH --account=cai_leg
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/training/logs/report_reg_all_%j.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/training/logs/report_reg_all_%j.err

set -euo pipefail
set -x

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_pipeline_py312}"
PIP_CACHE="${PIP_CACHE:-/path/to/pip-cache}"
VENV_LOCK="${VENV_DIR}.lock"
VENV_READY="${VENV_DIR}/.ready"
MAX_REGRESSION_SAMPLES="${MAX_REGRESSION_SAMPLES:-0}"
REQUIRE_HISTORY_DEFAULT="${REQUIRE_HISTORY_DEFAULT:-0}"

cd "${PROJECT_DIR}"
mkdir -p training/logs "${PIP_CACHE}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

module purge
module load python/3.12.4

if [[ ! -f "${VENV_READY}" ]]; then
  if mkdir "${VENV_LOCK}" 2>/dev/null; then
    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
    python -m pip install -U pip wheel setuptools --cache-dir "${PIP_CACHE}" --no-input
    if [[ -f requirements-cluster.txt ]]; then
      python -m pip install -r requirements-cluster.txt --cache-dir "${PIP_CACHE}" --no-input
    else
      python -m pip install numpy --cache-dir "${PIP_CACHE}" --no-input
    fi
    touch "${VENV_READY}"
    rmdir "${VENV_LOCK}"
  else
    until [[ -f "${VENV_READY}" ]]; do
      sleep 10
    done
  fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

run_quality() {
  local dataset_id="$1"
  local shard_dir="$2"
  local runs_dir="$3"
  local out_dir="$4"
  local require_history="${5:-${REQUIRE_HISTORY_DEFAULT}}"

  local history_flag=()
  if [[ "${require_history}" == "1" || "${require_history}" == "true" ]]; then
    history_flag=(--require-history)
  fi

  local dataset_flag=()
  if [[ -n "${dataset_id}" ]]; then
    dataset_flag=(--dataset-id "${dataset_id}")
  fi

  echo "[RUN] dataset_id=${dataset_id:-<none>} shard_dir=${shard_dir} out_dir=${out_dir}"
  python -u training/analyze_dataset_quality.py \
    "${dataset_flag[@]}" \
    --shard-dir "${shard_dir}" \
    --runs-dir "${runs_dir}" \
    --out-dir "${out_dir}" \
    --max-regression-samples "${MAX_REGRESSION_SAMPLES}" \
    "${history_flag[@]}"
}

# Historical simplified foot lineage used the default training/dataset_quality
# location in earlier report artifacts. Keep its output in a named folder now.
run_quality \
  "" \
  "shards" \
  "runs" \
  "training/dataset_quality/simplified_foot_report" \
  "0"

run_quality \
  "anatomic_foot_v2_pilot_stable3_1536_w64" \
  "shards/anatomic_foot_v2_pilot_stable3_1536_w64" \
  "runs/anatomic_foot_v2_pilot_stable3_1536_w64" \
  "training/dataset_quality/anatomic_foot_v2_pilot_stable3_1536_w64" \
  "0"

run_quality \
  "anatomic_v9_contact_v1" \
  "shards/anatomic_v9_contact_v1" \
  "runs/anatomic_v9_contact_v1" \
  "training/dataset_quality/anatomic_v9_contact_v1" \
  "1"

run_quality \
  "anatomic_v10_contact_v1" \
  "shards/anatomic_v10_contact_v1" \
  "runs/anatomic_v10_contact_v1" \
  "training/dataset_quality/anatomic_v10_contact_v1" \
  "1"

echo "[OK] Report regression/data-quality jobs completed."
