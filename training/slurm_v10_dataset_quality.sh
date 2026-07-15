#!/usr/bin/env bash
# Run dataset-quality analysis for the v10 contact shards.

#SBATCH --job-name=v10_quality
#SBATCH --partition=cpu
#SBATCH --account=cai_leg
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/training/logs/v10_quality_%j.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/training/logs/v10_quality_%j.err

set -euo pipefail
set -x

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
DATASET_ID="${DATASET_ID:-anatomic_v10_contact_v1}"
SHARD_DIR="${SHARD_DIR:-shards/${DATASET_ID}}"
RUNS_DIR="${RUNS_DIR:-runs/${DATASET_ID}}"
OUT_DIR="${OUT_DIR:-training/dataset_quality/${DATASET_ID}}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_pipeline_py312}"
PIP_CACHE="${PIP_CACHE:-/path/to/pip-cache}"
VENV_LOCK="${VENV_DIR}.lock"
VENV_READY="${VENV_DIR}/.ready"

cd "${PROJECT_DIR}"
mkdir -p training/logs "${PIP_CACHE}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
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
    python -m pip install -r requirements-cluster.txt --cache-dir "${PIP_CACHE}" --no-input
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

python training/analyze_dataset_quality.py \
  --dataset-id "${DATASET_ID}" \
  --shard-dir "${SHARD_DIR}" \
  --runs-dir "${RUNS_DIR}" \
  --out-dir "${OUT_DIR}" \
  --require-history
