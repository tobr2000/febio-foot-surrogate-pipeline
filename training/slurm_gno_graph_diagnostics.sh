#!/bin/bash
#SBATCH --job-name=gno_graph_diag
#SBATCH --account=cai_leg
#SBATCH --partition=cpu
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/training/logs/gno_graph_diag_%j.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/training/logs/gno_graph_diag_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
DATASET_ID="${DATASET_ID:-anatomic_v10_contact_v1}"
SHARD_DIR="${SHARD_DIR:-shards_modelready/${DATASET_ID}}"
OUT_DIR="${OUT_DIR:-training/graph_diagnostics/${DATASET_ID}}"
K_LIST="${K_LIST:-4,6,8,10,12}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
SEED="${SEED:-42}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_training_torch251_cu124_py312_v2}"

cd "$PROJECT_DIR"
mkdir -p training/logs "$OUT_DIR"

module purge
module load python/3.12.4
source "$VENV_DIR/bin/activate"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "[JOB] PROJECT_DIR=${PROJECT_DIR}"
echo "[JOB] DATASET_ID=${DATASET_ID}"
echo "[JOB] SHARD_DIR=${SHARD_DIR}"
echo "[JOB] OUT_DIR=${OUT_DIR}"
echo "[JOB] K_LIST=${K_LIST}"
echo "[JOB] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[JOB] SEED=${SEED}"

python -u training/diagnose_gno_graph.py \
  --shard-dir "$SHARD_DIR" \
  --out-dir "$OUT_DIR" \
  --k-list "$K_LIST" \
  --max-samples "$MAX_SAMPLES" \
  --seed "$SEED"
