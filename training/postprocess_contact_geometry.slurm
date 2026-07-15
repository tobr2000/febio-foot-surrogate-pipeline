#!/bin/bash
#SBATCH --job-name=modelready
#SBATCH --account=cai_leg
#SBATCH --partition=cpu
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/training/logs/modelready_%j.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/training/logs/modelready_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
DATASET_ID="${DATASET_ID:-anatomic_v9_contact_v1}"
SHARD_DIR="${SHARD_DIR:-shards/${DATASET_ID}}"
OUT_DATASET_ID="${OUT_DATASET_ID:-${DATASET_ID}_modelready}"
OUT_SHARD_DIR="${OUT_SHARD_DIR:-shards/${OUT_DATASET_ID}}"
BASE_PROFILES="${BASE_PROFILES:-templates/base_models/anatomic_foot_v9_contact/base_model_profiles.json}"
SURFACE_NAME="${SURFACE_NAME:-AnatomicSoleContact}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_training_torch251_cu124_py312_v2}"
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-8}"
PRE_FLIGHT_ONLY="${PRE_FLIGHT_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"
ADD_CONTACT_GEOMETRY="${ADD_CONTACT_GEOMETRY:-1}"
REPACK_PINN_HISTORY="${REPACK_PINN_HISTORY:-1}"

cd "$PROJECT_DIR"
mkdir -p training/logs

module purge
module load python/3.12.4
source "$VENV_DIR/bin/activate"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "[JOB] PROJECT_DIR=${PROJECT_DIR}"
echo "[JOB] DATASET_ID=${DATASET_ID}"
echo "[JOB] SHARD_DIR=${SHARD_DIR}"
echo "[JOB] OUT_DATASET_ID=${OUT_DATASET_ID}"
echo "[JOB] OUT_SHARD_DIR=${OUT_SHARD_DIR}"
echo "[JOB] BASE_PROFILES=${BASE_PROFILES}"
echo "[JOB] SURFACE_NAME=${SURFACE_NAME}"
echo "[JOB] POSTPROCESS_WORKERS=${POSTPROCESS_WORKERS}"
echo "[JOB] ADD_CONTACT_GEOMETRY=${ADD_CONTACT_GEOMETRY}"
echo "[JOB] REPACK_PINN_HISTORY=${REPACK_PINN_HISTORY}"

ARGS=(
  --project-dir "$PROJECT_DIR"
  --shard-dir "$SHARD_DIR"
  --out-shard-dir "$OUT_SHARD_DIR"
  --base-profiles "$BASE_PROFILES"
  --surface-name "$SURFACE_NAME"
  --workers "$POSTPROCESS_WORKERS"
)

if [ "$PRE_FLIGHT_ONLY" = "1" ]; then
  ARGS+=(--preflight-only)
fi
if [ "$OVERWRITE" = "1" ]; then
  ARGS+=(--overwrite)
fi
if [ "$ADD_CONTACT_GEOMETRY" = "0" ]; then
  ARGS+=(--no-add-contact-geometry)
fi
if [ "$REPACK_PINN_HISTORY" = "0" ]; then
  ARGS+=(--no-repack-pinn-history)
fi

python -u training/postprocess_contact_geometry.py "${ARGS[@]}"
