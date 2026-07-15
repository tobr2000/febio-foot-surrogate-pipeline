#!/bin/bash
#SBATCH --job-name=foot_quality
#SBATCH --account=cai_leg
#SBATCH --partition=cpu
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/training/logs/dataset_quality_%j.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/training/logs/dataset_quality_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
VENV_DIR="${VENV_DIR:-/path/to/venvs/febio_ml_quality_py312}"
PIP_CACHE="${PIP_CACHE:-/path/to/pip-cache}"
DATASET_ID="${DATASET_ID:-}"
if [ -n "$DATASET_ID" ] && [ -z "${SHARD_DIR:-}" ]; then
  SHARD_DIR="shards/${DATASET_ID}"
else
  SHARD_DIR="${SHARD_DIR:-shards}"
fi
if [ -n "$DATASET_ID" ] && [ -z "${RUNS_DIR:-}" ]; then
  RUNS_DIR="runs/${DATASET_ID}"
else
  RUNS_DIR="${RUNS_DIR:-runs}"
fi
if [ -n "$DATASET_ID" ] && [ -z "${OUT_DIR:-}" ]; then
  OUT_DIR="training/dataset_quality/${DATASET_ID}"
else
  OUT_DIR="${OUT_DIR:-training/dataset_quality}"
fi
MAX_REGRESSION_SAMPLES="${MAX_REGRESSION_SAMPLES:-12000}"
REQUIRE_HISTORY="${REQUIRE_HISTORY:-0}"
PACKET_SIZE_ANALYSIS="${PACKET_SIZE_ANALYSIS:-0}"
PACKET_WORKERS="${PACKET_WORKERS:-64}"
PACKET_ARRAY_CONCURRENCY="${PACKET_ARRAY_CONCURRENCY:-4}"
PACKET_MIN="${PACKET_MIN:-64}"
PACKET_MAX="${PACKET_MAX:-256}"
PACKET_BOOTSTRAP="${PACKET_BOOTSTRAP:-500}"
PACKET_OVERHEAD_SEC="${PACKET_OVERHEAD_SEC:-0}"
PACKET_TARGET_ATTEMPTS="${PACKET_TARGET_ATTEMPTS:-}"
PACKET_TARGET_SUCCESSES="${PACKET_TARGET_SUCCESSES:-}"
PACKET_RUNTIME_INPUTS="${PACKET_RUNTIME_INPUTS:-}"

cd "$PROJECT_DIR"
mkdir -p training/logs "$PIP_CACHE"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

module purge
module load python/3.12.4

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE PIP_TARGET PIP_PREFIX
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip wheel setuptools --cache-dir "$PIP_CACHE" --no-input
python -m pip install --upgrade numpy --cache-dir "$PIP_CACHE" --no-input

HISTORY_FLAG=()
if [ "$REQUIRE_HISTORY" = "1" ] || [ "$REQUIRE_HISTORY" = "true" ]; then
  HISTORY_FLAG=(--require-history)
fi

PACKET_FLAGS=()
if [ "$PACKET_SIZE_ANALYSIS" = "1" ] || [ "$PACKET_SIZE_ANALYSIS" = "true" ]; then
  PACKET_FLAGS=(
    --packet-size-analysis
    --packet-workers "$PACKET_WORKERS"
    --packet-array-concurrency "$PACKET_ARRAY_CONCURRENCY"
    --packet-min "$PACKET_MIN"
    --packet-max "$PACKET_MAX"
    --packet-bootstrap "$PACKET_BOOTSTRAP"
    --packet-overhead-sec "$PACKET_OVERHEAD_SEC"
  )
  if [ -n "$PACKET_TARGET_ATTEMPTS" ]; then
    PACKET_FLAGS+=(--packet-target-attempts "$PACKET_TARGET_ATTEMPTS")
  fi
  if [ -n "$PACKET_TARGET_SUCCESSES" ]; then
    PACKET_FLAGS+=(--packet-target-successes "$PACKET_TARGET_SUCCESSES")
  fi
  if [ -n "$PACKET_RUNTIME_INPUTS" ]; then
    for packet_input in $PACKET_RUNTIME_INPUTS; do
      PACKET_FLAGS+=(--packet-runtime-input "$packet_input")
    done
  fi
fi

DATASET_FLAG=()
if [ -n "$DATASET_ID" ]; then
  DATASET_FLAG=(--dataset-id "$DATASET_ID")
fi

python -u training/analyze_dataset_quality.py \
  "${DATASET_FLAG[@]}" \
  --shard-dir "$SHARD_DIR" \
  --runs-dir "$RUNS_DIR" \
  --out-dir "$OUT_DIR" \
  "${HISTORY_FLAG[@]}" \
  "${PACKET_FLAGS[@]}" \
  --max-regression-samples "$MAX_REGRESSION_SAMPLES"
