#!/bin/bash
#SBATCH --job-name=foot_surrogate
#SBATCH --account=cai_leg
#SBATCH --partition=gpu
#SBATCH --output=/cluster/projects/2026_stmd_orthosense/febio_ml_pipeline/training/logs/febio_%A_%a.out
#SBATCH --error=/cluster/projects/2026_stmd_orthosense/febio_ml_pipeline/training/logs/febio_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/cluster/projects/2026_stmd_orthosense/febio_ml_pipeline}"
VENV_DIR="${VENV_DIR:-/cluster/home/brandtob/.venvs/febio_ml_training_torch251_cu124_py312_v2}"
PIP_CACHE="${PIP_CACHE:-/cluster/home/brandtob/.pip-cache}"
SETUP_LOCK="${SETUP_LOCK:-${VENV_DIR}.lock}"
READY_MARKER="${READY_MARKER:-${VENV_DIR}.ready}"
MODEL="${MODEL:-gno}"
SHARD_DIR="${SHARD_DIR:-shards}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TIMING_LOG_EVERY="${TIMING_LOG_EVERY:-100}"

if [ -z "${BATCH_SIZE:-}" ]; then
  if [ "$MODEL" = "pinn" ]; then
    BATCH_SIZE=8
  else
    BATCH_SIZE=256
  fi
fi

cd "$PROJECT_DIR"
mkdir -p training/logs "$PIP_CACHE"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

(
  flock -x 200

  ENV_OK=0
  if [ -f "$READY_MARKER" ]; then
    python - <<'PY' && ENV_OK=1 || ENV_OK=0
import torch
import yaml
import numpy
import wandb
raise SystemExit(0 if torch.__version__ == "2.5.1+cu124" else 1)
PY
  fi

  if [ "$ENV_OK" -eq 1 ]; then
    echo "[SETUP ] environment already prepared; skipping pip install"
  else
    python -m pip install --upgrade pip wheel setuptools --cache-dir "$PIP_CACHE" --no-input

    TORCH_OK=0
    python - <<'PY' && TORCH_OK=1 || TORCH_OK=0
import torch
raise SystemExit(0 if torch.__version__ == "2.5.1+cu124" else 1)
PY

    if [ "$TORCH_OK" -ne 1 ]; then
      # The Dallas GPU partition can assign Tesla V100 GPUs (compute capability 7.0).
      # Newer PyTorch/CUDA 13 wheels do not include sm70 kernels, causing:
      # "CUDA error: no kernel image is available for execution on the device".
      # Torch 2.5.1 + CUDA 12.4 still supports this hardware generation.
      python -m pip install \
        --upgrade \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 \
        --cache-dir "$PIP_CACHE" \
        --no-input
    else
      echo "[SETUP ] torch 2.5.1+cu124 already installed; skipping torch install"
    fi

    python -m pip install \
      --upgrade \
      -r training/requirements-training.txt \
      --cache-dir "$PIP_CACHE" \
      --no-input

    python - <<'PY'
import torch
import yaml
import numpy
import wandb
raise SystemExit(0 if torch.__version__ == "2.5.1+cu124" else 1)
PY
    touch "$READY_MARKER"
  fi
) 200>"$SETUP_LOCK"

python - <<'PY'
import sys
import torch
print("[PYTHON]", sys.executable)
print("[TORCH ]", torch.__version__)
print("[CUDA  ] available?", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[GPU   ]", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY

python -u "training/train_${MODEL}.py" \
  --config training/foot_config.yaml \
  --shard-dir "$SHARD_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --timing-log-every "$TIMING_LOG_EVERY" \
  --wandb-mode "$WANDB_MODE"
