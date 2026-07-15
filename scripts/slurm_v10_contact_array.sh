#!/usr/bin/env bash
# Submit the v10 anatomical contact dataset generation.
#
# Default layout:
#   8 array tasks x 300 samples = 2400 attempted samples
#
# Submit from the cluster project directory with:
#   sbatch scripts/slurm_v10_contact_array.sh

#SBATCH --job-name=febio_v10_contact
#SBATCH --partition=cpu
#SBATCH --account=cai_leg
#SBATCH --array=0-7
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=18:00:00
#SBATCH --chdir=/path/to/febio-foot-surrogate-pipeline
#SBATCH --output=/path/to/febio-foot-surrogate-pipeline/logs/febio_v10_%A_%a.out
#SBATCH --error=/path/to/febio-foot-surrogate-pipeline/logs/febio_v10_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
set -x

export DATASET_ID="${DATASET_ID:-anatomic_v10_contact_v1}"
export BASE_PRESET="${BASE_PRESET:-anatomic_v10_contact}"
export BASE_MODEL_DIR="${BASE_MODEL_DIR:-templates/base_models/anatomic_foot_v10_contact}"
export BASE_PROFILES="${BASE_PROFILES:-${BASE_MODEL_DIR}/base_model_profiles.json}"
export BASE_TEMPLATE_SOURCE="${BASE_TEMPLATE_SOURCE:-templates/anatomic_knee_down_foot_smooth_v10_contact.feb}"
export PACKET_SIZE="${PACKET_SIZE:-300}"
export MANIFEST_COUNT="${MANIFEST_COUNT:-2400}"
export SIM_WORKERS="${SIM_WORKERS:-16}"
export INCLUDE_HISTORY="${INCLUDE_HISTORY:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/slurm_array_example.sh"
