# PINN Dataset Contract

This project now treats each FEBio dataset as a named, versioned dataset family. Use a stable
`DATASET_ID` such as `simplefoot_v2` or `anatomical_foot_v1` so multiple generated datasets can
coexist on the cluster without overwriting each other.

## Directory Layout

For `DATASET_ID=anatomical_foot_v1`, the expected layout is:

- `data/datasets/anatomical_foot_v1/manifest.jsonl`
- `runs/anatomical_foot_v1/sample_000000/...`
- `shards/anatomical_foot_v1/batch_000000_000499.npz`
- `training/dataset_quality/anatomical_foot_v1/...`

Training can still read the old flat `shards/` folder. New datasets should be passed with:

```bash
SHARD_DIR=shards/anatomical_foot_v1 sbatch training/run_training.slurm
```

## Base Model Contract

The base model folder must contain 12 FEB files with the same semantic parts:

- base ids `0..9`: training family
- base ids `10..11`: validation holdout family

Every base FEB must contain these materials:

- `Flesh_soft_tissue`
- `Cortical_bone_stiff`
- `Ankle_joint_pad`
- `Heel_pad_soft`
- `Forefoot_pad_soft`
- `Ankle_ligament_collar`
- `Plantar_fascia_band`
- `Achilles_like_band`

Every base FEB must contain these solid domains:

- `Flesh`
- `TibiaBone`
- `FootBone`
- `JointPad`
- `HeelPad`
- `ForefootPad`
- `AnkleCollar`
- `PlantarFascia`
- `AchillesBand`

Every base FEB must contain these surfaces:

- `FootSoleContact`
- `TibiaTopDriveSurface`

Every base FEB must contain load controllers `1` and `2` for forward and vertical stance drive.

## Packed Shard Contract

Required fields for the current contact-pressure models:

- `sample_ids`
- `sample_names`
- `dataset_ids`
- `params_json`
- `sole_near_element_mask`
- `last_nodes`
- `last_elements`
- `last_contact`
- `last_element_von_mises`

Additional fields required for the next full-PINN training path:

- `node_times`
- `element_times`
- `contact_times`
- `node_history`
- `element_history`
- `contact_history`

These history fields are produced by `scripts/run_batch.py --include-history`.

## Preflight Checks

Before launching a large array:

```bash
python scripts/validate_pinn_dataset_contract.py \
  --dataset-id anatomical_foot_v1 \
  --manifest data/datasets/anatomical_foot_v1/manifest.jsonl \
  --base-model-dir templates/base_models \
  --base-profiles templates/base_models/base_model_profiles.json
```

After packing shards, require the full-PINN history fields:

```bash
python scripts/validate_pinn_dataset_contract.py \
  --dataset-id anatomical_foot_v1 \
  --shard-dir shards/anatomical_foot_v1 \
  --require-history
```

## Top-Up Sampling

If some base models terminate less often than others, create a top-up manifest from existing packed
shards:

```bash
python scripts/plan_topup_manifest.py \
  --dataset-id anatomical_foot_v1 \
  --shard-dir shards/anatomical_foot_v1 \
  --existing-manifest data/datasets/anatomical_foot_v1/manifest.jsonl \
  --target-valid-per-base 2000 \
  --success-rate-assumption 0.42 \
  --out data/datasets/anatomical_foot_v1/manifest_topup_001.jsonl
```

Submit the top-up manifest with the same `DATASET_ID`; it writes into the same dataset namespace and
uses new sample IDs, so it does not overwrite the original samples.
