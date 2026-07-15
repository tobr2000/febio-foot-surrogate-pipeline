# Anatomical Foot Pilot Dataset

This pilot moves from the simple-foot generator to `templates/anatomic_knee_down_foot_smooth_v6.feb`.

## Default Pilot

- Dataset id: `anatomic_foot_v2_pilot`
- Samples: `1000`
- Array packets: `10`
- Samples per packet: `100`
- FEBio workers per packet: `16`
- Time steps: `80`
- Step size: `0.0125`
- History export: off by default

Submit on the cluster with:

```bash
sbatch --array=0-9 /path/to/febio-foot-surrogate-pipeline/scripts/slurm_anatomic_pilot.sh
```

## Generated Paths

- Base models: `templates/base_models/anatomic_foot_v2_pilot/base_00.feb` ... `base_11.feb`
- Base profile metadata: `templates/base_models/anatomic_foot_v2_pilot/base_model_profiles.json`
- Manifest: `data/datasets/anatomic_foot_v2_pilot/manifest.jsonl`
- Runs: `runs/anatomic_foot_v2_pilot`
- Shards: `shards/anatomic_foot_v2_pilot`

## Why History Is Off Initially

The anatomical model has about 56k nodes and 49k elements. Full history export for 80 time steps is large enough that the first pilot should check solver stability, contact-pressure distributions, base-model variation, and baseline regression difficulty before creating history-heavy PINN data.

Turn history on only for a small second pilot:

```bash
sbatch --array=0-0 --export=ALL,DATASET_ID=anatomic_foot_v2_history_smoke,MANIFEST_COUNT=100,PACKET_SIZE=100,INCLUDE_HISTORY=1 /path/to/febio-foot-surrogate-pipeline/scripts/slurm_anatomic_pilot.sh
```

## Important Notes

- The renderer detects anatomical names such as `AnatomicSoleContact` and `KneeCutDriveSurface`.
- The 12 base models preserve topology and named parts while morphing foot length, width, arch, leg length, toe splay, and ankle bend.
- The pilot varies material stiffness, friction, gait displacement, lateral sway, arch/roll terms, and regional sole pressure scale factors.
