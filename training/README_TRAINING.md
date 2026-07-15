# FEBio Foot Surrogate Training

This folder trains models directly from packed FEBio shards such as `shards/batch_039500_039999.npz`.

## Quick Data Check

```bash
python training/analyze_shards.py --shard-dir shards_test --out training/shards_test_report.json
```

The packed shard contains:

- element centroids plus stress components in `last_elements`
- nodal coordinates/displacements in `last_nodes`
- contact face gap and pressure in `last_contact`
- element von Mises stress in `last_element_von_mises`
- a mask for elements near the sole in `sole_near_element_mask`
- JSON parameters for each simulation in `params_json`

## Models

All three models predict two target groups:

- element fields: `sxx, syy, szz, sxy, syz, sxz, von_mises`
- contact fields: `gap, pressure`

Available model names:

- `gno`: graph neural operator style model over the element-centroid graph, plus a contact head
- `pinn`: coordinate-conditioned neural field with supervised loss plus equilibrium and von Mises consistency penalties
- `ffn`: pointwise feed-forward baseline over element/contact coordinates and simulation parameters

## Local Smoke Tests

```bash
python training/train_ffn.py --config training/foot_config.yaml --shard-dir shards_test --epochs 2 --batch-size 2 --max-samples 20 --wandb-mode disabled
python training/train_gno.py --config training/foot_config.yaml --shard-dir shards_test --epochs 2 --batch-size 1 --max-samples 10 --wandb-mode disabled
python training/train_pinn.py --config training/foot_config.yaml --shard-dir shards_test --epochs 2 --batch-size 1 --max-samples 10 --wandb-mode disabled
```

The primary training target is normalized contact pressure. The shared evaluator
reports contact-pressure field metrics, total reaction proxy metrics, peak
pressure metrics, and center-of-pressure metrics for all models.

## Cluster Training

The included SLURM script is a starting point:

```bash
sbatch training/run_training.slurm
sbatch --export=ALL,MODEL=ffn training/run_training.slurm
sbatch --export=ALL,MODEL=pinn training/run_training.slurm
```

For GPU throughput, the SLURM script defaults to larger batches for the lightweight
contact-only models and a smaller batch for PINN:

- `MODEL=gno` or `MODEL=ffn`: `BATCH_SIZE=256`
- `MODEL=pinn`: `BATCH_SIZE=8`

You can override these:

```bash
sbatch --export=ALL,MODEL=gno,BATCH_SIZE=512,NUM_WORKERS=6 training/run_training.slurm
```

Adjust the Python module, GPU request, or virtual environment path to match the cluster partition you use.
