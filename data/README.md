# Dataset access

The complete simulation datasets are intentionally excluded from Git. The
cluster inventory found the following principal prepared lineages:

| Lineage | Approximate size |
|---|---:|
| Simplified-foot root shards | several GB across root packet files |
| V9 model-ready | 244.5 GiB |
| V10 model-ready | 481.9 GiB |
| Anatomical-foot v1 | 123.7 GiB |
| Anatomical pilot | 4.7 GiB |

Before public release, add one versioned manifest per published lineage with a
stable download URL, SHA-256 checksums, byte sizes, schema version, licence,
and the command used to generate the data. Small redistributable samples may be
placed in `samples/`; complete shards belong in a research-data repository.

Before building multi-hour archives, estimate ZIP-compatible compression from
a stratified sample:

```bash
python scripts/estimate_dataset_compression.py \
  /path/to/v9_modelready /path/to/v10_modelready \
  --sample-gib 5 --output compression_estimate.json
```

Many shards are already compressed `.npz` files, so a second ZIP layer may save
very little. The estimator projects the full archive size without writing or
modifying the source dataset.
