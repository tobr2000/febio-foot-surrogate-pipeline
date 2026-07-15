from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_manifest import load_base_profiles, sample_params


def shard_counts(shard_dir: Path) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    max_id = -1
    for path in sorted(shard_dir.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sample_ids = data["sample_ids"].astype(int)
            max_id = max(max_id, int(sample_ids.max(initial=max_id)))
            params_json = data["params_json"].astype(str) if "params_json" in data.files else np.asarray(["{}"] * len(sample_ids))
            for text in params_json:
                params = json.loads(str(text))
                counts[int(round(float(params.get("base_model_id", 0.0))))] += 1
    return counts, max_id


def manifest_max_id(path: Path) -> int:
    if not path.exists():
        return -1
    max_id = -1
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                max_id = max(max_id, int(json.loads(line)["sample_id"]))
    return max_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifest that tops up underrepresented valid base models.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-valid-per-base", type=int, required=True)
    parser.add_argument("--success-rate-assumption", type=float, default=0.42)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--base-profiles", type=Path, default=Path("templates/base_models/base_model_profiles.json"))
    parser.add_argument("--base-model-count", type=int, default=12)
    args = parser.parse_args()

    valid_counts, max_packed_id = shard_counts(args.shard_dir)
    max_manifest_id = manifest_max_id(args.existing_manifest) if args.existing_manifest else -1
    next_sample_id = max(max_packed_id, max_manifest_id) + 1
    base_profiles = load_base_profiles(args.base_profiles)
    rng = random.Random(args.seed)

    rows = []
    requested_by_base = {}
    for base_id in range(args.base_model_count):
        shortfall = max(0, args.target_valid_per_base - valid_counts.get(base_id, 0))
        requested = int(np.ceil(shortfall / max(1e-6, args.success_rate_assumption)))
        requested_by_base[base_id] = requested
        for _ in range(requested):
            seed = rng.randrange(1, 2**31 - 1)
            row_rng = random.Random(seed)
            row = {
                "dataset_id": args.dataset_id,
                "sample_id": next_sample_id,
                "sample_name": f"sample_{next_sample_id:06d}",
                "seed": seed,
                "topup_for_base_model_id": base_id,
                "params": sample_params(row_rng, base_id, base_profiles),
            }
            rows.append(row)
            next_sample_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "dataset_id": args.dataset_id,
        "valid_counts_by_base": dict(sorted(valid_counts.items())),
        "target_valid_per_base": args.target_valid_per_base,
        "success_rate_assumption": args.success_rate_assumption,
        "requested_by_base": requested_by_base,
        "topup_rows": len(rows),
        "out": str(args.out),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
