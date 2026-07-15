from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


RANGES = {
    "E_flesh": (0.45, 1.55),
    "E_bone": (1500.0, 6500.0),
    "E_joint": (0.12, 0.75),
    "E_heel": (0.07, 0.48),
    "E_forefoot": (0.08, 0.58),
    "E_collar": (1.2, 11.0),
    "E_plantar": (2.5, 20.0),
    "E_achilles": (1.8, 17.0),
    "friction": (0.08, 0.75),
    "forward_disp": (0.08, 0.55),
    "early_down_disp": (-0.075, -0.015),
    "peak_down_disp": (-0.22, -0.055),
    "final_down_disp": (-0.13, -0.015),
    "peak_time": (0.38, 0.82),
    "scale_x": (0.90, 1.12),
    "scale_y": (0.82, 1.18),
    "scale_z": (0.90, 1.14),
    "lateral_disp": (-0.045, 0.045),
    "toe_off_bias": (-0.035, 0.055),
    "heel_toe_roll": (-0.045, 0.045),
    "arch_lift": (-0.03, 0.065),
}


BASE_MODEL_COUNT = 12
TRAIN_BASE_IDS = tuple(range(10))
UNSEEN_VALIDATION_BASE_IDS = (10, 11)


BASE_PROFILES = [
    {"base_foot_length": 0.96, "base_foot_width": 0.92, "base_arch_lift": 0.010, "base_leg_length": 0.95, "base_toe_splay": -0.015},
    {"base_foot_length": 1.02, "base_foot_width": 0.90, "base_arch_lift": 0.030, "base_leg_length": 1.08, "base_toe_splay": 0.005},
    {"base_foot_length": 1.08, "base_foot_width": 1.05, "base_arch_lift": -0.005, "base_leg_length": 0.98, "base_toe_splay": 0.030},
    {"base_foot_length": 0.92, "base_foot_width": 1.12, "base_arch_lift": 0.045, "base_leg_length": 1.02, "base_toe_splay": -0.025},
    {"base_foot_length": 1.14, "base_foot_width": 0.96, "base_arch_lift": 0.018, "base_leg_length": 1.14, "base_toe_splay": 0.020},
    {"base_foot_length": 0.98, "base_foot_width": 1.18, "base_arch_lift": -0.020, "base_leg_length": 0.90, "base_toe_splay": 0.045},
    {"base_foot_length": 1.05, "base_foot_width": 1.00, "base_arch_lift": 0.060, "base_leg_length": 1.18, "base_toe_splay": -0.005},
    {"base_foot_length": 0.88, "base_foot_width": 0.86, "base_arch_lift": 0.025, "base_leg_length": 1.00, "base_toe_splay": 0.010},
    {"base_foot_length": 1.18, "base_foot_width": 1.08, "base_arch_lift": 0.005, "base_leg_length": 0.94, "base_toe_splay": 0.035},
    {"base_foot_length": 0.94, "base_foot_width": 1.04, "base_arch_lift": 0.052, "base_leg_length": 1.10, "base_toe_splay": -0.040},
    {"base_foot_length": 1.12, "base_foot_width": 0.84, "base_arch_lift": 0.075, "base_leg_length": 0.88, "base_toe_splay": 0.055},
    {"base_foot_length": 0.86, "base_foot_width": 1.22, "base_arch_lift": -0.035, "base_leg_length": 1.22, "base_toe_splay": -0.060},
]


def sample_params(rng: random.Random, base_model_id: int) -> dict[str, float]:
    params = {key: rng.uniform(lo, hi) for key, (lo, hi) in RANGES.items()}
    params["base_model_id"] = float(base_model_id)
    params["base_is_training_family"] = float(base_model_id in TRAIN_BASE_IDS)
    params.update({key: float(value) for key, value in BASE_PROFILES[base_model_id].items()})
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--base-model-count", type=int, default=BASE_MODEL_COUNT)
    args = parser.parse_args()
    if args.base_model_count != BASE_MODEL_COUNT:
        raise ValueError(f"This generator currently defines exactly {BASE_MODEL_COUNT} base profiles.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.out.open("w", encoding="utf-8") as f:
        for sample_id in range(args.count):
            seed = rng.randrange(1, 2**31 - 1)
            row_rng = random.Random(seed)
            base_model_id = sample_id % BASE_MODEL_COUNT
            row = {
                "sample_id": sample_id,
                "seed": seed,
                "params": sample_params(row_rng, base_model_id),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
