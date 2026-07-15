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
    "heel_pressure_scale": (0.70, 1.35),
    "midfoot_pressure_scale": (0.65, 1.40),
    "forefoot_pressure_scale": (0.70, 1.45),
    "toe_pressure_scale": (0.55, 1.55),
}

ANATOMIC_PILOT_RANGES = {
    "E_flesh": (0.45, 1.65),
    "E_bone": (2500.0, 8500.0),
    "E_joint": (6.0, 28.0),
    "E_heel": (0.12, 0.34),
    "E_forefoot": (0.09, 0.24),
    "E_collar": (35.0, 180.0),
    "E_plantar": (120.0, 650.0),
    "E_achilles": (120.0, 700.0),
    "friction": (0.18, 0.45),
    "forward_disp": (0.018, 0.042),
    "early_down_disp": (-0.0035, -0.0010),
    "peak_down_disp": (-0.014, -0.0055),
    "final_down_disp": (-0.010, -0.0035),
    "peak_time": (0.45, 0.72),
    "scale_x": (1.0, 1.0),
    "scale_y": (1.0, 1.0),
    "scale_z": (1.0, 1.0),
    "lateral_disp": (0.0, 0.0),
    "toe_off_bias": (0.0, 0.0),
    "heel_toe_roll": (0.0, 0.0),
    "arch_lift": (0.0, 0.0),
    "heel_pressure_scale": (0.85, 1.15),
    "midfoot_pressure_scale": (0.85, 1.18),
    "forefoot_pressure_scale": (0.85, 1.18),
    "toe_pressure_scale": (0.80, 1.20),
}

ANATOMIC_FAST_V7_RANGES = {
    "E_flesh": (0.35, 1.90),
    "E_bone": (1800.0, 9000.0),
    "E_joint": (4.0, 34.0),
    "E_heel": (0.07, 0.45),
    "E_forefoot": (0.06, 0.34),
    "E_collar": (20.0, 220.0),
    "E_plantar": (70.0, 780.0),
    "E_achilles": (70.0, 850.0),
    "friction": (0.12, 0.58),
    "forward_force_per_node": (0.00006, 0.00020),
    "down_force_per_node": (0.00020, 0.00055),
    "forward_disp": (0.0, 0.0),
    "early_down_disp": (0.0, 0.0),
    "peak_down_disp": (0.0, 0.0),
    "final_down_disp": (0.0, 0.0),
    "peak_time": (0.55, 0.84),
    "scale_x": (0.992, 1.008),
    "scale_y": (0.988, 1.012),
    "scale_z": (0.994, 1.006),
    "lateral_disp": (0.0, 0.0),
    "toe_off_bias": (0.0, 0.0),
    "heel_toe_roll": (0.0, 0.0),
    "arch_lift": (0.0, 0.0015),
}

ANATOMIC_V9_CONTACT_RANGES = {
    "E_flesh": (0.45, 1.55),
    "E_bone": (1800.0, 8000.0),
    "E_joint": (4.0, 26.0),
    "E_heel": (0.10, 0.34),
    "E_forefoot": (0.09, 0.30),
    "E_collar": (25.0, 180.0),
    "E_plantar": (90.0, 650.0),
    "E_achilles": (90.0, 720.0),
    "friction": (0.04, 0.20),
    "forward_force_per_node": (0.0000000040, 0.0000000090),
    "down_force_per_node": (0.000000014, 0.000000026),
    "force_ramp_mid": (0.20, 0.40),
    "force_ramp_final": (0.36, 0.52),
    "forward_disp": (0.0, 0.0),
    "early_down_disp": (0.0, 0.0),
    "peak_down_disp": (0.0, 0.0),
    "final_down_disp": (0.0, 0.0),
    "peak_time": (0.48, 0.84),
    "scale_x": (0.997, 1.003),
    "scale_y": (0.996, 1.004),
    "scale_z": (0.998, 1.003),
    "lateral_disp": (0.0, 0.0),
    "toe_off_bias": (-0.0005, 0.0012),
    "heel_toe_roll": (-0.0008, 0.0012),
    "arch_lift": (-0.0008, 0.0015),
}

ANATOMIC_V10_CONTACT_RANGES = {
    **ANATOMIC_V9_CONTACT_RANGES,
    "friction": (0.03, 0.28),
    "forward_force_per_node": (0.0000000030, 0.0000000110),
    "down_force_per_node": (0.000000012, 0.000000030),
    "force_ramp_mid": (0.16, 0.46),
    "force_ramp_final": (0.32, 0.62),
    "scale_x": (0.995, 1.005),
    "scale_y": (0.993, 1.007),
    "scale_z": (0.997, 1.004),
    "toe_off_bias": (-0.0012, 0.0022),
    "heel_toe_roll": (-0.0015, 0.0020),
    "arch_lift": (-0.0015, 0.0025),
    "insole_medial_bias": (-0.0008, 0.0014),
    "insole_lateral_bias": (-0.0008, 0.0014),
    "insole_heel_lift": (-0.0007, 0.0022),
    "insole_forefoot_lift": (-0.0007, 0.0024),
    "insole_ridge_amp": (-0.0010, 0.0020),
    "insole_pocket_amp": (-0.0005, 0.0022),
    **{
        f"insole_h_{ix:02d}_{iy:02d}": (-0.0009, 0.0012)
        for ix in range(6)
        for iy in range(3)
    },
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

ANATOMIC_BASE_PROFILES = [
    {"base_foot_length": 1.000, "base_foot_width": 1.000, "base_arch_lift": 0.000, "base_leg_length": 1.000, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.002, "base_foot_width": 0.998, "base_arch_lift": 0.000, "base_leg_length": 1.001, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 0.998, "base_foot_width": 1.002, "base_arch_lift": 0.000, "base_leg_length": 0.999, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.003, "base_foot_width": 1.000, "base_arch_lift": 0.000, "base_leg_length": 1.000, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 0.997, "base_foot_width": 1.000, "base_arch_lift": 0.000, "base_leg_length": 1.000, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.000, "base_foot_width": 1.003, "base_arch_lift": 0.000, "base_leg_length": 1.001, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.000, "base_foot_width": 0.997, "base_arch_lift": 0.000, "base_leg_length": 0.999, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.002, "base_foot_width": 1.002, "base_arch_lift": 0.000, "base_leg_length": 1.000, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 0.998, "base_foot_width": 0.998, "base_arch_lift": 0.000, "base_leg_length": 1.000, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.001, "base_foot_width": 0.999, "base_arch_lift": 0.000, "base_leg_length": 1.002, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 0.999, "base_foot_width": 1.001, "base_arch_lift": 0.000, "base_leg_length": 0.998, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.000, "base_foot_width": 1.000, "base_arch_lift": 0.000, "base_leg_length": 1.003, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
]

ANATOMIC_FAST_V7_BASE_PROFILES = [
    {"base_foot_length": 1.000, "base_foot_width": 1.000, "base_arch_lift": 0.000, "base_leg_length": 1.000, "base_toe_splay": 0.000, "base_ankle_bend": 0.000},
    {"base_foot_length": 1.012, "base_foot_width": 0.988, "base_arch_lift": 0.002, "base_leg_length": 1.004, "base_toe_splay": 0.001, "base_ankle_bend": 0.002},
    {"base_foot_length": 0.988, "base_foot_width": 1.018, "base_arch_lift": -0.001, "base_leg_length": 0.996, "base_toe_splay": -0.001, "base_ankle_bend": -0.002},
    {"base_foot_length": 1.020, "base_foot_width": 1.006, "base_arch_lift": 0.004, "base_leg_length": 1.000, "base_toe_splay": 0.002, "base_ankle_bend": 0.003},
    {"base_foot_length": 0.982, "base_foot_width": 0.992, "base_arch_lift": -0.002, "base_leg_length": 1.002, "base_toe_splay": -0.002, "base_ankle_bend": -0.003},
    {"base_foot_length": 1.006, "base_foot_width": 1.026, "base_arch_lift": 0.001, "base_leg_length": 1.006, "base_toe_splay": 0.003, "base_ankle_bend": 0.001},
    {"base_foot_length": 0.994, "base_foot_width": 0.974, "base_arch_lift": 0.005, "base_leg_length": 0.994, "base_toe_splay": -0.003, "base_ankle_bend": -0.001},
    {"base_foot_length": 1.016, "base_foot_width": 1.018, "base_arch_lift": 0.003, "base_leg_length": 0.998, "base_toe_splay": 0.002, "base_ankle_bend": 0.004},
    {"base_foot_length": 0.984, "base_foot_width": 1.010, "base_arch_lift": -0.003, "base_leg_length": 1.004, "base_toe_splay": -0.002, "base_ankle_bend": -0.004},
    {"base_foot_length": 1.008, "base_foot_width": 0.982, "base_arch_lift": 0.006, "base_leg_length": 1.008, "base_toe_splay": 0.001, "base_ankle_bend": 0.002},
    {"base_foot_length": 1.024, "base_foot_width": 0.968, "base_arch_lift": 0.008, "base_leg_length": 0.992, "base_toe_splay": 0.004, "base_ankle_bend": 0.005},
    {"base_foot_length": 0.976, "base_foot_width": 1.032, "base_arch_lift": -0.004, "base_leg_length": 1.010, "base_toe_splay": -0.004, "base_ankle_bend": -0.005},
]

ANATOMIC_V9_CONTACT_BASE_PROFILES = [
    {"base_foot_length": 1.000, "base_foot_width": 1.000, "base_arch_lift": 0.0000, "base_leg_length": 1.000, "base_toe_splay": 0.0000, "base_ankle_bend": 0.0000, "base_family_label": 0.0},
    {"base_foot_length": 1.012, "base_foot_width": 0.986, "base_arch_lift": 0.0018, "base_leg_length": 1.006, "base_toe_splay": 0.0008, "base_ankle_bend": 0.0008, "base_family_label": 1.0},
    {"base_foot_length": 0.988, "base_foot_width": 1.018, "base_arch_lift": -0.0008, "base_leg_length": 0.996, "base_toe_splay": -0.0008, "base_ankle_bend": -0.0008, "base_family_label": 2.0},
    {"base_foot_length": 1.018, "base_foot_width": 1.008, "base_arch_lift": 0.0025, "base_leg_length": 1.000, "base_toe_splay": 0.0012, "base_ankle_bend": 0.0010, "base_family_label": 3.0},
    {"base_foot_length": 0.982, "base_foot_width": 0.992, "base_arch_lift": 0.0005, "base_leg_length": 1.004, "base_toe_splay": -0.0010, "base_ankle_bend": -0.0010, "base_family_label": 4.0},
    {"base_foot_length": 1.006, "base_foot_width": 1.024, "base_arch_lift": -0.0010, "base_leg_length": 1.008, "base_toe_splay": 0.0012, "base_ankle_bend": 0.0005, "base_family_label": 5.0},
    {"base_foot_length": 0.996, "base_foot_width": 0.976, "base_arch_lift": 0.0032, "base_leg_length": 0.992, "base_toe_splay": -0.0012, "base_ankle_bend": -0.0005, "base_family_label": 6.0},
    {"base_foot_length": 1.014, "base_foot_width": 1.014, "base_arch_lift": 0.0015, "base_leg_length": 0.996, "base_toe_splay": 0.0010, "base_ankle_bend": 0.0012, "base_family_label": 7.0},
    {"base_foot_length": 0.986, "base_foot_width": 1.010, "base_arch_lift": -0.0015, "base_leg_length": 1.006, "base_toe_splay": -0.0010, "base_ankle_bend": -0.0012, "base_family_label": 8.0},
    {"base_foot_length": 1.008, "base_foot_width": 0.982, "base_arch_lift": 0.0038, "base_leg_length": 1.012, "base_toe_splay": 0.0005, "base_ankle_bend": 0.0008, "base_family_label": 9.0},
    {"base_foot_length": 1.020, "base_foot_width": 0.974, "base_arch_lift": 0.0045, "base_leg_length": 0.990, "base_toe_splay": 0.0015, "base_ankle_bend": 0.0015, "base_family_label": 10.0},
    {"base_foot_length": 0.980, "base_foot_width": 1.026, "base_arch_lift": -0.0020, "base_leg_length": 1.015, "base_toe_splay": -0.0015, "base_ankle_bend": -0.0015, "base_family_label": 11.0},
]

RANGE_PRESETS = {
    "anatomic_v10_contact": ANATOMIC_V10_CONTACT_RANGES,
    "anatomic_v9_contact": ANATOMIC_V9_CONTACT_RANGES,
    "anatomic_fast_v7": ANATOMIC_FAST_V7_RANGES,
    "simplefoot": RANGES,
    "anatomic_pilot": ANATOMIC_PILOT_RANGES,
}

BASE_PROFILE_PRESETS = {
    "anatomic_v10_contact": ANATOMIC_V9_CONTACT_BASE_PROFILES,
    "anatomic_v9_contact": ANATOMIC_V9_CONTACT_BASE_PROFILES,
    "anatomic_fast_v7": ANATOMIC_FAST_V7_BASE_PROFILES,
    "simplefoot": BASE_PROFILES,
    "anatomic_pilot": ANATOMIC_BASE_PROFILES,
}


def load_base_profiles(path: Path | None, preset: str = "simplefoot") -> list[dict[str, float]]:
    if path is None or not path.exists():
        return BASE_PROFILE_PRESETS[preset]
    rows = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[dict[str, float]] = []
    for row in sorted(rows, key=lambda item: int(item["base_model_id"])):
        profiles.append({key: float(value) for key, value in row["profile"].items()})
    return profiles


def sample_params(
    rng: random.Random,
    base_model_id: int,
    base_profiles: list[dict[str, float]],
    ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    params = {key: rng.uniform(lo, hi) for key, (lo, hi) in ranges.items()}
    params["base_model_id"] = float(base_model_id)
    params["base_is_training_family"] = float(base_model_id in TRAIN_BASE_IDS)
    params.update({key: float(value) for key, value in base_profiles[base_model_id].items()})
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--dataset-id", default="default")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--base-model-count", type=int, default=BASE_MODEL_COUNT)
    parser.add_argument("--base-profiles", type=Path, default=Path("templates/base_models/base_model_profiles.json"))
    parser.add_argument("--preset", choices=sorted(RANGE_PRESETS), default="simplefoot")
    args = parser.parse_args()
    base_profiles = load_base_profiles(args.base_profiles, preset=args.preset)
    ranges = RANGE_PRESETS[args.preset]
    if args.base_model_count != len(base_profiles):
        raise ValueError(
            f"Requested {args.base_model_count} base models, but {args.base_profiles} defines "
            f"{len(base_profiles)} profiles."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.out.open("w", encoding="utf-8") as f:
        for offset in range(args.count):
            sample_id = args.start_id + offset
            seed = rng.randrange(1, 2**31 - 1)
            row_rng = random.Random(seed)
            base_model_id = sample_id % args.base_model_count
            row = {
                "dataset_id": args.dataset_id,
                "sample_id": sample_id,
                "sample_name": f"sample_{sample_id:06d}",
                "seed": seed,
                "params": sample_params(row_rng, base_model_id, base_profiles, ranges),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
