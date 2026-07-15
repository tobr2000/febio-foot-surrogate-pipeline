from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_TEMPLATE, apply_shape_family
from generate_manifest import BASE_PROFILES


def write_base_template(source: Path, out_path: Path, base_model_id: int, profile: dict[str, float]) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    params = {
        "base_model_id": float(base_model_id),
        "arch_lift": 0.0,
        "heel_toe_roll": 0.0,
        "toe_off_bias": 0.0,
    }
    params.update({key: float(value) for key, value in profile.items()})
    apply_shape_family(root, params, include_base_profile=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="\t")
    tree.write(out_path, encoding="ISO-8859-1", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 12 explicit simple-foot base FEB templates.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out-dir", type=Path, default=Path("templates/base_models"))
    parser.add_argument("--metadata", type=Path, default=Path("templates/base_models/base_model_profiles.json"))
    args = parser.parse_args()

    generated = []
    for base_model_id, profile in enumerate(BASE_PROFILES):
        out_path = args.out_dir / f"simplefoot_base_{base_model_id:02d}.feb"
        write_base_template(args.source, out_path, base_model_id, profile)
        generated.append(
            {
                "base_model_id": base_model_id,
                "template": str(out_path),
                "profile": profile,
                "split_role": "train" if base_model_id < 10 else "validation_holdout",
            }
        )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps({"generated": len(generated), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
