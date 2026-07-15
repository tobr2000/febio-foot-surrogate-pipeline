from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ROOT, DEFAULT_TEMPLATE, SOLE_SURFACE_CANDIDATES, TOP_SURFACE_CANDIDATES, apply_shape_family, first_existing_surface
from generate_manifest import BASE_PROFILE_PRESETS


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


def material_names(root: ET.Element) -> list[str]:
    return sorted(
        item.attrib["name"]
        for item in root.findall("./Material/material")
        if "name" in item.attrib
    )


def solid_domain_names(root: ET.Element) -> list[str]:
    return sorted(
        item.attrib["name"]
        for item in root.findall("./MeshDomains/SolidDomain")
        if "name" in item.attrib
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 12 explicit FEB base templates from a source model.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out-dir", type=Path, default=Path("templates/base_models"))
    parser.add_argument("--metadata", type=Path, default=Path("templates/base_models/base_model_profiles.json"))
    parser.add_argument("--preset", choices=sorted(BASE_PROFILE_PRESETS), default="simplefoot")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    profiles = BASE_PROFILE_PRESETS[args.preset]
    prefix = args.prefix or ("simplefoot_base" if args.preset == "simplefoot" else "base")
    generated = []
    for base_model_id, profile in enumerate(profiles):
        out_path = args.out_dir / f"{prefix}_{base_model_id:02d}.feb"
        write_base_template(args.source, out_path, base_model_id, profile)
        root = ET.parse(out_path).getroot()
        try:
            template_ref = out_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            template_ref = out_path.as_posix()
        generated.append(
            {
                "schema_version": 2,
                "base_model_id": base_model_id,
                "template": template_ref,
                "profile": profile,
                "split_role": "train" if base_model_id < 10 else "validation_holdout",
                "required_surfaces": [
                    first_existing_surface(root, SOLE_SURFACE_CANDIDATES),
                    first_existing_surface(root, TOP_SURFACE_CANDIDATES),
                ],
                "required_load_controllers": [item.attrib["id"] for item in root.findall("./LoadData/load_controller") if "id" in item.attrib],
                "required_materials": material_names(root),
                "solid_domains": solid_domain_names(root),
                "preset": args.preset,
            }
        )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps({"generated": len(generated), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
