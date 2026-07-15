from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_MATERIALS = {
    "Flesh_soft_tissue",
    "Cortical_bone_stiff",
    "Ankle_joint_pad",
    "Heel_pad_soft",
    "Forefoot_pad_soft",
    "Ankle_ligament_collar",
    "Plantar_fascia_band",
    "Achilles_like_band",
}

REQUIRED_SOLID_DOMAINS = {
    "Flesh",
    "TibiaBone",
    "FootBone",
    "JointPad",
    "HeelPad",
    "ForefootPad",
    "AnkleCollar",
    "PlantarFascia",
    "AchillesBand",
}

REQUIRED_SURFACES = {
    "FootSoleContact",
    "TibiaTopDriveSurface",
}

REQUIRED_LOAD_CONTROLLERS = {"1", "2"}

REQUIRED_SHARD_KEYS = {
    "sample_ids",
    "sample_names",
    "dataset_ids",
    "params_json",
    "sole_near_element_mask",
    "last_nodes",
    "last_elements",
    "last_contact",
    "last_element_von_mises",
}

PINN_HISTORY_KEYS = {
    "node_times",
    "element_times",
    "contact_times",
    "node_history",
    "element_history",
    "contact_history",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"[CONTRACT ERROR] {message}")


def surface_names(root: ET.Element) -> set[str]:
    return {item.attrib["name"] for item in root.findall("./Mesh/Surface") if "name" in item.attrib}


def material_names(root: ET.Element) -> set[str]:
    return {item.attrib["name"] for item in root.findall("./Material/material") if "name" in item.attrib}


def solid_domain_names(root: ET.Element) -> set[str]:
    return {item.attrib["name"] for item in root.findall("./MeshDomains/SolidDomain") if "name" in item.attrib}


def load_controller_ids(root: ET.Element) -> set[str]:
    return {item.attrib["id"] for item in root.findall("./LoadData/load_controller") if "id" in item.attrib}


def output_logger_names(root: ET.Element) -> set[str]:
    names = set()
    output = root.find("./Output/logfile")
    if output is None:
        return names
    for tag in ("node_data", "element_data", "face_data"):
        for item in output.findall(tag):
            if "name" in item.attrib:
                names.add(item.attrib["name"])
    return names


def validate_base_model(path: Path, row: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing base FEB model: {path}")
    root = ET.parse(path).getroot()
    row = row or {}
    required_materials = set(row.get("required_materials", REQUIRED_MATERIALS))
    required_solid_domains = set(row.get("solid_domains", REQUIRED_SOLID_DOMAINS))
    required_surfaces = set(row.get("required_surfaces", REQUIRED_SURFACES))
    required_load_controllers = set(row.get("required_load_controllers", REQUIRED_LOAD_CONTROLLERS))
    missing = {
        "materials": sorted(required_materials - material_names(root)),
        "solid_domains": sorted(required_solid_domains - solid_domain_names(root)),
        "surfaces": sorted(required_surfaces - surface_names(root)),
        "load_controllers": sorted(required_load_controllers - load_controller_ids(root)),
    }
    if any(missing.values()):
        fail(f"{path} is missing required FEB objects: {missing}")
    return {
        "path": str(path),
        "output_loggers_present": sorted(output_logger_names(root)),
    }


def validate_profiles(path: Path, base_model_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        fail(f"Missing base model profile file: {path}")
    rows = load_json(path)
    if len(rows) != 12:
        fail(f"Expected 12 base model profiles, found {len(rows)} in {path}")

    ids = [int(row["base_model_id"]) for row in rows]
    if sorted(ids) != list(range(12)):
        fail(f"Expected base_model_id values 0..11, found {sorted(ids)}")

    split_counts = Counter(str(row.get("split_role", "")) for row in rows)
    if split_counts.get("train", 0) != 10 or split_counts.get("validation_holdout", 0) != 2:
        fail(f"Expected 10 train and 2 validation_holdout profiles, found {dict(split_counts)}")

    reports = []
    for row in sorted(rows, key=lambda item: int(item["base_model_id"])):
        base_id = int(row["base_model_id"])
        template = Path(str(row.get("template", "")))
        candidates = [
            template,
            Path.cwd() / template if not template.is_absolute() else template,
            base_model_dir / template.name,
            base_model_dir / f"base_{base_id:02d}.feb",
            base_model_dir / f"simplefoot_base_{base_id:02d}.feb",
        ]
        template = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        reports.append({"base_model_id": base_id, "split_role": row.get("split_role"), **validate_base_model(template, row)})
    return reports


def validate_manifest(path: Path, dataset_id: str | None) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing manifest: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        fail(f"Manifest is empty: {path}")
    dataset_ids = {str(row.get("dataset_id", "default")) for row in rows}
    if dataset_id and dataset_ids != {dataset_id}:
        fail(f"Manifest dataset_id mismatch. Expected {dataset_id}, found {sorted(dataset_ids)}")
    base_counts = Counter(int(round(float(row["params"]["base_model_id"]))) for row in rows)
    missing_bases = sorted(set(range(12)) - set(base_counts))
    if missing_bases:
        fail(f"Manifest has no samples for base model ids: {missing_bases}")
    return {"rows": len(rows), "dataset_ids": sorted(dataset_ids), "base_counts": dict(sorted(base_counts.items()))}


def validate_shards(shard_dir: Path, require_history: bool) -> dict[str, Any]:
    if not shard_dir.exists():
        return {"checked": False, "reason": "shard_dir_missing"}
    paths = sorted(shard_dir.glob("batch_*.npz"))
    if not paths:
        return {"checked": False, "reason": "no_shards"}

    sample_count = 0
    missing_by_file = {}
    required = set(REQUIRED_SHARD_KEYS)
    if require_history:
        required |= PINN_HISTORY_KEYS
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required - set(data.files))
            if missing:
                missing_by_file[str(path)] = missing
            sample_count += int(data["sample_ids"].shape[0]) if "sample_ids" in data.files else 0
    if missing_by_file:
        fail(f"Packed shard contract failed: {missing_by_file}")
    return {"checked": True, "shards": len(paths), "samples": sample_count, "require_history": require_history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FEBio foot dataset prerequisites before a large run.")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-model-dir", type=Path, default=Path("templates/base_models"))
    parser.add_argument("--base-profiles", type=Path, default=Path("templates/base_models/base_model_profiles.json"))
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--require-history", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {
        "dataset_id": args.dataset_id,
        "base_models": validate_profiles(args.base_profiles, args.base_model_dir),
    }
    if args.manifest:
        report["manifest"] = validate_manifest(args.manifest, args.dataset_id)
    if args.shard_dir:
        report["shards"] = validate_shards(args.shard_dir, args.require_history)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
