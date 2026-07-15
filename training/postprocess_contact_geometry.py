from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from numpy.lib import format as npy_format


REGION_NAMES = np.asarray(["heel", "midfoot", "forefoot", "toe"])
DEFAULT_SURFACE = "AnatomicSoleContact"
REQUIRED_SHARD_KEYS = {"sample_ids", "params_json", "last_contact"}
HISTORY_KEYS = ("node_history", "element_history", "contact_history")


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    log(f"[FAIL] {message}")
    raise SystemExit(2)


def pass_step(message: str) -> None:
    log(f"[PASS] {message}")


def warn(message: str) -> None:
    log(f"[WARN] {message}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_node_text(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) < 3:
        raise ValueError(f"Expected 3 coordinate values, got {text!r}")
    return parts[0], parts[1], parts[2]


def parse_id_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


@dataclass
class ContactGeometry:
    base_model_id: int
    template: str
    face_ids: np.ndarray
    face_node_ids: np.ndarray
    centroids: np.ndarray
    centroids_norm: np.ndarray
    region_ids: np.ndarray
    region_onehot: np.ndarray


def find_nodes(root: ET.Element) -> dict[int, np.ndarray]:
    nodes: dict[int, np.ndarray] = {}
    for node in root.findall(".//Nodes/node"):
        node_id = node.attrib.get("id")
        if node_id is None or node.text is None:
            continue
        nodes[int(node_id)] = np.asarray(parse_node_text(node.text), dtype=np.float32)
    return nodes


def find_surface(root: ET.Element, surface_name: str) -> ET.Element | None:
    for surface in root.findall(".//Surface"):
        if surface.attrib.get("name") == surface_name:
            return surface
    return None


def normalize_centroids(centroids: np.ndarray) -> np.ndarray:
    lo = centroids.min(axis=0)
    hi = centroids.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    return ((centroids - lo) / span * 2.0 - 1.0).astype(np.float32)


def region_ids_from_centroids(centroids: np.ndarray) -> np.ndarray:
    spans = centroids.max(axis=0) - centroids.min(axis=0)
    length_axis = int(np.argmax(spans))
    coord = centroids[:, length_axis]
    q = (coord - coord.min()) / max(float(coord.max() - coord.min()), 1e-8)
    region = np.zeros(coord.shape[0], dtype=np.int16)
    region[(q >= 0.25) & (q < 0.55)] = 1
    region[(q >= 0.55) & (q < 0.82)] = 2
    region[q >= 0.82] = 3
    return region


def onehot(region_ids: np.ndarray, n_regions: int = 4) -> np.ndarray:
    out = np.zeros((region_ids.shape[0], n_regions), dtype=np.float32)
    out[np.arange(region_ids.shape[0]), region_ids.astype(int)] = 1.0
    return out


def load_contact_geometry(template: Path, base_model_id: int, surface_name: str) -> ContactGeometry:
    root = ET.parse(template).getroot()
    nodes = find_nodes(root)
    if not nodes:
        fail(f"base {base_model_id}: no Mesh/Nodes/node coordinates found in {template}")
    surface = find_surface(root, surface_name)
    if surface is None:
        fail(f"base {base_model_id}: surface {surface_name!r} not found in {template}")

    faces: list[tuple[int, list[int]]] = []
    for item in list(surface):
        if item.text is None:
            continue
        tag = item.tag.lower()
        if tag not in {"tri3", "quad4", "quad8", "quad9"}:
            continue
        face_id = int(item.attrib.get("id", len(faces) + 1))
        node_ids = parse_id_list(item.text)
        if len(node_ids) < 3:
            fail(f"base {base_model_id}: face {face_id} has fewer than 3 nodes")
        faces.append((face_id, node_ids))

    if not faces:
        fail(f"base {base_model_id}: no tri/quad faces found on {surface_name}")
    faces.sort(key=lambda row: row[0])

    max_face_nodes = max(len(node_ids) for _, node_ids in faces)
    face_node_ids = np.full((len(faces), max_face_nodes), -1, dtype=np.int32)
    centroids = np.zeros((len(faces), 3), dtype=np.float32)
    face_ids = np.zeros((len(faces),), dtype=np.int32)
    for i, (face_id, node_ids) in enumerate(faces):
        missing = [node_id for node_id in node_ids if node_id not in nodes]
        if missing:
            fail(f"base {base_model_id}: face {face_id} references missing nodes {missing[:5]}")
        coords = np.stack([nodes[node_id] for node_id in node_ids], axis=0)
        face_ids[i] = face_id
        face_node_ids[i, : len(node_ids)] = node_ids
        centroids[i] = coords.mean(axis=0)

    region_ids = region_ids_from_centroids(centroids)
    return ContactGeometry(
        base_model_id=base_model_id,
        template=str(template),
        face_ids=face_ids,
        face_node_ids=face_node_ids,
        centroids=centroids,
        centroids_norm=normalize_centroids(centroids),
        region_ids=region_ids,
        region_onehot=onehot(region_ids),
    )


def npy_bytes(array: np.ndarray) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)
        handle.seek(0)
        return handle.read()


def npz_member_key(info: zipfile.ZipInfo) -> str | None:
    if not info.filename.endswith(".npy"):
        return None
    return Path(info.filename).stem


def npy_header_from_npz(path: Path, key: str) -> dict[str, Any]:
    """Read only the .npy header for a member inside an NPZ archive."""
    member = f"{key}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if member not in names:
            raise KeyError(member)
        with archive.open(member, "r") as handle:
            version = npy_format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
            else:
                shape, fortran_order, dtype = npy_format._read_array_header(handle, version)
    return {
        "shape": tuple(int(x) for x in shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
    }


def inspect_one_shard(path: Path, repack_history: bool) -> dict[str, Any]:
    log(f"[PREFLIGHT] inspecting shard: {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        missing = sorted(REQUIRED_SHARD_KEYS - set(keys))
        if missing:
            fail(f"sample shard is missing required keys: {missing}")
        n_samples = int(data["sample_ids"].shape[0])
        last_contact_shape = tuple(int(x) for x in data["last_contact"].shape)
        if len(last_contact_shape) != 3:
            fail(f"last_contact must be rank 3, got shape {last_contact_shape}")
        if last_contact_shape[0] != n_samples:
            fail(f"last_contact sample dimension {last_contact_shape[0]} != sample_ids {n_samples}")
        if repack_history:
            missing_history = [key for key in HISTORY_KEYS if key not in keys]
            if missing_history:
                fail(f"history repack requested, but first shard is missing {missing_history}")
        history_info = {
            key: npy_header_from_npz(path, key)
            for key in HISTORY_KEYS
            if key in keys
        }
        params_json = data["params_json"]
        base_ids = sorted(
            {
                int(round(float(json.loads(str(params_json[i])).get("base_model_id", -1))))
                for i in range(n_samples)
            }
        )
        info = {
            "keys": keys,
            "n_samples": n_samples,
            "contact_face_count": last_contact_shape[1],
            "last_contact_shape": last_contact_shape,
            "history_info": history_info,
            "base_ids_in_first_shard": base_ids,
        }
        log(f"[PREFLIGHT] shard keys: {keys}")
        log(f"[PREFLIGHT] first shard info: {json.dumps(info, sort_keys=True)}")
        pass_step("sample shard has required fields and usable contact/history schema")
        return info


def load_base_metadata(project_dir: Path, metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        fail(f"base metadata not found: {metadata_path}")
    rows = read_json(metadata_path)
    if not isinstance(rows, list) or not rows:
        fail(f"base metadata is not a non-empty list: {metadata_path}")
    for row in rows:
        if "base_model_id" not in row or "template" not in row:
            fail("base metadata rows must contain base_model_id and template")
        template = project_dir / str(row["template"])
        if not template.exists():
            fail(f"base template for id {row['base_model_id']} not found: {template}")
    pass_step(f"base metadata found with {len(rows)} template rows")
    return rows


def build_geometry_by_base(
    project_dir: Path,
    base_rows: list[dict[str, Any]],
    surface_name: str,
    expected_face_count: int,
) -> dict[int, ContactGeometry]:
    geometries: dict[int, ContactGeometry] = {}
    for row in sorted(base_rows, key=lambda item: int(item["base_model_id"])):
        base_id = int(row["base_model_id"])
        template = project_dir / str(row["template"])
        log(f"[PREFLIGHT] parsing base {base_id}: {template}")
        geom = load_contact_geometry(template, base_id, surface_name)
        if geom.centroids.shape[0] != expected_face_count:
            fail(
                f"base {base_id}: {surface_name} has {geom.centroids.shape[0]} faces, "
                f"but shard last_contact has {expected_face_count}"
            )
        counts = np.bincount(geom.region_ids.astype(int), minlength=4).tolist()
        log(
            f"[PREFLIGHT] base {base_id}: faces={geom.centroids.shape[0]} "
            f"region_counts={dict(zip(REGION_NAMES.tolist(), counts))}"
        )
        geometries[base_id] = geom
    pass_step("all base contact surfaces match shard contact face count")
    return geometries


def sample_base_ids_from_shard(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        params_json = data["params_json"]
        base_ids = [
            int(round(float(json.loads(str(params_json[i])).get("base_model_id", -1))))
            for i in range(params_json.shape[0])
        ]
    return np.asarray(base_ids, dtype=np.int16)


def arrays_for_shard(path: Path, geometries: dict[int, ContactGeometry]) -> dict[str, np.ndarray]:
    base_ids = sample_base_ids_from_shard(path)
    missing = sorted(set(int(x) for x in base_ids) - set(geometries))
    if missing:
        raise RuntimeError(f"{path.name}: shard references base ids without geometry: {missing}")
    first_geom = next(iter(geometries.values()))
    n_samples = int(base_ids.shape[0])
    n_faces = int(first_geom.centroids.shape[0])

    contact_pos = np.empty((n_samples, n_faces, 3), dtype=np.float32)
    contact_pos_norm = np.empty((n_samples, n_faces, 3), dtype=np.float32)
    contact_region_id = np.empty((n_samples, n_faces), dtype=np.int16)
    contact_region_onehot = np.empty((n_samples, n_faces, 4), dtype=np.float32)
    contact_face_id = first_geom.face_ids.astype(np.int32)

    for base_id in sorted(set(int(x) for x in base_ids)):
        mask = base_ids == base_id
        geom = geometries[base_id]
        contact_pos[mask] = geom.centroids
        contact_pos_norm[mask] = geom.centroids_norm
        contact_region_id[mask] = geom.region_ids
        contact_region_onehot[mask] = geom.region_onehot

    return {
        "contact_base_model_id": base_ids,
        "contact_face_id": contact_face_id,
        "contact_pos": contact_pos,
        "contact_pos_norm": contact_pos_norm,
        "contact_region_id": contact_region_id,
        "contact_region_onehot": contact_region_onehot,
        "contact_region_names": REGION_NAMES.astype("U16"),
    }


def atomic_extract_member(zin: zipfile.ZipFile, info: zipfile.ZipInfo, dst: Path) -> int:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zin.open(info, "r") as reader, tmp.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    size = tmp.stat().st_size
    tmp.replace(dst)
    return size


def copy_npz_with_added_arrays(
    src: Path,
    dst: Path,
    added: dict[str, np.ndarray],
    overwrite: bool,
    repack_history: bool,
) -> dict[str, Any]:
    if dst.exists() and not overwrite:
        raise RuntimeError(f"output shard exists and --overwrite was not given: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    sidecars: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zout:
        infos = zin.infolist()
        existing_npz_keys = {npz_member_key(info) for info in infos if npz_member_key(info) is not None}
        conflicts = sorted(set(added) & existing_npz_keys)
        if conflicts:
            raise RuntimeError(f"{src.name}: output arrays already exist in input shard: {conflicts}")

        history_infos = {npz_member_key(info): info for info in infos if npz_member_key(info) in HISTORY_KEYS}
        if repack_history:
            missing_history = [key for key in HISTORY_KEYS if key not in history_infos]
            if missing_history:
                raise RuntimeError(f"{src.name}: history repack requested but missing {missing_history}")

        for info in infos:
            key = npz_member_key(info)
            if repack_history and key in HISTORY_KEYS:
                sidecar = dst.with_name(f"{dst.stem}_{key}.npy")
                if sidecar.exists() and not overwrite:
                    raise RuntimeError(f"history sidecar exists and --overwrite was not given: {sidecar}")
                size = atomic_extract_member(zin, info, sidecar)
                sidecars[key] = {
                    "file": sidecar.name,
                    "bytes": size,
                    "source_member": info.filename,
                }
                continue

            out_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
            out_info.compress_type = info.compress_type
            out_info.external_attr = info.external_attr
            with zin.open(info, "r") as reader, zout.open(out_info, "w") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)

        for key, array in added.items():
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=time.localtime(time.time())[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            with zout.open(info, "w") as writer:
                writer.write(npy_bytes(array))
    tmp.replace(dst)
    return {"npz_file": dst.name, "history_sidecars": sidecars}


def preflight(args: argparse.Namespace) -> tuple[list[Path], dict[int, ContactGeometry], dict[str, Any]]:
    log("[PREFLIGHT] starting model-ready postprocessing preflight")
    project_dir = args.project_dir.resolve()
    shard_dir = (project_dir / args.shard_dir).resolve() if not args.shard_dir.is_absolute() else args.shard_dir.resolve()
    base_profiles = (project_dir / args.base_profiles).resolve() if not args.base_profiles.is_absolute() else args.base_profiles.resolve()

    if not project_dir.exists():
        fail(f"project_dir does not exist: {project_dir}")
    if not shard_dir.exists():
        fail(f"shard_dir does not exist: {shard_dir}")
    shards = sorted(shard_dir.glob("batch_*.npz"))
    if not shards:
        fail(f"no batch_*.npz files found in {shard_dir}")
    pass_step(f"found {len(shards)} shard(s) in {shard_dir}")

    first_info = inspect_one_shard(shards[0], repack_history=args.repack_pinn_history)
    if args.add_contact_geometry:
        base_rows = load_base_metadata(project_dir, base_profiles)
        geometries = build_geometry_by_base(
            project_dir=project_dir,
            base_rows=base_rows,
            surface_name=args.surface_name,
            expected_face_count=int(first_info["contact_face_count"]),
        )
    else:
        geometries = {}
        pass_step("contact geometry augmentation disabled")

    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            missing = sorted(REQUIRED_SHARD_KEYS - set(data.files))
            if missing:
                fail(f"{shard.name}: missing required keys {missing}")
            face_count = int(data["last_contact"].shape[1])
            if face_count != int(first_info["contact_face_count"]):
                fail(f"{shard.name}: contact face count {face_count} differs from first shard")
            if args.repack_pinn_history:
                missing_history = [key for key in HISTORY_KEYS if key not in data.files]
                if missing_history:
                    fail(f"{shard.name}: missing history keys {missing_history}")
                for key in HISTORY_KEYS:
                    header = npy_header_from_npz(shard, key)
                    if tuple(header["shape"]) != tuple(first_info["history_info"][key]["shape"]):
                        warn(f"{shard.name}: {key} shape {header['shape']} differs from first shard; continuing")
    pass_step("all shards have required fields; history arrays are present when requested")

    log("[PREFLIGHT] all checks passed; postprocessing may begin")
    return shards, geometries, first_info


def process_one_shard(task: dict[str, Any]) -> dict[str, Any]:
    src = Path(task["src"])
    dst = Path(task["dst"])
    geometries = task["geometries"]
    add_contact_geometry = bool(task["add_contact_geometry"])
    repack_history = bool(task["repack_history"])
    overwrite = bool(task["overwrite"])
    started = time.time()
    logs: list[str] = [f"[POST] shard start src={src.name} dst={dst.name}"]
    try:
        added = arrays_for_shard(src, geometries) if add_contact_geometry else {}
        if added:
            logs.append("[POST] " + src.name + " adding arrays: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in added.items()))
        result = copy_npz_with_added_arrays(
            src=src,
            dst=dst,
            added=added,
            overwrite=overwrite,
            repack_history=repack_history,
        )
        elapsed = time.time() - started
        logs.append(f"[POST] shard complete src={src.name} elapsed_sec={elapsed:.1f}")
        return {"ok": True, "src": src.name, "dst": dst.name, "elapsed_sec": elapsed, "logs": logs, **result}
    except Exception as exc:
        elapsed = time.time() - started
        logs.append(f"[FAIL] shard failed src={src.name} elapsed_sec={elapsed:.1f}: {exc}")
        logs.append(traceback.format_exc())
        return {"ok": False, "src": src.name, "dst": dst.name, "elapsed_sec": elapsed, "logs": logs, "error": str(exc)}


def postprocess(
    args: argparse.Namespace,
    shards: list[Path],
    geometries: dict[int, ContactGeometry],
    first_info: dict[str, Any],
) -> None:
    project_dir = args.project_dir.resolve()
    in_dir = (project_dir / args.shard_dir).resolve() if not args.shard_dir.is_absolute() else args.shard_dir.resolve()
    out_dir = (project_dir / args.out_shard_dir).resolve() if not args.out_shard_dir.is_absolute() else args.out_shard_dir.resolve()

    if in_dir == out_dir:
        fail("in-place augmentation is intentionally not supported; choose a different --out-shard-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    log(f"[POST] writing model-ready shards to {out_dir}")
    log(f"[POST] workers={workers} add_contact_geometry={args.add_contact_geometry} repack_pinn_history={args.repack_pinn_history}")

    tasks = [
        {
            "src": str(src),
            "dst": str(out_dir / src.name),
            "geometries": geometries,
            "add_contact_geometry": args.add_contact_geometry,
            "repack_history": args.repack_pinn_history,
            "overwrite": args.overwrite,
        }
        for src in shards
    ]

    results: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            result = process_one_shard(task)
            results.append(result)
            for line in result["logs"]:
                log(line)
            if not result["ok"]:
                fail(f"postprocessing stopped after failed shard {result['src']}")
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_src = {pool.submit(process_one_shard, task): Path(task["src"]).name for task in tasks}
            for fut in futures.as_completed(future_to_src):
                result = fut.result()
                results.append(result)
                for line in result["logs"]:
                    log(line)
                if not result["ok"]:
                    fail(f"postprocessing failed for shard {result['src']}")

    sidecar_manifest = {
        "source_shard_dir": str(in_dir),
        "output_shard_dir": str(out_dir),
        "history_keys": list(HISTORY_KEYS),
        "repacked": bool(args.repack_pinn_history),
        "shards": [
            {
                "source": result["src"],
                "npz_file": result.get("npz_file", result["dst"]),
                "history_sidecars": result.get("history_sidecars", {}),
                "elapsed_sec": result["elapsed_sec"],
            }
            for result in sorted(results, key=lambda item: item["src"])
        ],
    }
    (out_dir / "pinn_history_manifest.json").write_text(json.dumps(sidecar_manifest, indent=2), encoding="utf-8")

    metadata = {
        "source_shard_dir": str(in_dir),
        "output_shard_dir": str(out_dir),
        "surface_name": args.surface_name,
        "region_names": REGION_NAMES.tolist(),
        "arrays_added": [
            "contact_base_model_id",
            "contact_face_id",
            "contact_pos",
            "contact_pos_norm",
            "contact_region_id",
            "contact_region_onehot",
            "contact_region_names",
        ] if args.add_contact_geometry else [],
        "history_repacked_to_sidecars": bool(args.repack_pinn_history),
        "history_keys": list(HISTORY_KEYS) if args.repack_pinn_history else [],
        "first_shard_schema": first_info,
        "shard_count": len(shards),
        "workers": workers,
        "created_unix_time": time.time(),
    }
    (out_dir / "modelready_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log("[DONE] model-ready postprocessing finished successfully")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create model-ready FEBio shards with contact geometry and PINN sidecar histories.")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out-shard-dir", type=Path, required=True)
    parser.add_argument("--base-profiles", type=Path, default=Path("templates/base_models/anatomic_foot_v9_contact/base_model_profiles.json"))
    parser.add_argument("--surface-name", default=DEFAULT_SURFACE)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--add-contact-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repack-pinn-history", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    shards, geometries, first_info = preflight(args)
    if args.preflight_only:
        log("[DONE] preflight-only requested; no shards were written")
        return
    postprocess(args, shards, geometries, first_info)


if __name__ == "__main__":
    main()
