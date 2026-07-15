from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


REGION_NAMES = ["heel", "midfoot", "forefoot", "toe"]


def list_shards(shard_dir: Path) -> list[Path]:
    shards = sorted(shard_dir.glob("batch_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No batch_*.npz files found in {shard_dir}")
    return shards


def load_param_json(text: Any) -> dict[str, Any]:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(str(text))


def quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in qs}
    vals = np.quantile(arr, qs)
    return {f"q{int(q * 100):02d}": float(v) for q, v in zip(qs, vals)}


def contact_grid(face_count: int) -> np.ndarray:
    side = int(round(face_count**0.5))
    if side * side != face_count:
        ids = np.linspace(-1.0, 1.0, face_count, dtype=np.float32)
        return np.stack([ids, np.zeros_like(ids), np.zeros_like(ids)], axis=1)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, side, dtype=np.float32),
        np.linspace(-1.0, 1.0, side, dtype=np.float32),
        indexing="ij",
    )
    zz = np.zeros_like(xx)
    return np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)


def build_knn_edges(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(points, dtype=np.float64)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    order = np.argsort(dist, axis=1)[:, 1 : k + 1]
    dst = np.repeat(np.arange(coords.shape[0], dtype=np.int64), k)
    src = order.reshape(-1).astype(np.int64)
    return src, dst


def connected_components(n_nodes: int, src: np.ndarray, dst: np.ndarray) -> tuple[int, int]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in zip(src.tolist(), dst.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n_nodes, dtype=bool)
    sizes: list[int] = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        size = 0
        while q:
            node = q.popleft()
            size += 1
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return len(sizes), max(sizes) if sizes else 0


def reciprocal_fraction(src: np.ndarray, dst: np.ndarray) -> float:
    edges = {(int(a), int(b)) for a, b in zip(src, dst)}
    reciprocal = sum((int(b), int(a)) in edges for a, b in zip(src, dst))
    return float(reciprocal / max(1, len(edges)))


def graph_geometry_metrics(base_id: int, coords: np.ndarray, regions: np.ndarray | None, k: int) -> dict[str, Any]:
    src, dst = build_knn_edges(coords, k)
    lengths = np.linalg.norm(coords[dst] - coords[src], axis=1)
    component_count, largest_component = connected_components(coords.shape[0], src, dst)
    row: dict[str, Any] = {
        "base_id": int(base_id),
        "k": int(k),
        "n_nodes": int(coords.shape[0]),
        "n_edges_directed": int(src.shape[0]),
        "edge_length_mean": float(np.mean(lengths)),
        "edge_length_std": float(np.std(lengths)),
        "edge_reciprocal_fraction": reciprocal_fraction(src, dst),
        "component_count_undirected": int(component_count),
        "largest_component_fraction": float(largest_component / max(1, coords.shape[0])),
    }
    row.update({f"edge_length_{key}": value for key, value in quantiles(lengths).items()})
    if regions is not None:
        row["cross_region_edge_fraction"] = float(np.mean(regions[src] != regions[dst]))
        counts = np.bincount(regions.astype(int), minlength=len(REGION_NAMES))
        for idx, name in enumerate(REGION_NAMES):
            row[f"region_count_{name}"] = int(counts[idx])
    else:
        row["cross_region_edge_fraction"] = float("nan")
    return row


def random_pair_indices(rng: np.random.Generator, n_nodes: int, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.integers(0, n_nodes, size=n_pairs)
    b = rng.integers(0, n_nodes - 1, size=n_pairs)
    b = np.where(b >= a, b + 1, b)
    return a, b


def pressure_graph_metrics(
    pressure: np.ndarray,
    coords: np.ndarray,
    regions: np.ndarray | None,
    k: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    src, dst = build_knn_edges(coords, k)
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    edge_abs = np.abs(p[dst] - p[src])
    rand_a, rand_b = random_pair_indices(rng, p.shape[0], src.shape[0])
    random_abs = np.abs(p[rand_b] - p[rand_a])
    p_centered = p - np.mean(p)
    denom = float(np.mean(p_centered**2))
    if denom > 1.0e-20:
        neighbor_corr = float(np.mean(p_centered[src] * p_centered[dst]) / denom)
    else:
        neighbor_corr = float("nan")

    active = p > 1.0e-10
    top_count = max(1, int(round(0.10 * p.shape[0])))
    top_idx = np.argpartition(p, -top_count)[-top_count:]
    top = np.zeros(p.shape[0], dtype=bool)
    top[top_idx] = True
    top_edge_fraction = float(np.mean(top[src] & top[dst]))
    top_neighbor_coverage = float(np.mean(np.bincount(dst[top[src]], minlength=p.shape[0])[top] > 0))

    out = {
        "pressure_edge_abs_mean": float(np.mean(edge_abs)),
        "pressure_random_abs_mean": float(np.mean(random_abs)),
        "pressure_edge_to_random_abs_ratio": float(np.mean(edge_abs) / max(np.mean(random_abs), 1.0e-20)),
        "pressure_neighbor_corr": neighbor_corr,
        "active_face_fraction": float(np.mean(active)),
        "top10_edge_fraction": top_edge_fraction,
        "top10_neighbor_coverage": top_neighbor_coverage,
        "peak_pressure": float(np.max(p)),
        "mean_pressure": float(np.mean(p)),
    }
    if regions is not None:
        for idx, name in enumerate(REGION_NAMES):
            mask = regions == idx
            out[f"region_pressure_mean_{name}"] = float(np.mean(p[mask])) if np.any(mask) else float("nan")
            out[f"region_pressure_peak_{name}"] = float(np.max(p[mask])) if np.any(mask) else float("nan")
    return out


def collect_sample_refs(shards: list[Path]) -> list[tuple[Path, int, int, int]]:
    refs: list[tuple[Path, int, int, int]] = []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            sample_ids = data["sample_ids"].astype(int)
            params_json = data["params_json"]
            for local_idx, sample_id in enumerate(sample_ids.tolist()):
                params = load_param_json(params_json[local_idx])
                base_id = int(round(float(params.get("base_model_id", -1))))
                refs.append((shard, local_idx, int(sample_id), base_id))
    return refs


def choose_refs(
    refs: list[tuple[Path, int, int, int]],
    max_samples: int,
    seed: int,
) -> list[tuple[Path, int, int, int]]:
    if max_samples <= 0 or len(refs) <= max_samples:
        return refs
    by_base: dict[int, list[tuple[Path, int, int, int]]] = defaultdict(list)
    for ref in refs:
        by_base[ref[3]].append(ref)
    rng = np.random.default_rng(seed)
    per_base = max(1, int(math.ceil(max_samples / max(1, len(by_base)))))
    selected: list[tuple[Path, int, int, int]] = []
    for base_id in sorted(by_base):
        group = by_base[base_id]
        if len(group) <= per_base:
            selected.extend(group)
        else:
            indices = rng.choice(len(group), size=per_base, replace=False)
            selected.extend(group[int(i)] for i in indices)
    selected.sort(key=lambda item: item[2])
    return selected[:max_samples]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GNO contact graph quality on model-ready FEBio foot shards.")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-list", default="4,6,8,10,12")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    k_values = [int(part) for part in args.k_list.split(",") if part.strip()]
    shards = list_shards(shard_dir)
    refs = collect_sample_refs(shards)
    selected_refs = choose_refs(refs, args.max_samples, args.seed)
    rng = np.random.default_rng(args.seed)

    base_geometry: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    pressure_rows: list[dict[str, Any]] = []

    refs_by_shard: dict[Path, list[tuple[Path, int, int, int]]] = defaultdict(list)
    for ref in selected_refs:
        refs_by_shard[ref[0]].append(ref)

    for shard_i, shard in enumerate(sorted(refs_by_shard), start=1):
        print(f"[GRAPH] shard {shard_i}/{len(refs_by_shard)} {shard.name}", flush=True)
        with np.load(shard, allow_pickle=False) as data:
            has_geom = "contact_pos_norm" in data.files
            has_region = "contact_region_id" in data.files
            for _, local_idx, sample_id, base_id in refs_by_shard[shard]:
                pressure = data["last_contact"][local_idx, :, 2]
                if has_geom:
                    coords = np.asarray(data["contact_pos_norm"][local_idx], dtype=np.float64)
                else:
                    coords = contact_grid(int(pressure.shape[0]))
                regions = np.asarray(data["contact_region_id"][local_idx], dtype=np.int16) if has_region else None
                base_geometry.setdefault(base_id, (coords, regions))
                for k in k_values:
                    row: dict[str, Any] = {
                        "sample_id": int(sample_id),
                        "base_id": int(base_id),
                        "k": int(k),
                    }
                    row.update(pressure_graph_metrics(pressure, coords, regions, k, rng))
                    pressure_rows.append(row)

    geometry_rows: list[dict[str, Any]] = []
    for base_id, (coords, regions) in sorted(base_geometry.items()):
        for k in k_values:
            geometry_rows.append(graph_geometry_metrics(base_id, coords, regions, k))

    aggregate_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pressure_rows:
        grouped[(int(row["base_id"]), int(row["k"]))].append(row)
    for (base_id, k), rows in sorted(grouped.items()):
        numeric_keys = [
            key
            for key in rows[0]
            if key not in {"sample_id", "base_id", "k"} and isinstance(rows[0][key], (int, float, np.floating))
        ]
        out: dict[str, Any] = {"base_id": base_id, "k": k, "n_samples": len(rows)}
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            out[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
            out[f"{key}_std"] = float(np.std(values)) if values.size else float("nan")
        aggregate_rows.append(out)

    write_csv(out_dir / "graph_geometry_by_base_k.csv", geometry_rows)
    write_csv(out_dir / "pressure_graph_by_sample_k.csv", pressure_rows)
    write_csv(out_dir / "pressure_graph_by_base_k.csv", aggregate_rows)

    summary = {
        "shard_dir": str(shard_dir),
        "out_dir": str(out_dir),
        "shard_count": len(shards),
        "sample_count_total": len(refs),
        "sample_count_analyzed": len(selected_refs),
        "base_ids_seen": sorted(int(x) for x in base_geometry),
        "k_values": k_values,
        "files": {
            "graph_geometry_by_base_k": str(out_dir / "graph_geometry_by_base_k.csv"),
            "pressure_graph_by_sample_k": str(out_dir / "pressure_graph_by_sample_k.csv"),
            "pressure_graph_by_base_k": str(out_dir / "pressure_graph_by_base_k.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
