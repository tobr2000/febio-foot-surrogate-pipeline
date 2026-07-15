from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


PARAM_NAMES = [
    "base_model_id",
    "base_is_training_family",
    "base_foot_length",
    "base_foot_width",
    "base_arch_lift",
    "base_leg_length",
    "base_toe_splay",
    "scale_x",
    "scale_y",
    "scale_z",
    "E_flesh",
    "E_bone",
    "E_joint",
    "E_collar",
    "E_heel",
    "E_forefoot",
    "E_plantar",
    "E_achilles",
    "friction",
    "early_down_disp",
    "peak_down_disp",
    "final_down_disp",
    "forward_disp",
    "peak_time",
    "lateral_disp",
    "toe_off_bias",
    "heel_toe_roll",
    "arch_lift",
]


@dataclass(frozen=True)
class SampleRef:
    shard_path: Path
    local_index: int
    sample_id: int
    base_model_id: int


def list_shards(shard_dir: str | Path) -> list[Path]:
    paths = sorted(Path(shard_dir).glob("batch_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No batch_*.npz files found in {shard_dir}")
    return paths


def contact_grid(face_count: int = 256) -> np.ndarray:
    side = int(round(face_count ** 0.5))
    if side * side != face_count:
        ids = np.linspace(-1.0, 1.0, face_count, dtype=np.float32)
        return np.stack([ids, np.zeros_like(ids)], axis=1)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, side, dtype=np.float32),
        np.linspace(-1.0, 1.0, side, dtype=np.float32),
        indexing="ij",
    )
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float64)
    std[std < 1e-8] = 1.0
    return std


def _params_dict(text: str) -> dict:
    raw = json.loads(text)
    return raw


def _params_from_json(text: str) -> np.ndarray:
    raw = _params_dict(text)
    return np.asarray([raw.get(name, 0.0) for name in PARAM_NAMES], dtype=np.float32)


def _sample_refs(shards: list[Path]) -> list[SampleRef]:
    refs: list[SampleRef] = []
    for shard_path in shards:
        data = np.load(shard_path, allow_pickle=False)
        sample_ids = data["sample_ids"].astype(int).tolist()
        params_json = data["params_json"].tolist() if "params_json" in data.files else ["{}"] * len(sample_ids)
        data.close()
        for local_index, sample_id in enumerate(sample_ids):
            params = _params_dict(str(params_json[local_index]))
            refs.append(
                SampleRef(
                    shard_path=shard_path,
                    local_index=local_index,
                    sample_id=sample_id,
                    base_model_id=int(round(float(params.get("base_model_id", 0.0)))),
                )
            )
    return refs


def split_refs(
    refs: list[SampleRef],
    split: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
    train_base_ids: list[int] | None = None,
    validation_base_ids: list[int] | None = None,
    include_all_unseen_base_ids_in_val: bool = True,
) -> list[SampleRef]:
    rng = np.random.default_rng(seed)
    train_base_ids = train_base_ids or list(range(10))
    validation_base_ids = validation_base_ids or list(range(12))
    train_set = set(int(x) for x in train_base_ids)
    val_set = set(int(x) for x in validation_base_ids)

    seen_refs = [ref for ref in refs if ref.base_model_id in train_set]
    unseen_refs = [ref for ref in refs if ref.base_model_id in val_set and ref.base_model_id not in train_set]

    order = np.arange(len(seen_refs))
    rng.shuffle(order)
    val_count = max(1, int(round(len(seen_refs) * val_fraction)))
    if split == "val":
        selected = [seen_refs[int(i)] for i in order[:val_count]]
        if include_all_unseen_base_ids_in_val:
            selected.extend(unseen_refs)
        else:
            unseen_order = np.arange(len(unseen_refs))
            rng.shuffle(unseen_order)
            unseen_count = max(1, int(round(len(unseen_refs) * val_fraction))) if unseen_refs else 0
            selected.extend(unseen_refs[int(i)] for i in unseen_order[:unseen_count])
    else:
        selected = [seen_refs[int(i)] for i in order[val_count:]]
    selected.sort(key=lambda ref: ref.sample_id)
    if max_samples:
        selected = selected[:max_samples]
    return selected


def compute_stats(
    shard_dir: str | Path,
    out_path: str | Path | None = None,
    max_samples: int | None = 2000,
) -> dict[str, Any]:
    refs = _sample_refs(list_shards(shard_dir))
    if max_samples and len(refs) > max_samples:
        step = max(1, len(refs) // max_samples)
        refs = refs[::step][:max_samples]

    params_chunks = []
    pos_chunks = []
    node_pos_chunks = []
    node_disp_chunks = []
    stress_chunks = []
    vm_chunks = []
    contact_chunks = []
    pressure_chunks = []
    cache: dict[Path, Any] = {}
    for ref in refs:
        data = cache.get(ref.shard_path)
        if data is None:
            data = np.load(ref.shard_path, allow_pickle=False)
            cache[ref.shard_path] = data
        i = ref.local_index
        params_chunks.append(_params_from_json(str(data["params_json"][i])))
        node_pos_chunks.append(data["last_nodes"][i, :, 1:4].astype(np.float32))
        node_disp_chunks.append(data["last_nodes"][i, :, 4:7].astype(np.float32))
        pos_chunks.append(data["last_elements"][i, :, 1:4].astype(np.float32))
        stress_chunks.append(data["last_elements"][i, :, 4:10].astype(np.float32))
        vm_chunks.append(data["last_element_von_mises"][i, :, None].astype(np.float32))
        contact_chunks.append(data["last_contact"][i, :, 1:3].astype(np.float32))
        pressure_chunks.append(data["last_contact"][i, :, 2:3].astype(np.float32))

    for data in cache.values():
        data.close()

    params = np.stack(params_chunks)
    node_pos = np.concatenate(node_pos_chunks, axis=0)
    node_disp = np.concatenate(node_disp_chunks, axis=0)
    pos = np.concatenate(pos_chunks, axis=0)
    stress = np.concatenate(stress_chunks, axis=0)
    vm = np.concatenate(vm_chunks, axis=0)
    contact = np.concatenate(contact_chunks, axis=0)
    pressure = np.concatenate(pressure_chunks, axis=0)
    stats = {
        "param_names": PARAM_NAMES,
        "params_mean": params.mean(axis=0).tolist(),
        "params_std": _safe_std(params.std(axis=0)).tolist(),
        "node_pos_mean": node_pos.mean(axis=0).tolist(),
        "node_pos_std": _safe_std(node_pos.std(axis=0)).tolist(),
        "node_disp_mean": node_disp.mean(axis=0).tolist(),
        "node_disp_std": _safe_std(node_disp.std(axis=0)).tolist(),
        "element_pos_mean": pos.mean(axis=0).tolist(),
        "element_pos_std": _safe_std(pos.std(axis=0)).tolist(),
        "stress_mean": stress.mean(axis=0).tolist(),
        "stress_std": _safe_std(stress.std(axis=0)).tolist(),
        "vm_mean": vm.mean(axis=0).tolist(),
        "vm_std": _safe_std(vm.std(axis=0)).tolist(),
        "contact_mean": contact.mean(axis=0).tolist(),
        "contact_std": _safe_std(contact.std(axis=0)).tolist(),
        "pressure_mean": pressure.mean(axis=0).tolist(),
        "pressure_std": _safe_std(pressure.std(axis=0)).tolist(),
        "stats_sample_count": len(refs),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class FootShardDataset(Dataset):
    def __init__(
        self,
        shard_dir: str | Path,
        split: str,
        stats: dict[str, Any],
        val_fraction: float = 0.1,
        seed: int = 42,
        max_samples: int | None = None,
        max_open_shards: int = 4,
        train_base_ids: list[int] | None = None,
        validation_base_ids: list[int] | None = None,
        include_all_unseen_base_ids_in_val: bool = True,
    ) -> None:
        self.refs = split_refs(
            _sample_refs(list_shards(shard_dir)),
            split=split,
            val_fraction=val_fraction,
            seed=seed,
            max_samples=max_samples,
            train_base_ids=train_base_ids,
            validation_base_ids=validation_base_ids,
            include_all_unseen_base_ids_in_val=include_all_unseen_base_ids_in_val,
        )
        self.stats = stats
        self.max_open_shards = max_open_shards
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._contact_features = torch.from_numpy(contact_grid())

    def __len__(self) -> int:
        return len(self.refs)

    def _open(self, path: Path) -> dict[str, np.ndarray]:
        data = self._cache.get(path)
        if data is not None:
            self._cache.move_to_end(path)
            return data
        npz = np.load(path, allow_pickle=False)
        data = {key: npz[key] for key in npz.files}
        npz.close()
        self._cache[path] = data
        if len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return data

    def close(self) -> None:
        self._cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _norm(self, value: np.ndarray, mean_key: str, std_key: str) -> np.ndarray:
        mean = np.asarray(self.stats[mean_key], dtype=np.float32)
        std = np.asarray(self.stats[std_key], dtype=np.float32)
        return (value.astype(np.float32) - mean) / std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        data = self._open(ref.shard_path)
        i = ref.local_index

        params = _params_from_json(str(data["params_json"][i]))
        node_pos = data["last_nodes"][i, :, 1:4].astype(np.float32)
        node_disp = data["last_nodes"][i, :, 4:7].astype(np.float32)
        element_pos = data["last_elements"][i, :, 1:4].astype(np.float32)
        stress = data["last_elements"][i, :, 4:10].astype(np.float32)
        vm = data["last_element_von_mises"][i, :, None].astype(np.float32)
        contact_y = data["last_contact"][i, :, 1:3].astype(np.float32)
        pressure = data["last_contact"][i, :, 2:3].astype(np.float32)
        sole_mask = data["sole_near_element_mask"][i, :, None].astype(np.float32)

        target = np.concatenate(
            [
                self._norm(stress, "stress_mean", "stress_std"),
                self._norm(vm, "vm_mean", "vm_std"),
            ],
            axis=1,
        )

        return {
            "sample_id": torch.tensor(ref.sample_id, dtype=torch.long),
            "base_model_id": torch.tensor(ref.base_model_id, dtype=torch.long),
            "params": torch.from_numpy(self._norm(params, "params_mean", "params_std")),
            "params_raw": torch.from_numpy(params),
            "node_pos": torch.from_numpy(self._norm(node_pos, "node_pos_mean", "node_pos_std")),
            "node_pos_raw": torch.from_numpy(node_pos),
            "node_disp": torch.from_numpy(self._norm(node_disp, "node_disp_mean", "node_disp_std")),
            "node_disp_raw": torch.from_numpy(node_disp),
            "element_pos": torch.from_numpy(
                self._norm(element_pos, "element_pos_mean", "element_pos_std")
            ),
            "element_pos_raw": torch.from_numpy(element_pos),
            "sole_mask": torch.from_numpy(sole_mask),
            "element_y": torch.from_numpy(target),
            "contact_x": self._contact_features.clone(),
            "contact_y": torch.from_numpy(self._norm(contact_y, "contact_mean", "contact_std")),
            "contact_y_raw": torch.from_numpy(contact_y),
            "pressure": torch.from_numpy(self._norm(pressure, "pressure_mean", "pressure_std")),
            "pressure_raw": torch.from_numpy(pressure),
        }
