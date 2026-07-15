from __future__ import annotations

import json
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


PARAM_NAMES = [
    "base_model_id",
    "base_is_training_family",
    "base_foot_length",
    "base_foot_width",
    "base_arch_lift",
    "base_leg_length",
    "base_toe_splay",
    "base_ankle_bend",
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
    "heel_pressure_scale",
    "midfoot_pressure_scale",
    "forefoot_pressure_scale",
    "toe_pressure_scale",
]


HISTORY_PAD_KEYS = {
    "node_history_pos",
    "node_history_disp",
    "node_history_times",
    "node_history_valid_mask",
    "element_history_pos",
    "element_history_y",
    "element_history_times",
    "element_history_valid_mask",
    "contact_history_y",
    "contact_history_raw",
    "contact_history_times",
    "contact_history_valid_mask",
}


def _pad_history_tensors(values: list[torch.Tensor], key: str) -> torch.Tensor:
    max_steps = max(int(value.shape[0]) for value in values)
    trailing_shape = values[0].shape[1:]
    for value in values:
        if value.shape[1:] != trailing_shape:
            raise RuntimeError(
                f"Cannot collate {key}: trailing shapes differ "
                f"({tuple(value.shape)} vs {(int(value.shape[0]), *trailing_shape)})."
            )

    fill_value = 0.0
    if key.endswith("_times"):
        fill_value = 1.0
    padded = []
    for value in values:
        if int(value.shape[0]) == max_steps:
            padded.append(value)
            continue
        out_shape = (max_steps, *trailing_shape)
        out = value.new_full(out_shape, fill_value)
        out[: value.shape[0]] = value
        padded.append(out)
    return torch.stack(padded, dim=0)


def foot_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate foot samples, padding variable-length PINN histories by time.

    The FEBio writer pads histories inside each shard, but different shards can
    still have different time counts. Masks mark real frames, so padding here
    keeps mixed-shard validation batches valid without changing the losses.
    """

    result: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in HISTORY_PAD_KEYS and isinstance(values[0], torch.Tensor):
            step_counts = {int(value.shape[0]) for value in values}
            if len(step_counts) > 1:
                result[key] = _pad_history_tensors(values, key)
                continue
        result[key] = default_collate(values)
    return result


@dataclass(frozen=True)
class SampleRef:
    shard_path: Path
    local_index: int
    sample_id: int
    base_model_id: int
    dataset_id: str = "default"


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
        dataset_ids = data["dataset_ids"].astype(str).tolist() if "dataset_ids" in data.files else ["default"] * len(sample_ids)
        data.close()
        for local_index, sample_id in enumerate(sample_ids):
            params = _params_dict(str(params_json[local_index]))
            refs.append(
                SampleRef(
                    shard_path=shard_path,
                    local_index=local_index,
                    sample_id=sample_id,
                    base_model_id=int(round(float(params.get("base_model_id", 0.0)))),
                    dataset_id=str(dataset_ids[local_index]),
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
    """Compute normalization stats from a bounded sample subset.

    These V9 shards can contain full time histories and are enormous.  The old
    implementation accessed compressed NPZ arrays sample-by-sample, which can
    repeatedly decompress the same shard.  This version groups refs by shard,
    loads each required last-step array once, and accumulates moments without
    keeping all selected samples concatenated in memory.
    """

    shards = list_shards(shard_dir)
    refs = _sample_refs(shards)
    if max_samples and len(refs) > max_samples:
        ids = np.linspace(0, len(refs) - 1, int(max_samples), dtype=int)
        refs = [refs[int(i)] for i in ids]

    def init_acc(width: int) -> dict[str, np.ndarray | int]:
        return {
            "n": 0,
            "sum": np.zeros(width, dtype=np.float64),
            "sum_sq": np.zeros(width, dtype=np.float64),
        }

    def add(acc: dict[str, np.ndarray | int], values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        acc["n"] = int(acc["n"]) + int(arr.shape[0])
        acc["sum"] = np.asarray(acc["sum"]) + arr.sum(axis=0)
        acc["sum_sq"] = np.asarray(acc["sum_sq"]) + np.square(arr).sum(axis=0)

    def finish(acc: dict[str, np.ndarray | int]) -> tuple[list[float], list[float]]:
        n = max(1, int(acc["n"]))
        mean = np.asarray(acc["sum"], dtype=np.float64) / n
        var = np.maximum(np.asarray(acc["sum_sq"], dtype=np.float64) / n - np.square(mean), 0.0)
        return mean.tolist(), _safe_std(np.sqrt(var)).tolist()

    params_chunks: list[np.ndarray] = []
    acc_node_pos = init_acc(3)
    acc_node_disp = init_acc(3)
    acc_element_pos = init_acc(3)
    acc_stress = init_acc(6)
    acc_vm = init_acc(1)
    acc_contact = init_acc(2)
    acc_pressure = init_acc(1)

    refs_by_shard: dict[Path, list[SampleRef]] = {}
    for ref in refs:
        refs_by_shard.setdefault(ref.shard_path, []).append(ref)

    for shard_index, shard_path in enumerate(sorted(refs_by_shard), start=1):
        shard_refs = refs_by_shard[shard_path]
        local_indices = np.asarray([ref.local_index for ref in shard_refs], dtype=int)
        print(
            f"[STATS] shard {shard_index}/{len(refs_by_shard)} "
            f"path={shard_path.name} selected_samples={len(local_indices)}",
            flush=True,
        )
        with np.load(shard_path, allow_pickle=False) as data:
            params_json = data["params_json"]
            for ref in shard_refs:
                params_chunks.append(_params_from_json(str(params_json[ref.local_index])))

            last_nodes = data["last_nodes"][local_indices].astype(np.float32)
            add(acc_node_pos, last_nodes[:, :, 1:4])
            add(acc_node_disp, last_nodes[:, :, 4:7])
            del last_nodes

            last_elements = data["last_elements"][local_indices].astype(np.float32)
            add(acc_element_pos, last_elements[:, :, 1:4])
            add(acc_stress, last_elements[:, :, 4:10])
            del last_elements

            add(acc_vm, data["last_element_von_mises"][local_indices, :, None].astype(np.float32))

            last_contact = data["last_contact"][local_indices].astype(np.float32)
            add(acc_contact, last_contact[:, :, 1:3])
            add(acc_pressure, last_contact[:, :, 2:3])
            del last_contact

    params = np.stack(params_chunks)
    node_pos_mean, node_pos_std = finish(acc_node_pos)
    node_disp_mean, node_disp_std = finish(acc_node_disp)
    element_pos_mean, element_pos_std = finish(acc_element_pos)
    stress_mean, stress_std = finish(acc_stress)
    vm_mean, vm_std = finish(acc_vm)
    contact_mean, contact_std = finish(acc_contact)
    pressure_mean, pressure_std = finish(acc_pressure)

    stats = {
        "param_names": PARAM_NAMES,
        "params_mean": params.mean(axis=0).tolist(),
        "params_std": _safe_std(params.std(axis=0)).tolist(),
        "node_pos_mean": node_pos_mean,
        "node_pos_std": node_pos_std,
        "node_disp_mean": node_disp_mean,
        "node_disp_std": node_disp_std,
        "element_pos_mean": element_pos_mean,
        "element_pos_std": element_pos_std,
        "stress_mean": stress_mean,
        "stress_std": stress_std,
        "vm_mean": vm_mean,
        "vm_std": vm_std,
        "contact_mean": contact_mean,
        "contact_std": contact_std,
        "pressure_mean": pressure_mean,
        "pressure_std": pressure_std,
        "stats_sample_count": len(refs),
        "stats_source_shard_count": len(refs_by_shard),
        "stats_method": "bounded_grouped_npz_last_step_moments",
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
        include_history: bool = False,
        history_time_samples: int | None = None,
        node_history_point_samples: int | None = None,
        element_history_point_samples: int | None = None,
        contact_history_point_samples: int | None = None,
    ) -> None:
        self.split = split
        self.history_time_samples = int(history_time_samples) if history_time_samples else None
        self.node_history_point_samples = int(node_history_point_samples) if node_history_point_samples else None
        self.element_history_point_samples = int(element_history_point_samples) if element_history_point_samples else None
        self.contact_history_point_samples = int(contact_history_point_samples) if contact_history_point_samples else None
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
        self.include_history = bool(include_history)
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._sidecar_cache: OrderedDict[tuple[Path, str], np.ndarray] = OrderedDict()
        self._contact_feature_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def _open(self, path: Path) -> dict[str, np.ndarray]:
        data = self._cache.get(path)
        if data is not None:
            self._cache.move_to_end(path)
            return data
        npz = np.load(path, allow_pickle=False)
        history_prefixes = ("node_history", "element_history", "contact_history")
        data = {
            key: npz[key]
            for key in npz.files
            if self.include_history or not key.startswith(history_prefixes)
        }
        npz.close()
        self._cache[path] = data
        if len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return data

    def _history_sidecar_path(self, shard_path: Path, key: str) -> Path:
        return shard_path.with_name(f"{shard_path.stem}_{key}.npy")

    def _open_sidecar(self, shard_path: Path, key: str) -> np.ndarray | None:
        path = self._history_sidecar_path(shard_path, key)
        if not path.exists():
            return None
        cache_key = (path, key)
        arr = self._sidecar_cache.get(cache_key)
        if arr is not None:
            self._sidecar_cache.move_to_end(cache_key)
            return arr
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        self._sidecar_cache[cache_key] = arr
        if len(self._sidecar_cache) > max(3, self.max_open_shards * 3):
            self._sidecar_cache.popitem(last=False)
        return arr

    def _history_array(self, shard_path: Path, data: dict[str, np.ndarray], key: str, local_index: int) -> np.ndarray | None:
        if key in data:
            return data[key][local_index]
        sidecar = self._open_sidecar(shard_path, key)
        if sidecar is None:
            return None
        return sidecar[local_index]

    def close(self) -> None:
        self._cache.clear()
        self._sidecar_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _norm(self, value: np.ndarray, mean_key: str, std_key: str) -> np.ndarray:
        mean = np.asarray(self.stats[mean_key], dtype=np.float32)
        std = np.asarray(self.stats[std_key], dtype=np.float32)
        return (value.astype(np.float32) - mean) / std

    def _norm_history(self, value: np.ndarray, mean_key: str, std_key: str) -> np.ndarray:
        mean = np.asarray(self.stats[mean_key], dtype=np.float32)
        std = np.asarray(self.stats[std_key], dtype=np.float32)
        return (value.astype(np.float32) - mean.reshape((1, 1, -1))) / std.reshape((1, 1, -1))

    def _rng_for(self, ref: SampleRef, stream: int) -> np.random.Generator:
        if self.split == "train":
            seed = np.random.SeedSequence().entropy
            return np.random.default_rng(seed)
        seed = (int(ref.sample_id) * 1000003 + stream * 9176 + 12345) % (2**32)
        return np.random.default_rng(seed)

    @staticmethod
    def _sample_axis(length: int, count: int | None, rng: np.random.Generator) -> np.ndarray:
        if count is None or count <= 0 or count >= length:
            return np.arange(length, dtype=np.int64)
        return np.sort(rng.choice(length, size=int(count), replace=False).astype(np.int64))

    def _sample_history(
        self,
        ref: SampleRef,
        history: np.ndarray,
        times: np.ndarray,
        mask: np.ndarray,
        point_count: int | None,
        stream: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = self._rng_for(ref, stream)
        time_idx = self._sample_axis(int(history.shape[0]), self.history_time_samples, rng)
        point_idx = self._sample_axis(int(history.shape[1]), point_count, rng)
        history_sub = np.asarray(history[np.ix_(time_idx, point_idx)], dtype=np.float32)
        times_sub = np.asarray(times[time_idx], dtype=np.float32)
        mask_sub = np.asarray(mask[time_idx], dtype=np.float32)
        return history_sub, times_sub, mask_sub, point_idx

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
        contact_count = int(contact_y.shape[0])
        if "contact_pos_norm" in data:
            contact_geom_x = data["contact_pos_norm"][i].astype(np.float32)
            if "contact_region_onehot" in data:
                contact_x_np = np.concatenate(
                    [contact_geom_x, data["contact_region_onehot"][i].astype(np.float32)],
                    axis=1,
                )
            else:
                contact_x_np = contact_geom_x
        else:
            if contact_count not in self._contact_feature_cache:
                self._contact_feature_cache[contact_count] = torch.from_numpy(contact_grid(contact_count))
            contact_x_np = self._contact_feature_cache[contact_count].numpy().astype(np.float32)
            contact_geom_x = contact_x_np

        target = np.concatenate(
            [
                self._norm(stress, "stress_mean", "stress_std"),
                self._norm(vm, "vm_mean", "vm_std"),
            ],
            axis=1,
        )

        item = {
            "sample_id": torch.tensor(ref.sample_id, dtype=torch.long),
            "base_model_id": torch.tensor(ref.base_model_id, dtype=torch.long),
            "dataset_id_hash": torch.tensor(zlib.crc32(ref.dataset_id.encode("utf-8")), dtype=torch.long),
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
            "contact_x": torch.from_numpy(contact_x_np),
            "contact_geom_x": torch.from_numpy(contact_geom_x),
            "contact_y": torch.from_numpy(self._norm(contact_y, "contact_mean", "contact_std")),
            "contact_y_raw": torch.from_numpy(contact_y),
            "pressure": torch.from_numpy(self._norm(pressure, "pressure_mean", "pressure_std")),
            "pressure_raw": torch.from_numpy(pressure),
        }

        if self.include_history:
            node_history_arr = self._history_array(ref.shard_path, data, "node_history", i)
        else:
            node_history_arr = None
        if node_history_arr is not None:
            node_mask_all = data.get("node_history_valid_mask", data.get("node_times_valid_mask", np.isfinite(data["node_times"])))[i].astype(np.float32)
            node_history, node_times, node_mask, _ = self._sample_history(
                ref,
                node_history_arr,
                data["node_times"][i].astype(np.float32),
                node_mask_all,
                self.node_history_point_samples,
                stream=1,
            )
            item["node_history_pos"] = torch.from_numpy(
                np.nan_to_num(self._norm_history(node_history[:, :, 1:4], "node_pos_mean", "node_pos_std"))
            )
            item["node_history_disp"] = torch.from_numpy(
                np.nan_to_num(self._norm_history(node_history[:, :, 4:7], "node_disp_mean", "node_disp_std"))
            )
            item["node_history_times"] = torch.from_numpy(np.nan_to_num(node_times, nan=1.0))
            item["node_history_valid_mask"] = torch.from_numpy(node_mask)
        if self.include_history:
            element_history_arr = self._history_array(ref.shard_path, data, "element_history", i)
        else:
            element_history_arr = None
        if element_history_arr is not None:
            element_mask_all = data.get("element_history_valid_mask", data.get("element_times_valid_mask", np.isfinite(data["element_times"])))[i].astype(np.float32)
            element_history, element_times, element_mask, _ = self._sample_history(
                ref,
                element_history_arr,
                data["element_times"][i].astype(np.float32),
                element_mask_all,
                self.element_history_point_samples,
                stream=2,
            )
            hist_stress = self._norm_history(element_history[:, :, 4:10], "stress_mean", "stress_std")
            hist_vm = np.asarray(
                [
                    np.sqrt(
                        0.5
                        * (
                            (step[:, 4] - step[:, 5]) ** 2
                            + (step[:, 5] - step[:, 6]) ** 2
                            + (step[:, 6] - step[:, 4]) ** 2
                        )
                        + 3.0 * (step[:, 7] ** 2 + step[:, 8] ** 2 + step[:, 9] ** 2)
                    )
                    for step in element_history
                ],
                dtype=np.float32,
            )[:, :, None]
            item["element_history_pos"] = torch.from_numpy(
                np.nan_to_num(self._norm_history(element_history[:, :, 1:4], "element_pos_mean", "element_pos_std"))
            )
            item["element_history_y"] = torch.from_numpy(
                np.nan_to_num(np.concatenate([hist_stress, self._norm_history(hist_vm, "vm_mean", "vm_std")], axis=-1))
            )
            item["element_history_times"] = torch.from_numpy(np.nan_to_num(element_times, nan=1.0))
            item["element_history_valid_mask"] = torch.from_numpy(element_mask)
        if self.include_history:
            contact_history_arr = self._history_array(ref.shard_path, data, "contact_history", i)
        else:
            contact_history_arr = None
        if contact_history_arr is not None:
            contact_mask_all = data.get("contact_history_valid_mask", data.get("contact_times_valid_mask", np.isfinite(data["contact_times"])))[i].astype(np.float32)
            contact_history, contact_times, contact_mask, contact_point_idx = self._sample_history(
                ref,
                contact_history_arr,
                data["contact_times"][i].astype(np.float32),
                contact_mask_all,
                self.contact_history_point_samples,
                stream=3,
            )
            item["contact_history_y"] = torch.from_numpy(
                np.nan_to_num(self._norm_history(contact_history[:, :, 1:3], "contact_mean", "contact_std"))
            )
            item["contact_history_raw"] = torch.from_numpy(np.nan_to_num(contact_history[:, :, 1:3]))
            item["contact_history_times"] = torch.from_numpy(np.nan_to_num(contact_times, nan=1.0))
            item["contact_history_valid_mask"] = torch.from_numpy(contact_mask)
            contact_idx = torch.from_numpy(contact_point_idx.astype(np.int64))
            item["contact_history_x"] = item["contact_x"].index_select(0, contact_idx)
        return item
