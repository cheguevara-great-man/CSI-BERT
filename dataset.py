from torch.utils.data import Dataset
import numpy as np
import os
import glob
import csv
from collections import OrderedDict
from scipy.interpolate import interp1d
import torch

class CSI_dataset(Dataset):
    def __init__(self, magnitudes, phases=None, timestamp=None, label_action=None, label_people=None):
        super().__init__()
        self.magnitudes = magnitudes
        self.phases = phases
        self.timestamp=timestamp
        self.label_action = label_action
        self.label_people = label_people
        self.num=self.magnitudes.shape[0]
        if self.phases is None:
            self.timestamp = [-1] * len(self.num)
        if self.timestamp is None:
            self.timestamp = [-1] * len(self.num)
        if self.label_action is None:
            self.label_action = [-1] * len(self.num)
        if self.label_people is None:
            self.label_people = [-1] * len(self.num)


    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.magnitudes[index],self.phases[index], self.label_action[index], self.label_people[index], self.timestamp[index]


def load_zero_people(test_people_list):
    magnitude=np.load("./data/magnitude.npy").astype(np.float32)
    phase=np.load("./data/phase.npy").astype(np.float32)
    timestamp=np.load("./data/timestamp.npy").astype(np.float32)
    people=np.load("./data/people.npy").astype(np.int64)
    action=np.load("./data/action.npy").astype(np.int64)
    a = np.zeros_like(people)
    b = np.zeros_like(people)
    for i in range(people.shape[0]):
        a[i]=(people[i] not in test_people_list)
        b[i]=not a[i]
    a=a.astype(bool)
    b=b.astype(bool)
    return CSI_dataset(magnitude[a], phase[a], timestamp[a], action[a], people[a]),CSI_dataset(magnitude[b], phase[b], timestamp[a], action[b], people[b])

def load_all(magnitude_path="./data/magnitude.npy",phase_path="./data/magnitude.npy"):
    magnitude=np.load(magnitude_path).astype(np.float32)
    phase=np.load(phase_path).astype(np.float32)
    phase[np.isnan(phase)] = -1000
    timestamp=np.load("./data/timestamp.npy").astype(np.float32)
    people=np.load("./data/people.npy").astype(np.int64)
    action=np.load("./data/action.npy").astype(np.int64)
    return CSI_dataset(magnitude, phase, timestamp, action, people)

def load_data(magnitude_path="./data/magnitude.npy",train_prop=None):
    magnitude=np.load(magnitude_path).astype(np.float32)
    phase=np.load("./data/phase.npy").astype(np.float32)
    timestamp=np.load("./data/timestamp.npy").astype(np.float32)
    people=np.load("./data/people.npy").astype(np.int64)
    action=np.load("./data/action.npy").astype(np.int64)
    if train_prop is None:
        return CSI_dataset(magnitude, phase, timestamp, action, people)
    else:
        a = np.zeros_like(people)
        num=[]
        current_num=0
        current_action=None
        for i in range(action.shape[0]):
            if action[i]==current_action:
                current_num+=1
            else:
                current_action = action[i]
                if current_action is None:
                    current_num+=1
                else:
                    num.append(current_num)
                    current_num=0
        num.append(current_num)
        current_num=0
        for i in range(len(num)):
            a[current_num:current_num+int(num[i]*train_prop)]=1
            current_num+=num[i]
        b=1-a
        a = a.astype(bool)
        b = b.astype(bool)
        return CSI_dataset(magnitude[a], phase[a], timestamp[a], action[a], people[a]), CSI_dataset(magnitude[b], phase[b], timestamp[b], action[b], people[b])


def resample_signal_data(x, sample_rate, sample_method, use_mask_0, interpolation_method, is_rec=0):
    """
    Signal resampling/interpolation for input x shaped (T, F).
    """
    if sample_rate >= 1.0:
        if is_rec:
            mask = np.ones_like(x, dtype=np.float32)
            return x, mask
        return x

    original_len = x.shape[0]
    resample_len = int(original_len * sample_rate)

    if sample_method == "uniform_nearest":
        pick_indices_float = np.linspace(0, original_len - 1, resample_len)
        pick_indices_int = np.round(pick_indices_float).astype(int)
    elif sample_method == "equidistant":
        step = original_len / resample_len
        pick_indices_int = np.arange(0, original_len, step).astype(int)[:resample_len]
    elif sample_method == "gaussian":
        intervals = np.random.normal(loc=1.0, scale=0.5, size=resample_len - 1)
        intervals = np.abs(intervals)
        total_duration = original_len - 1
        intervals = intervals / intervals.sum() * total_duration
        pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
        pick_indices_int = np.round(pick_indices_float).astype(int)
    elif sample_method == "poisson":
        intervals = np.random.exponential(scale=1.0, size=resample_len - 1)
        total_duration = original_len - 1
        intervals = intervals / intervals.sum() * total_duration
        pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
        pick_indices_int = np.round(pick_indices_float).astype(int)
    else:
        raise ValueError(f"Unknown sample method: {sample_method}")

    pick_indices_int = np.unique(pick_indices_int)

    mask = np.zeros_like(x, dtype=np.float32)
    mask[pick_indices_int, :] = 1.0

    if use_mask_0 == 1:
        x_sparse = np.zeros_like(x)
        x_sparse[pick_indices_int, :] = x[pick_indices_int, :]
        x = x_sparse
    elif use_mask_0 == 2:
        x = x[pick_indices_int, :]
    else:
        x_downsampled = x[pick_indices_int, :]
        x_known = pick_indices_int
        x_new = np.arange(original_len)
        y_known = x_downsampled
        if interpolation_method in ["linear", "nearest", "cubic"]:
            interp_kind = interpolation_method
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        f_interp = interp1d(x_known, y_known, kind=interp_kind, axis=0, bounds_error=False, fill_value="extrapolate")
        x = f_interp(x_new)

    if is_rec:
        return x, mask
    return x


def _is_digit_gesture(name: str) -> bool:
    if not isinstance(name, str):
        return False
    if not name.startswith("Draw-"):
        return False
    suf = name.split("Draw-")[-1]
    return suf.isdigit() and 0 <= int(suf) <= 9


class WidarDigitShardDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        variant: str = "amp",
        split: str = "train",
        digits_only: bool = True,
        shard_cache: int = 2,
        is_rec: int = 1,
        sample_rate: float = 1.0,
        sample_method: str = "uniform_nearest",
        interpolation_method: str = "linear",
        use_mask_0: int = 0,
        traffic_train_pt: str = None,
        traffic_test_pt: str = None,
        **kwargs,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.variant = variant
        self.split = split
        self.digits_only = digits_only
        self.shard_cache = max(int(shard_cache), 0)
        self.return_rec = int(is_rec)

        self.sample_rate = float(sample_rate)
        self.sample_method = sample_method
        self.interpolation_method = interpolation_method
        self.use_mask_0 = int(use_mask_0)

        if variant not in ("amp", "conj"):
            raise ValueError("variant must be 'amp' or 'conj'")
        if split in ("train", "test"):
            split_root = os.path.join(root_dir, split)
        elif split == "all":
            split_root = root_dir
        else:
            raise ValueError("split must be train/test/all (new reshards layout)")

        self.shard_dir = os.path.join(split_root, variant, "shards")
        self.x_key = "X"

        index_path = os.path.join(split_root, "meta", "index.csv")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"index.csv not found: {index_path}\n"
                f"Expected root_dir like: .../Widar_digit (containing amp/, conj/, meta/)"
            )

        rows = []
        with open(index_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    gesture_name = r.get("gesture_name", "")
                    label = int(float(r.get("label", -1)))
                    shard_id = int(float(r.get("shard_id", -1)))
                    offset = int(float(r.get("offset", -1)))
                    sample_id = r.get("sample_id", "")
                    if digits_only and gesture_name and (not _is_digit_gesture(gesture_name)):
                        continue
                    if label < 0 or shard_id < 0 or offset < 0:
                        continue
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "gesture_name": gesture_name,
                            "label": label,
                            "shard_id": shard_id,
                            "offset": offset,
                        }
                    )
                except Exception:
                    continue

        if len(rows) == 0:
            raise RuntimeError(
                f"No valid rows found in {index_path}. "
                f"Check columns (sample_id,label,gesture_name,shard_id,offset) and digits_only={digits_only}."
            )

        self.items = rows
        # trafficlike offline masks
        self._traffic_active = (self.sample_method == "trafficlike")
        self._traffic_masks = None
        self._traffic_rate = None
        self._traffic_perm = None
        self._traffic_subset = None
        self._traffic_rate_idx = None
        self._traffic_epoch = 0
        if self._traffic_active:
            pt_path = traffic_train_pt if split == "train" else traffic_test_pt
            if not pt_path:
                raise ValueError("trafficlike requires traffic_train_pt/test_pt")
            payload = torch.load(pt_path, map_location="cpu")
            masks = payload.get("masks", None)
            if masks is None:
                raise KeyError(f"'masks' not found in {pt_path}")
            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()
            masks = np.asarray(masks)
            if masks.ndim != 2:
                raise ValueError(f"traffic masks must be 2D (N,T), got {masks.shape}")
            self._traffic_masks = masks.astype(np.uint8, copy=False)
            rate = payload.get("rate_hz", None)
            if rate is not None:
                if torch.is_tensor(rate):
                    rate = rate.cpu().numpy()
                rate = np.asarray(rate)
                if rate.ndim == 1 and rate.shape[0] == self._traffic_masks.shape[0]:
                    self._traffic_rate = rate.astype(np.int16, copy=False)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch: int):
        if not self._traffic_active:
            return
        self._traffic_epoch = int(epoch)
        rng = np.random.RandomState(self._traffic_epoch)
        self._traffic_perm = rng.permutation(self._traffic_masks.shape[0])
        self._traffic_subset = None

    def set_eval_subset(self, n: int, seed: int = None):
        if not self._traffic_active:
            return
        n = int(n)
        if n <= 0:
            self._traffic_subset = None
            return
        pool = self._traffic_rate_idx
        if pool is None:
            pool = np.arange(self._traffic_masks.shape[0], dtype=np.int64)
        if pool.size == 0:
            self._traffic_subset = None
            return
        rng = np.random.RandomState(int(seed) if seed is not None else self._traffic_epoch)
        replace = n > pool.size
        self._traffic_subset = rng.choice(pool, size=n, replace=replace)

    def set_rate_filter(self, rate_hz):
        if not self._traffic_active:
            return
        if rate_hz is None:
            self._traffic_rate_idx = None
            self._traffic_subset = None
            return
        if self._traffic_rate is None:
            raise RuntimeError("trafficlike rate_hz not available in masks file")
        rate = int(rate_hz)
        idx = np.nonzero(self._traffic_rate == rate)[0]
        self._traffic_rate_idx = idx
        self._traffic_subset = None

    def get_available_rates(self):
        if (not self._traffic_active) or (self._traffic_rate is None):
            return []
        rates = np.unique(self._traffic_rate)
        return [int(r) for r in rates.tolist()]

    def _pick_traffic_mask_index(self, idx: int) -> int:
        if self._traffic_masks is None:
            raise RuntimeError("trafficlike masks not loaded")
        if self._traffic_subset is not None:
            subset = self._traffic_subset
            return int(subset[idx % subset.shape[0]])
        if self._traffic_rate_idx is not None:
            pool = self._traffic_rate_idx
            if pool.size > 0:
                return int(pool[idx % pool.shape[0]])
        if self._traffic_perm is None:
            rng = np.random.RandomState(None)
            self._traffic_perm = rng.permutation(self._traffic_masks.shape[0])
        return int(self._traffic_perm[idx % self._traffic_perm.shape[0]])

    def _resolve_shard_path(self, shard_id: int) -> str:
        cand = [
            os.path.join(self.shard_dir, f"shard-{shard_id:05d}.npz"),
            os.path.join(self.shard_dir, f"shard-{(shard_id + 1):05d}.npz"),
            os.path.join(self.shard_dir, f"shard-{max(shard_id - 1, 0):05d}.npz"),
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        g = glob.glob(os.path.join(self.shard_dir, f"*{shard_id}*.npz"))
        if g:
            return g[0]
        raise FileNotFoundError(f"Cannot find shard file for shard_id={shard_id} under {self.shard_dir}")

    def _load_shard(self, shard_id: int):
        if self.shard_cache > 0 and shard_id in self._cache:
            obj = self._cache.pop(shard_id)
            self._cache[shard_id] = obj
            return obj

        shard_path = self._resolve_shard_path(shard_id)
        npz = np.load(shard_path, allow_pickle=True)
        obj = {
            self.x_key: npz[self.x_key].astype(np.float32, copy=False),
            "y": npz["y"].astype(np.int64, copy=False),
        }
        npz.close()

        if self.shard_cache > 0:
            self._cache[shard_id] = obj
            while len(self._cache) > self.shard_cache:
                self._cache.popitem(last=False)
        return obj

    def __getitem__(self, idx):
        row = self.items[idx]
        shard_id = row["shard_id"]
        shard = self._load_shard(shard_id)
        off = int(row["offset"])
        label = int(row["label"])
        x = shard[self.x_key][off]
        if x.ndim == 3:
            x = x[0]
        x_original = x.copy()
        mask = np.ones((x.shape[0],), dtype=np.float32)

        if self.sample_method == "trafficlike":
            mask_idx = self._pick_traffic_mask_index(idx)
            mask_1d = self._traffic_masks[mask_idx]
            if mask_1d.shape[0] != x.shape[0]:
                raise ValueError(f"traffic mask length {mask_1d.shape[0]} != T {x.shape[0]}")
            mask = mask_1d.astype(np.float32, copy=False)[:, None]
            if x.shape[1] != 1:
                mask = np.repeat(mask, x.shape[1], axis=1)
            x = x * mask
        elif self.sample_rate < 1.0 or self.use_mask_0:
            x, mask = resample_signal_data(
                x,
                sample_rate=self.sample_rate,
                sample_method=self.sample_method,
                use_mask_0=self.use_mask_0,
                interpolation_method=self.interpolation_method,
                is_rec=1,
            )
        else:
            mask = np.ones_like(x, dtype=np.float32)

        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        y_t = torch.tensor(label, dtype=torch.long)
        mask_t = torch.from_numpy(mask.astype(np.float32, copy=False))
        x_gt_t = torch.from_numpy(x_original.astype(np.float32, copy=False))
        timestamp = torch.arange(x_t.shape[0], dtype=torch.long)

        if self.return_rec == 0:
            return x_t, mask_t, y_t, timestamp
        return x_t, mask_t, y_t, x_gt_t, timestamp


def Widar_digit_amp_dataset(root_dir, split="train", **kwargs):
    return WidarDigitShardDataset(root_dir=root_dir, variant="amp", split=split, **kwargs)
