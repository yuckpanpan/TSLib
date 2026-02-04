import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


class Dataset_WindowsT0(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None,
                 features='M', target='y', timeenc=0, freq='h', seasonal_patterns=None):

        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        if size is None:
            raise ValueError("size must be provided: [seq_len, label_len, pred_len]")
        self.seq_len, self.label_len, self.pred_len = map(int, size)

        # ---------- Load Data ----------
        # 兼容处理文件名：支持传 14_14_with_Y_PV 或 14_14_with_Y_PV.npz

        candidates = [
            os.path.join(root_path, f'wea_data_{data_path}.npz'),
            os.path.join(root_path, f'wea_data_{data_path}'),
            os.path.join(root_path, f'{data_path}.npz'),
            os.path.join(root_path, data_path),
        ]

        fp = None
        for c in candidates:
            if os.path.exists(c):
                fp = c
                break
        if fp is None:
            raise FileNotFoundError(f"Cannot find npz file. Tried: {candidates}")

        data = np.load(fp, allow_pickle=True)
        X = data['X']  # [N, L, D]

        # ---------- Choose Y key (FIXED) ----------
        use_real = bool(getattr(args, "use_real_y", False))
        if use_real and 'Y_real' in data.files:
            key = 'Y_real'
        elif 'Y_before' in data.files:
            key = 'Y_before'
        elif 'Y' in data.files:
            key = 'Y'
        elif 'Y_real' in data.files:
            # 兜底：只有 Y_real 时就用它
            key = 'Y_real'
        else:
            raise KeyError(f"Cannot find Y in npz. Available keys={data.files}")
        Y = data[key]

        # t0_idx（不一定用，但保留）
        self.t0_idx = int(data['t0_idx']) if 't0_idx' in data.files else None

        N, L, D = X.shape
        self.N_all, self.L, self.D_all = N, L, D

        # ---------- Split Ratios ----------
        train_ratio = float(getattr(args, "train_ratio", 0.7))
        val_ratio = float(getattr(args, "val_ratio", 0.1))
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)

        if flag.lower() == "train":
            sl = slice(0, n_train)
        elif flag.lower() in ["val", "valid"]:
            sl = slice(n_train, n_train + n_val)
        elif flag.lower() == "test":
            sl = slice(n_train + n_val, N)
        else:
            raise ValueError("flag must be train/val/test")

        # ---------- Preprocess X ----------
        # X[...,0] 是 timestamp 用于 mark
        ts = X[:, :, 0]
        # X[...,1] 忽略；X[...,2:] 是未来已知协变量
        X_num = X[:, :, 2:].astype(np.float32)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Preprocess Y ----------
        Y = Y.astype(np.float32)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        if Y.ndim == 2:
            Y = Y[..., None]
        Dy = Y.shape[2]

        # y_cols
        if hasattr(args, "y_cols") and getattr(args, "y_cols"):
            y_cols = [int(x) for x in str(args.y_cols).split(",") if str(x).strip() != ""]
        else:
            y_cols = [0]

        # bounds check (FIXED)
        for c in y_cols:
            if c < 0 or c >= Dy:
                raise ValueError(f"y_cols out of range: {y_cols}, Dy={Dy}")

        self.y_cols = y_cols
        self.c_out = len(y_cols)

        # ---------- Scaling (TSLib style: fit on TRAIN only) ----------
        self.scale = bool(getattr(args, "scale", True))

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler_x = StandardScaler()

            # Fit Y on train only (selected cols)
            train_y = Y[:n_train, :, :][:, :, self.y_cols].reshape(-1, self.c_out)
            self.scaler.fit(train_y)

            # Transform all Y (selected cols)
            Y_sel = Y[:, :, self.y_cols].reshape(-1, self.c_out)
            Y_sel = self.scaler.transform(Y_sel).reshape(N, L, self.c_out).astype(np.float32)
            Y[:, :, self.y_cols] = Y_sel

            # Fit X on train only (all exog dims)
            Dx = X_num.shape[2]
            if Dx > 0:
                train_x = X_num[:n_train, :, :].reshape(-1, Dx)
                self.scaler_x.fit(train_x)

                X_flat = X_num.reshape(-1, Dx)
                X_num = self.scaler_x.transform(X_flat).reshape(N, L, Dx).astype(np.float32)
        else:
            self.scaler = None
            self.scaler_x = None

        # ---------- Apply Split ----------
        self.X_num = X_num[sl]
        self.Y = Y[sl]
        self.ts = ts[sl]

        self.use_time_features = (time_features is not None) and (timeenc == 1)

        self.exog_dim = self.X_num.shape[2]
        # x 的实际宽度是 exog_dim + c_out（你 run.py 里 enc_in/dec_in 应该传这个）
        self.enc_in = self.exog_dim + self.c_out

        # sanity
        if self.seq_len + self.pred_len > self.L:
            raise ValueError(f"seq_len+pred_len must <= L. Got {self.seq_len}+{self.pred_len}>{self.L}")
        if self.label_len > self.seq_len:
            raise ValueError(f"label_len must <= seq_len. Got {self.label_len}>{self.seq_len}")

    def __len__(self):
        return self.X_num.shape[0]

    def _make_mark(self, ts_1d):
        if not self.use_time_features:
            return np.zeros((len(ts_1d), 4), dtype=np.float32)
        dt = pd.to_datetime(ts_1d)
        feats = time_features(dt, freq=self.freq)
        feats = np.asarray(feats)
        if feats.ndim == 2 and feats.shape[0] < feats.shape[1]:
            feats = feats.T
        return feats.astype(np.float32)

    def __getitem__(self, idx):
        x_exog = self.X_num[idx, :self.seq_len, :]
        x_y = self.Y[idx, :self.seq_len, :][:, self.y_cols]
        x = np.concatenate([x_exog, x_y], axis=1)

        y_begin = self.seq_len - self.label_len
        y_end = self.seq_len + self.pred_len

        y_exog = self.X_num[idx, y_begin:y_end, :]
        y_y = self.Y[idx, y_begin:y_end, :][:, self.y_cols]
        y = np.concatenate([y_exog, y_y], axis=1)

        ts_x = self.ts[idx, :self.seq_len]
        ts_y = self.ts[idx, y_begin:y_end]
        x_mark = self._make_mark(ts_x)
        y_mark = self._make_mark(ts_y)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(x_mark).float(),
            torch.from_numpy(y_mark).float(),
        )

    def inverse_transform(self, data, strict=True):
        """
        反标准化：只对 y 通道做 inverse。
        - strict=True：要求输入最后一维 == c_out，否则报错（推荐，避免悄悄吞错）
        - strict=False：若最后一维 > c_out，则默认取最后 c_out 列来 inverse（兼容旧逻辑）
        """
        if (not self.scale) or (self.scaler is None):
            return data

        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
            is_tensor = True
            device = data.device
            dtype = data.dtype
        else:
            data_np = np.asarray(data)
            is_tensor = False
            device = None
            dtype = None

        if data_np.shape[-1] != self.c_out:
            if strict:
                raise ValueError(f"inverse_transform expects last dim == c_out({self.c_out}), got {data_np.shape[-1]}")
            if data_np.shape[-1] < self.c_out:
                raise ValueError(f"inverse_transform last dim < c_out({self.c_out}), got {data_np.shape[-1]}")
            data_np = data_np[..., -self.c_out:]

        shp = data_np.shape
        inv = self.scaler.inverse_transform(data_np.reshape(-1, self.c_out)).reshape(shp)

        if is_tensor:
            return torch.from_numpy(inv).to(device=device, dtype=dtype)
        return inv