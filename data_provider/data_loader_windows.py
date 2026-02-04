import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_WindowsT0(Dataset):
    """
    windows_t0: 你的 npz 是窗口级样本
      X: [N, L, D]，其中：
         X[...,0] = timestamp(用于time mark)
         X[...,1] = timestamp(可忽略)
         X[...,2:] = exog(未来已知协变量)
      Y_before / Y_real / Y: [N, L, Dy] 或 [N, L]
      t0_idx: 所有样本相同的分界点（历史/未来）
    输出给 TSLib：
      x      : [seq_len, exog_dim + c_out]   = [exog_hist, y_hist]
      y      : [label_len+pred_len, exog_dim + c_out] = [exog_seg,  y_seg]
      x_mark : [seq_len, mark_dim]
      y_mark : [label_len+pred_len, mark_dim]
    """
    def __init__(self, args,
                 root_path,
                 data_path,
                 flag='train',
                 size=None,
                 features='M',
                 target='y',
                 timeenc=0,
                 freq='h',
                 seasonal_patterns=None):

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

        # ---------- load npz ----------
        fp = os.path.join(root_path, f'wea_data_{data_path}')
        data = np.load(fp, allow_pickle=True)

        X = data['X']  # [N, L, D] dtype may be object because timestamps
        if hasattr(args, "use_real_y") and args.use_real_y:
            key = "Y_real" if "Y_real" in data.files else ("Y" if "Y" in data.files else None)
        else:
            key = "Y_before" if "Y_before" in data.files else ("Y" if "Y" in data.files else None)

        if key is None or key not in data.files:
            raise KeyError(f"Cannot find Y_before/Y_real/Y in npz. keys={data.files}")
        Y = data[key]

        # t0_idx (optional sanity)
        self.t0_idx = int(data['t0_idx']) if np.ndim(data['t0_idx']) == 0 else int(data['t0_idx'].item())

        N, L, D = X.shape
        self.N_all = N
        self.L = L
        self.D_all = D

        # ---------- split ratios (window-level, ordered to avoid leakage) ----------
        train_ratio = float(getattr(args, "train_ratio", 0.7))
        val_ratio = float(getattr(args, "val_ratio", 0.1))
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        n_test = N - n_train - n_val

        if flag.lower() == "train":
            sl = slice(0, n_train)
        elif flag.lower() in ["val", "valid"]:
            sl = slice(n_train, n_train + n_val)
        elif flag.lower() == "test":
            sl = slice(n_train + n_val, N)
        else:
            raise ValueError("flag must be train/val/test")

        # ---------- timestamps & exog ----------
        ts = X[:, :, 0]          # [N, L] timestamp-like
        X_num = X[:, :, 2:]      # [N, L, D-2] exog only
        X_num = X_num.astype(np.float32)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Y cleanup ----------
        Y = Y.astype(np.float32)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        if Y.ndim == 2:
            Y = Y[..., None]  # [N, L, 1]
        Dy = Y.shape[2]

        # choose y columns (default: 0)
        if hasattr(args, "y_cols") and getattr(args, "y_cols"):
            y_cols = [int(x) for x in str(args.y_cols).split(",") if str(x).strip() != ""]
        else:
            y_cols = [0]

        for c in y_cols:
            if c < 0 or c >= Dy:
                raise ValueError(f"y_cols out of range: {y_cols}, Dy={Dy}")
        self.y_cols = y_cols
        self.c_out = len(y_cols)

        # ---------- normalize Y using TRAIN only (avoid leakage) ----------
        y_scale = float(np.nanmax(Y[:n_train, :, :][:, :, self.y_cols]))
        if (not np.isfinite(y_scale)) or y_scale == 0:
            y_scale = 1.0
        self.y_scale = y_scale

        Y = Y / self.y_scale

        # dataset interface flags for exp/test
        self.scale = True
        self.scaler = None

        # ---------- apply split ----------
        self.X_num = X_num[sl]  # [N', L, exog_dim]
        self.Y = Y[sl]          # [N', L, Dy] normalized
        self.ts = ts[sl]        # [N', L]

        # time feature usage
        self.use_time_features = (time_features is not None) and (timeenc == 1)

        # enc_in reported for reference (exog_dim)
        self.exog_dim = self.X_num.shape[2]
        self.enc_in = self.exog_dim  # NOTE: real model enc_in should be exog_dim + c_out (see run args)

        # sanity checks
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
        # history exog
        x_exog = self.X_num[idx, :self.seq_len, :]  # [seq_len, exog_dim]

        # history y (selected cols)
        x_y = self.Y[idx, :self.seq_len, :][:, self.y_cols]  # [seq_len, c_out]

        # x = [exog, y]
        x = np.concatenate([x_exog, x_y], axis=1)  # [seq_len, exog_dim + c_out]

        # segment for decoder: label_len + pred_len
        y_begin = self.seq_len - self.label_len
        y_end = self.seq_len + self.pred_len

        y_exog = self.X_num[idx, y_begin:y_end, :]              # [label+pred, exog_dim]
        y_y = self.Y[idx, y_begin:y_end, :][:, self.y_cols]     # [label+pred, c_out]
        y = np.concatenate([y_exog, y_y], axis=1)               # [label+pred, exog_dim + c_out]

        # marks
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

    def inverse_transform(self, data):
        # data can be numpy or torch
        return data * self.y_scale