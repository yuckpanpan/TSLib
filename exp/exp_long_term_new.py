from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping  # adjust_learning_rate/visual 不再强依赖
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @property
    def _exog_dim(self):
        # dec_in = exog_dim + c_out
        return self.args.dec_in - self.args.c_out

    def _build_dec_inp(self, batch_y):
        """
        batch_y: [B, label+pred, dec_in] = [exog, y]
        dec_inp: keep exog for all; keep y only for label part; y future = 0
        """
        B, T, _ = batch_y.shape
        dec_inp = batch_y.new_zeros((B, T, self.args.dec_in))
        exog_dim = self._exog_dim
        dec_inp[:, :, :exog_dim] = batch_y[:, :, :exog_dim]
        if self.args.label_len > 0:
            dec_inp[:, :self.args.label_len, exog_dim:] = batch_y[:, :self.args.label_len, exog_dim:]
        return dec_inp

    def _select_y_true(self, batch_y):
        return batch_y[:, -self.args.pred_len:, -self.args.c_out:]

    def _select_y_pred(self, outputs):
        # outputs could be [B, pred_len, c_out] or [B, pred_len, dec_in]
        return outputs[:, -self.args.pred_len:, -self.args.c_out:]

    def _run_epoch(self, loader, criterion, optimizer=None):
        """
        train mode if optimizer is not None, else eval mode.
        returns average loss over loader.
        """
        is_train = optimizer is not None
        self.model.train(is_train)

        losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.to(self.device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(self.device, dtype=torch.float32, non_blocking=True)
            batch_x_mark = batch_x_mark.to(self.device, dtype=torch.float32, non_blocking=True)
            batch_y_mark = batch_y_mark.to(self.device, dtype=torch.float32, non_blocking=True)

            dec_inp = self._build_dec_inp(batch_y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_y = self._select_y_pred(outputs)
            true_y = self._select_y_true(batch_y)

            loss = criterion(pred_y, true_y)

            if is_train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses)) if losses else float('nan')

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            t0 = time.time()

            train_loss = self._run_epoch(train_loader, criterion, optimizer=optimizer)
            vali_loss = self._run_epoch(vali_loader, criterion, optimizer=None)
            test_loss = self._run_epoch(test_loader, criterion, optimizer=None)

            print(f"Epoch: {epoch+1} cost time: {time.time() - t0:.4f}")
            print(f"Epoch: {epoch+1}, Steps: {len(train_loader)} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')

        if test:
            ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.float32, non_blocking=True)

                dec_inp = self._build_dec_inp(batch_y)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred_y = self._select_y_pred(outputs)   # torch, on GPU
                true_y = self._select_y_true(batch_y)

                preds.append(pred_y.detach().cpu().numpy())
                trues.append(true_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # inverse ONCE, only if requested AND y was scaled
        do_inverse = bool(getattr(self.args, "inverse", 0)) and getattr(test_data, "scale", False) and hasattr(test_data, "inverse_transform")
        if do_inverse:
            shp = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, shp[-1])).reshape(shp)
            trues = test_data.inverse_transform(trues.reshape(-1, shp[-1])).reshape(shp)
            print('Data inverse transformed for metric calculation.')

        # (optional) save results, default off to reduce IO
        if bool(getattr(self.args, "save_np", 0)):
            res_path = os.path.join('./results/', setting)
            os.makedirs(res_path, exist_ok=True)
            np.save(os.path.join(res_path, 'pred.npy'), preds)
            np.save(os.path.join(res_path, 'true.npy'), trues)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')


        # 额外再打印 WAPE（建议主看这个）
        wape = np.sum(np.abs(preds - trues)) / np.maximum(np.sum(np.abs(trues)), 1e-6) * 100
        print(f"wape:{wape}")

        # 再加一个业务阈值 mask 的 mape（诊断你 mask 是否太宽）
        y = trues
        p = preds
        thr = 0.02 * np.max(np.abs(y))
        mask = np.abs(y) > thr
        if mask.any():
            mape2 = np.mean(np.abs((p[mask] - y[mask]) / (y[mask] + 1e-6))) * 100
            print(f"mape_mask@2%max:{mape2}, mask_ratio:{mask.mean()}")
        else:
            print("mape_mask@2%max: mask empty")

        return