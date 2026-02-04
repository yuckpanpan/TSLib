from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import accelerated_dtw

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _build_dec_inp(self, batch_x, batch_y):
        B = batch_y.shape[0]
        T = self.args.label_len + self.args.pred_len
        dec_inp = torch.zeros([B, T, self.args.dec_in], device=batch_x.device)

        exog_dim = self.args.dec_in - self.args.c_out
        dec_inp[:, :, :exog_dim] = batch_y[:, :, :exog_dim]  # exog full
        dec_inp[:, :self.args.label_len, exog_dim:] = batch_y[:, :self.args.label_len, exog_dim:]  # y label only
        return dec_inp

    def _select_y_true(self, batch_y):
        return batch_y[:, -self.args.pred_len:, -self.args.c_out:]

    def _select_y_pred(self, outputs):
        # outputs might be [B, pred_len, c_out] or [B, pred_len, dec_in]
        return outputs[:, -self.args.pred_len:, -self.args.c_out:]

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._build_dec_inp(batch_x, batch_y)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred_y = self._select_y_pred(outputs)
                true_y = self._select_y_true(batch_y)

                loss = criterion(pred_y, true_y)
                total_loss.append(loss.item())

        self.model.train()
        return float(np.mean(total_loss))

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._build_dec_inp(batch_x, batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        pred_y = self._select_y_pred(outputs)
                        true_y = self._select_y_true(batch_y)
                        loss = criterion(pred_y, true_y)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    pred_y = self._select_y_pred(outputs)
                    true_y = self._select_y_true(batch_y)
                    loss = criterion(pred_y, true_y)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss_epoch = float(np.mean(train_loss))

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss_epoch:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        preds, trues = [], []

        folder_path = os.path.join('./test_results/', setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._build_dec_inp(batch_x, batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred_y = self._select_y_pred(outputs).detach().cpu().numpy()   # (B, pred_len, c_out)
                true_y = self._select_y_true(batch_y).detach().cpu().numpy()  # (B, pred_len, c_out)

                preds.append(pred_y)
                trues.append(true_y)

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    hist_y = inp[0, :, -self.args.c_out:].reshape(-1)
                    gt = np.concatenate([hist_y, true_y[0].reshape(-1)], axis=0)
                    pd_ = np.concatenate([hist_y, pred_y[0].reshape(-1)], axis=0)
                    visual(gt, pd_, os.path.join(folder_path, f'{i}.pdf'))

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

        # save results
        res_path = os.path.join('./results/', setting)
        os.makedirs(res_path, exist_ok=True)
        np.save(os.path.join(res_path, 'pred.npy'), preds)
        np.save(os.path.join(res_path, 'true.npy'), trues)

        # dtw
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_value = float(np.mean(dtw_list))
        else:
            dtw_value = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, dtw:{dtw_value}')
        return