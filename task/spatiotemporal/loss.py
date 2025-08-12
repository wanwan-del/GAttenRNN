import ast

import torch
from torch import nn


class MSE_loss(nn.Module):
    def __init__(self, args: ast.literal_eval):
        super().__init__()
        self.target_length = args["target_length"]
        self.decouple_loss_term = args.get("decouple_loss_term", None)
        self.decouple_loss_weight=args.get("decouple_loss_weight", None)
        self.pred_mask_value = args.get("pred_mask_value", None)
        self.MSE_criterion = nn.MSELoss(reduction="none")

    def forward(self, preds, batch, aux=None,):

        targ = batch[:, -preds.shape[1]:, ..., ]
        mse = self.MSE_criterion(targ, preds)
        mse = torch.mean(mse, dim=1)
        if self.pred_mask_value:
            pred_mask = (
                (preds != self.pred_mask_value).bool().type_as(preds).max(1)[0]
            )
            mse = (mse * pred_mask).sum() / (pred_mask.sum() + 1e-8)
        else:
            mse = mse.mean()

        logs = {"loss": mse}
        if self.decouple_loss_term:
            extra_loss = aux[self.decouple_loss_term]
            logs["mse"] = mse
            logs[self.decouple_loss_term] = extra_loss
            mse += self.decouple_loss_weight * extra_loss
            logs["loss"] = mse

        return mse, logs


class MaskedL2NDVILoss(nn.Module):
    def __init__(self, args: ast.literal_eval):
        super().__init__()

        self.lc_min = args["lc_min"]
        self.lc_max = args["lc_max"]

        self.target_length = args["target_length"]
        self.ndvi_pred_idx = args["ndvi_pred_idx"]  # index of the NDVI band
        self.ndvi_targ_idx = args["ndvi_targ_idx"]  # index of the NDVI band
        self.pred_mask_value = args.get("pred_mask_value", None)

        self.scale_by_std = args.get("scale_by_std", None)
        self.weight_by_std = args.get("weight_by_std", None)
        if self.scale_by_std:
            print(f"Using Masked L2/Std NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max}).")
        else:
            print(f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max}).")

        self.decouple_loss_term = args.get("decouple_loss_term", None)
        self.decouple_loss_weight = args.get("decouple_loss_weight", None)

    def forward(self, preds, batch, aux,):

        s2_mask = (batch["dynamic_mask"][0][:,-self.target_length:] < 1.0).bool().type_as(preds)
          # b t c h w

        # Landcover mask
        lc = batch["landcover"]
        lc_mask = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(preds)  # b c h w
        ndvi_targ = batch["dynamic"][0][:, -self.target_length:, self.ndvi_targ_idx,...,].unsqueeze(2)  # b t c h w

        ndvi_pred = preds[:, -ndvi_targ.shape[1]:, self.ndvi_pred_idx, ...].unsqueeze(2)  # b t c h w

        sum_squared_error = (((ndvi_targ - ndvi_pred) * s2_mask) ** 2).sum(1)  # b c h w
        mse = sum_squared_error / (s2_mask.sum(1) + 1e-8)  # b c h w

        if self.scale_by_std:
            mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
                s2_mask.sum(1).unsqueeze(1) + 1e-8
            )  # b t c h w
            sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
                1
            )  # b c h w
            mse = sum_squared_error / sum_squared_deviation.clip(
                min=0.01
            )  # mse b c h w
        elif self.weight_by_std:
            mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (
                s2_mask.sum(1).unsqueeze(1) + 1e-8
            )  # b t c h w
            sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(
                1
            )  # b c h w
            mse = sum_squared_error * (
                ((sum_squared_deviation / (s2_mask.sum(1) + 1e-8)) ** 0.5) / 0.1
            ).clip(
                min=0.01, max=100.0
            )  # b c h w

        if self.pred_mask_value:  # what is that?
            pred_mask = (
                (ndvi_pred != self.pred_mask_value).bool().type_as(preds).max(1)[0]
            )
            mse_lc = (mse * lc_mask * pred_mask).sum() / (
                (lc_mask * pred_mask).sum() + 1e-8
            )
        else:
            mse_lc = (mse * lc_mask).sum() / (lc_mask.sum() + 1e-8)

        logs = {"loss": mse_lc}

        if self.decouple_loss_term:
            extra_loss = aux[self.decouple_loss_term]
            logs["mse_lc"] = mse_lc
            logs[self.decouple_loss_term] = extra_loss
            mse_lc += self.decouple_loss_weight * extra_loss
            logs["loss"] = mse_lc

        return mse_lc, logs





