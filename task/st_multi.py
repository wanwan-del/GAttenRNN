import argparse
import ast
import copy
import json
import os

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch import nn

import warnings

from task.metric.s_metric import METRICS
from task.spatiotemporal.loss import MaskedL2NDVILoss
from task.spatiotemporal.shedule import SHEDULERS

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


class SpatioTemporal_multi(pl.LightningModule):
    def __init__(self, model: nn.Module, hparams: argparse.Namespace, logger=None):
        super().__init__()

        self.save_hyperparameters(copy.deepcopy(hparams))
        self.logger_ = logger
        if hparams.pred_dir is None:
            self.pred_dir = (
                Path(self.logger_.log_dir) / "predictions"
                if self.logger_ is not None
                else Path.cwd() / "experiments" / "predictions"
            )  # logger: hyperparameter of LightningModule for the Trainer
        else:
            self.pred_dir = Path(self.hparams.pred_dir)


        self.model = model
        self.loss = MaskedL2NDVILoss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length

        self.lc_min = hparams.lc_min
        self.lc_max = hparams.lc_max

        self.n_stochastic_preds = hparams.n_stochastic_preds

        self.shedulers = []
        for shedule in self.hparams.shedulers:
            self.shedulers.append(
                (shedule["call_name"], SHEDULERS[shedule["name"]](**shedule["args"]))
            )

        self.metric = METRICS[self.hparams.metric](**self.hparams.metric_kwargs)

    @staticmethod
    def add_task_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):  # Optional[X] is equivalent to Union[X, None].
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--pred_dir", type=str, default=None)
        parser.add_argument("--setting", type=str, default="greenearthnet")
        parser.add_argument("--loss", type=ast.literal_eval,
            default='{"name": "masked", "args": {"distance_type": "L1"}}',
        )
        # Metric used for the test set and the validation set.
        parser.add_argument("--metric", type=str, default="RMSE")
        parser.add_argument("--metric_kwargs", type=ast.literal_eval, default="{}")

        # Context and target length for temporal model. A temporal model use a context period to learn the temporal dependencies and predict the target period.
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)

        # Landcover bounds. Used as mask on the non-vegetation pixel.
        parser.add_argument("--lc_min", type=int, default=10)
        parser.add_argument("--lc_max", type=int, default=40)

        # Number of stochastic prediction for statistical models.
        parser.add_argument("--n_stochastic_preds", type=int, default=1)

        # Number of batches to be displayed in the logger
        parser.add_argument("--n_log_batches", type=int, default=2)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        # optimizer: Function that adjusts the attributes of the neural network, such as weights and learning rates.
        parser.add_argument(
            "--optimization",
            type=ast.literal_eval,
            default='{"optimizer": [{"name": "AdamW", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], "lr_shedule": [{"name": "multistep", "args": {"milestones": [25, 40], "gamma": 0.1} }]}',
        )
        # Sheduler: methods to adjust the learning rate based on the number of epochs
        parser.add_argument("--shedulers", type=ast.literal_eval, default="[]")
        return parser

    def forward(self, data, kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self.model(
            data, **kwargs,
        )

    def configure_optimizers(self):
        # 创建优化器对象列表
        optimizers = [
                getattr(torch.optim, o["name"])(self.parameters(), **o["args"])
                for o in self.hparams.optimization["optimizer"]
            ]

        schedulers = [
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)
              for optimizer in optimizers
          ]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):

        kwargs = {}
        for shedule_name, shedule in self.shedulers:
            kwargs[shedule_name] = shedule(self.global_step)

        # Predictions generation
        preds, aux = self(batch, kwargs=kwargs)
        loss, logs = self.loss(preds, batch, aux, )
        # Logs
        for shedule_name in kwargs:
            if len(kwargs[shedule_name]) > 1:
                for i, shed_val in enumerate(kwargs[shedule_name]):
                    logs[f"{shedule_name}_i"] = shed_val
            else:
                logs[shedule_name] = kwargs[shedule_name]
        logs["batch_size"] = torch.tensor(
            self.hparams.train_batch_size, dtype=torch.float32
        )
        # Metric logging method
        self.log_dict(logs)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform one evaluation epoch over the validation set. Operates on a single batch of models_pytorch from the validation set. In this step you d might generate examples or calculate anything of interest like accuracy.
        Return  a List of dictionaries with metrics logged during the validation phase
        """

        data = copy.deepcopy(batch)

        loss_logs = []  # list of loss values

        # Predictions of the model
        preds, aux = self(data)

        # Loss computation
        mse, logs = self.loss(preds, batch, aux)

        if np.isfinite(mse.cpu().detach().numpy()):  # 检查loss是否为有限值
            loss_logs.append(logs)

        # Update of the metric
        self.metric.update(preds, batch)

        mean_logs = {
            log_name: torch.tensor(
                [log[log_name] for log in loss_logs],
                dtype=torch.float32,
                device=self.device,
            ).mean()
            for log_name in loss_logs[0]
        }

        # loss_val
        self.log_dict(
            {log_name + "_val": mean_logs[log_name] for log_name in mean_logs},
            sync_dist=True,
        )

    def on_validation_epoch_end(self,):
        current_scores = self.metric.compute()
        self.log_dict(current_scores, sync_dist=True)
        self.metric.reset()  # lagacy? To remove, shoudl me managed by the logger?
        if (
            self.logger is not None
            and type(self.logger.experiment).__name__ != "DummyExperiment"
            and self.trainer.is_global_zero
        ):
            current_scores["epoch"] = self.current_epoch
            current_scores = {
                k: (
                    str(v.detach().cpu().item())
                    if isinstance(v, torch.Tensor)
                    else str(v)
                )
                for k, v in current_scores.items()
            }
            outpath = Path(self.logger.log_dir) / "validation_scores.json"
            if outpath.is_file():
                with open(outpath, "r") as fp:
                    past_scores = json.load(fp)
                scores = past_scores + [current_scores]
            else:
                scores = [current_scores]

            with open(outpath, "w") as fp:
                json.dump(scores, fp)

    def test_step(self, batch, batch_idx):
        """
        Operates on a single batch of data from the test set.
        In this step you generate examples or calculate anything of interest such as accuracy.
        """

        for i in range(self.n_stochastic_preds):
            preds, aux = self(
                batch,
            )
            if self.hparams.setting == "greenearthnet":
                for j in range(preds.shape[0]):
                    # Targets
                    targ_path = Path(batch["filepath"][j])
                    targ_cube = xr.open_dataset(targ_path)

                    lat = targ_cube.lat
                    lon = targ_cube.lon

                    ndvi_preds = preds[j, :, 0, ...].detach().cpu().numpy()
                    pred_cube = xr.Dataset(
                        {
                            "ndvi_pred": xr.DataArray(
                                data=ndvi_preds,
                                coords={
                                    "time": targ_cube.time.isel(
                                        time=slice(4, None, 5)
                                    ).isel(
                                        time=slice(
                                            self.context_length,
                                            self.context_length + self.target_length,
                                        )
                                    ),
                                    "lat": lat,
                                    "lon": lon,
                                },
                                dims=["time", "lat", "lon"],
                            )
                        }
                    )

                    pred_dir = self.pred_dir
                    pred_path = pred_dir / targ_path.parent.stem / targ_path.name
                    pred_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录

                    if pred_path.is_file():
                        pred_path.unlink()

                    if not pred_path.is_file():
                        pred_cube.to_netcdf(
                            pred_path, encoding={"ndvi_pred": {"dtype": "float32"}}
                        )






