import argparse
import ast
import copy
import json
import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import warnings

from task.spatiotemporal.loss import MSE_loss
from task.metric.s_metric import METRICS
from task.spatiotemporal.shedule import SHEDULERS
from task.spatiotemporal.visualization import Vision
from utils.tools import str2bool

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


class SpatioTemporal_variable(pl.LightningModule):
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
        self.loss = MSE_loss(hparams.loss)

        self.n_stochastic_preds = hparams.n_stochastic_preds

        self.shedulers = []
        for shedule in self.hparams.shedulers:
            self.shedulers.append(
                (shedule["call_name"], SHEDULERS[shedule["name"]](**shedule["args"]))
            )

        self.metric = METRICS[self.hparams.metric](**self.hparams.metric_kwargs)

        self.vision = Vision(save_dir=self.pred_dir, context_length=hparams.context_length,
                         data_name=self.hparams.data_name)
    @staticmethod
    def add_task_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):  # Optional[X] is equivalent to Union[X, None].
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        # Path of the directory to save the prediction
        parser.add_argument("--pred_dir", type=str, default=None)

        # Name of the dataset, involves major differences in the variables available and the tasks to be performed.
        parser.add_argument("--data_name", type=str, default="moving_mnist")

        # Dictionnary of the loss name and the distance norm used.
        parser.add_argument("--loss",
            type=ast.literal_eval,
            default='{"target_length": 10,}',
        )
        # Metric used for the test set and the validation set.
        parser.add_argument("--metric", type=str, default="MSE")
        parser.add_argument("--metric_kwargs", type=ast.literal_eval, default="{}")

        # Context and target length for temporal model. A temporal model use a context period to learn the temporal dependencies and predict the target period.
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=10)
        parser.add_argument("--val_length", type=int, default=20)

        # Number of stochastic prediction for statistical models.
        parser.add_argument("--n_stochastic_preds", type=int, default=1)
        parser.add_argument("--train_reversed", type=str2bool, default=False)

        # Number of batches to be displayed in the logger
        parser.add_argument("--n_log_batches", type=int, default=2)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        # optimizer: Function that adjusts the attributes of the neural network, such as weights and learning rates.
        parser.add_argument(
            "--optimization",
            type=ast.literal_eval,
            default='{"optimizer": [{"name": "AdamW", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], "lr_shedule": [{"name": "multistep", "args": {"milestones": [2000], "gamma": 0.1} }]}',
        )
        # Sheduler: methods to adjust the learning rate based on the number of epochs
        parser.add_argument("--shedulers", type=ast.literal_eval, default="[]")

        return parser

    def forward(self, data, kwargs=None):

        if kwargs is None:
            kwargs = {}
        return self.model(
            data,
            **kwargs,
        )

    """define and load optimizers and shedulers"""
    def configure_optimizers(self):
        # 创建优化器对象列表
        optimizers = []
        for o in self.hparams.optimization["optimizer"]:
            optimizer_class = getattr(torch.optim, o["name"])
            optimizer = optimizer_class(self.parameters(), **o["args"])
            optimizers.append(optimizer)

        # 创建学习率调度器列表
        schedulers = []
        for optimizer, sched in zip(optimizers, self.hparams.optimization["lr_shedule"]):
            scheduler_class = getattr(torch.optim.lr_scheduler, sched["name"])
            scheduler = scheduler_class(optimizer, **sched["args"])
            schedulers.append(scheduler)
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        """compute and return the training loss and some additional metrics for e.g. the progress bar or logger"""

        # Learning rate scheduling should be applied after optimizer’s update
        kwargs = {}
        for shedule_name, shedule in self.shedulers:
            kwargs[shedule_name] = shedule(self.global_step)

        # Predictions generation
        preds, aux = self(batch, kwargs=kwargs)
        loss, logs = self.loss(preds, batch, aux)
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
        label = copy.deepcopy(batch)
        preds = label[:, :self.hparams.context_length, :]
        input = label[:, :self.hparams.context_length, :]
        while preds.shape[1] < label.shape[1]:
            output, _ = self(input)
            input = output[:, -self.hparams.target_length:, :]
            preds = torch.cat((preds, input), dim=1)

        # Update of the metric
        self.metric.update(preds, batch)

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
            label = copy.deepcopy(batch)
            preds = label[:, :self.hparams.context_length, :]
            input = label[:, :self.hparams.context_length, :]
            while preds.shape[1] < label.shape[1]:
                output, _ = self(input)
                input = output[:, -self.hparams.target_length:, :]
                preds = torch.cat((preds, input), dim=1)

            if batch_idx < 1:
                self.vision(preds=preds, batch=batch)

        # Update of the metric
            self.metric.update(preds, batch, test=True)
        # if batch_idx < 10:
        #     self.vision(preds=preds, batch=batch, global_step=batch_idx, epoch=self.current_epoch)

    def on_test_epoch_end(self,):

        overall_scores, frame_scores = self.metric.compute(validation=False)
        self.log_dict(overall_scores, sync_dist=True)
        # 将字典转换为DataFrame
        df = pd.DataFrame(list(overall_scores.items()), columns=['Metric', 'Score'])
        # 指定保存路径
        overall_scores_save_path = os.path.join(self.pred_dir, 'overall_scores.csv')
        # 保存DataFrame为CSV文件
        df.to_csv(overall_scores_save_path, index=False)

        frame_scores_df = pd.DataFrame({
            key: value.cpu().numpy()  # 将每个张量转换为 numpy 数组
            for key, value in frame_scores.items()
        })
        frame_scores_save_path = os.path.join(self.pred_dir, 'frame_scores.csv')
        frame_scores_df.to_csv(frame_scores_save_path, index=False)