import argparse
import copy
import os
import random
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from utils.tools import str2bool


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, X, Y, use_augment=False):
        super(TaxibjDataset, self).__init__()
        self.X = (X+1) / 2  # channel is 2
        self.Y = (Y+1) / 2
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return torch.cat((data, labels), dim=0)


class Taxibj(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = os.path.join(hparams.base_dir, "taxibj")

    @staticmethod
    def add_data_specific_args(
            parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--base_dir", type=str, default="models_pytorch/datasets/")

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--use_augment", type=str2bool, default=False)
        parser.add_argument("--shuffle", type=str2bool, default=True)
        parser.add_argument(
            "--num_workers", type=int, default=0
        )

        return parser


    def setup(self, stage: str =None):

        dataset = np.load(os.path.join(self.base_dir, 'dataset.npz'))
        X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']

        if stage == "fit" or None:
            self.train_set = TaxibjDataset(X=X_train, Y=Y_train, use_augment=self.hparams.use_augment)
            self.val_set = TaxibjDataset(X=X_test, Y=Y_test, use_augment=False)
            f"train dataset: {len(self.train_set)} sequences, val dataset: {len(self.val_set)} sequences,"
        if stage == "test" or None:
            self.test_set = TaxibjDataset(X=X_test, Y=Y_test, use_augment=False)
            f"test dataset: {len(self.test_set)} sequences,"

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
            pin_memory=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )



