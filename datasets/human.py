import argparse
import copy
import os
from typing import Optional, Union

import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from utils.tools import str2bool


class HumanDataset(Dataset):
    """ Human 3.6M Dataset
        Modified to automatically filter and focus only on 'Walking' scenarios.
    """

    def __init__(self, data_root, list_path, image_size=256, scene=None,
                 pre_seq_length=4, aft_seq_length=4, step=5, use_augment=False):
        super(HumanDataset, self).__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.step = step
        self.use_augment = use_augment
        self.input_shape = (self.seq_length, self.image_size, self.image_size, 3)

        # Load all lines from the list file
        with open(list_path, 'r') as f:
            all_files = f.readlines()
        if scene is not None:
            # Filter to include only lines that contain 'Walking'
            self.file_list = [line.strip() for line in all_files if scene in line]
        else:
            self.file_list = all_files

        self.mean = None
        self.std = None

    def _augment_seq(self, imgs, h, w):
        """Augmentations for video"""
        ih, iw, _ = imgs[0].shape
        # Random Crop
        length = len(imgs)
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(length):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        # Random Rotation
        if random.randint(-2, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(-2, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif random.randint(-2, 1):
            for i in range(length):
                imgs[i] = cv2.flip(imgs[i], flipCode=1)  # horizontal flip
        # to tensor
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).float()
        return imgs

    def _to_tensor(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = torch.from_numpy(imgs[i].copy()).float()
        return imgs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item_list = self.file_list[idx].split(',')
        begin = int(item_list[1])
        end = begin + self.seq_length * self.step

        raw_input_shape = self.input_shape if not self.use_augment \
            else (self.seq_length, int(self.image_size / 0.975), int(self.image_size / 0.975), 3)
        img_seq = []
        i = 0
        for j in range(begin, end, self.step):
            # e.g., images/S11_Walking.60457274_001621.jpg
            base_str = '0' * (6 - len(str(j))) + str(j) + '.jpg'
            file_name = os.path.join(self.data_root, item_list[0] + base_str)
            image = cv2.imread(file_name)
            if image.shape[0] != raw_input_shape[2]:
                image = cv2.resize(image, (raw_input_shape[1], raw_input_shape[2]), interpolation=cv2.INTER_CUBIC)
            img_seq.append(image)
            i += 1

        # augmentation
        if self.use_augment:
            img_seq = self._augment_seq(img_seq, h=self.image_size, w=self.image_size)
        else:
            img_seq = self._to_tensor(img_seq)

        # transform
        img_seq = torch.stack(img_seq, 0).permute(0, 3, 1, 2) / 255  # min-max to [0, 1]
        data = img_seq[:self.pre_seq_length, ...]
        labels = img_seq[self.aft_seq_length:, ...]

        return torch.cat((data, labels), dim=0)


class Human(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = os.path.join(hparams.base_dir, "human",)


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
        parser.add_argument("--context_length", type=int, default=4)
        parser.add_argument("--target_length", type=int, default=4)
        parser.add_argument("--use_augment", type=str2bool, default=False)
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--scene", type=str, default=None)
        parser.add_argument("--shuffle", type=str2bool, default=True)
        parser.add_argument(
            "--num_workers", type=int, default=0
        )

        return parser


    def setup(self, stage: str =None):

        if stage == "fit" or None:
            self.train_set = HumanDataset(self.base_dir, os.path.join(self.base_dir, 'train.txt'), self.hparams.image_size,
                                     pre_seq_length=self.hparams.context_length, aft_seq_length=self.hparams.target_length,
                                     step=5, use_augment=self.hparams.use_augment, scene=self.hparams.scene)
            self.val_set = HumanDataset(self.base_dir, os.path.join(self.base_dir, 'test.txt'), self.hparams.image_size,
                                     pre_seq_length=self.hparams.context_length, aft_seq_length=self.hparams.target_length,
                                     step=5, use_augment=self.hparams.use_augment, scene=self.hparams.scene)
            print(f"train dataset: {len(self.train_set)} sequences, val dataset: {len(self.val_set)} sequences,")

        if stage == "test" or None:
            self.test_set = HumanDataset(self.base_dir, os.path.join(self.base_dir, 'test.txt'), self.hparams.image_size,
                                        pre_seq_length=self.hparams.context_length,
                                        aft_seq_length=self.hparams.target_length,
                                        step=5, use_augment=self.hparams.use_augment, scene=self.hparams.scene)
            print(f"test dataset: {len(self.test_set)} sequences,")

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


