import argparse
import copy
import os
import random
from typing import Optional, Union
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
try:
    import tensorflow as tf
except ImportError:
    tf = None

from utils.tools import str2bool


class BAIRDataset(Dataset):
    """ BAIR Robot Pushing Action Dataset
            <https://arxiv.org/abs/1710.05268  >`_
        """

    def __init__(self, path, image_size=64, pre_seq_length=2, aft_seq_length=10, use_augment=False,):
        super(BAIRDataset, self).__init__()
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.tot_seq_length = self.pre_seq_length + self.aft_seq_length
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.datas, self.indices = self.load_data(path)

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 3, 64, 64]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x + h, y:y + w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3,))  # horizontal flip
        return imgs

    def load_data(self, path):
        """Loads the dataset.
        Args:
            path: action_path.
        Returns:
            A dataset and indices of the sequence.
        """

        video_fullpaths = []
        indices = []

        tfrecords = os.listdir(path)
        tfrecords.sort()
        num_pictures = 0
        assert tf is not None and 'Please install tensorflow, e.g., pip install tensorflow'

        for tfrecord in tfrecords:
            filepath = os.path.join(path, tfrecord)
            video_fullpaths.append(filepath)
            k = 0
            # 使用tf.data.TFRecordDataset代替tf_record_iterator
            for serialized_example in tf.data.TFRecordDataset(filepath):
                example = tf.train.Example()
                example.ParseFromString(serialized_example.numpy())
                i = 0
                while True:
                    action_name = str(i) + '/action'
                    action_value = np.array(example.features.feature[action_name].float_list.value)
                    if action_value.shape == (0,):  # End of frames/data
                        break
                    i += 1
                num_pictures += i
                for j in range(i - self.tot_seq_length + 1):
                    indices.append((filepath, k, j))
                k += 1
        return video_fullpaths, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]

        input_batch = np.zeros(
            (self.pre_seq_length + self.aft_seq_length, self.image_size, self.image_size, 3)).astype(np.float32)
        begin = batch_ind[-1]
        end = begin + self.tot_seq_length
        k = 0
        # 使用tf.data.TFRecordDataset代替tf_record_iterator
        for serialized_example in tf.data.TFRecordDataset(batch_ind[0]):
            if k == batch_ind[1]:
                example = tf.train.Example()
                example.ParseFromString(serialized_example.numpy())
                break
            k += 1
        for j in range(begin, end):
            aux1_image_name = str(j) + '/image_aux1/encoded'
            aux1_byte_str = example.features.feature[aux1_image_name].bytes_list.value[0]
            aux1_img = Image.frombytes('RGB', (self.image_size, self.image_size), aux1_byte_str)
            aux1_arr = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))

            input_batch[j - begin, :, :, :3] = aux1_arr.reshape(self.image_size, self.image_size, 3) / 255

        input_batch = torch.tensor(input_batch).float().permute(0, 3, 1, 2)
        # data = input_batch[:self.pre_seq_length, ::]
        # labels = input_batch[self.pre_seq_length:self.tot_seq_length, ::]
        # if self.use_augment:
        #     imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
        #     data = imgs[:self.pre_seq_length, ...]
        #     labels = imgs[self.pre_seq_length:self.tot_seq_length, ...]

        return input_batch


class BAIR(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = os.path.join(hparams.base_dir, "bair", "softmotion30_44k")

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
        parser.add_argument("--image_size", type=int, default=64)
        parser.add_argument("--pre_seq_length", type=int, default=2)
        parser.add_argument("--aft_seq_length", type=int, default=10)
        parser.add_argument("--use_augment", type=str2bool, default=False)
        parser.add_argument("--shuffle", type=str2bool, default=True)
        parser.add_argument(
            "--num_workers", type=int, default=0
        )

        return parser


    def setup(self, stage: str =None):

        if stage == "fit" or None:
            path_train = os.path.join(self.base_dir, "train")
            path_test = os.path.join(self.base_dir, "test")
            self.train_set = BAIRDataset(path_train, image_size=self.hparams.image_size,
                                         pre_seq_length=self.hparams.pre_seq_length,
                                         aft_seq_length=self.hparams.aft_seq_length)
            self.val_set = BAIRDataset(path_test, image_size=self.hparams.image_size,
                                       pre_seq_length=self.hparams.pre_seq_length,
                                       aft_seq_length=self.hparams.aft_seq_length)
            f"train dataset: {len(self.train_set)} sequences, val dataset: {len(self.val_set)} sequences,"

        if stage == "test" or None:
            path_test = os.path.join(self.base_dir, "test")
            self.test_set = BAIRDataset(path_test, image_size=self.hparams.image_size,
                                        pre_seq_length=self.hparams.pre_seq_length,
                                        aft_seq_length=self.hparams.aft_seq_length)
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