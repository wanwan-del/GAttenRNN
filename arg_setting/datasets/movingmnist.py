import argparse
import copy
import gzip
import multiprocessing
import os
import random
from typing import Optional, Union
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


class Moving_MNIST_train(Dataset):
    def __init__(self, args, train_data_dir, split):
        super().__init__()

        with gzip.open(train_data_dir, 'rb') as f:
            self.datas = np.frombuffer(f.read(), np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, *args.image_size)
        self.split = split
        if split == 'train':
            self.datas = self.datas[args.train_samples[0]: args.train_samples[1]]
        else:
            self.datas = self.datas[args.valid_samples[0]: args.valid_samples[1]]

        self.image_size = args.image_size
        self.input_size = args.input_size
        self.step_length = args.step_length
        self.num_objects = args.num_objects

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output
        print('Loaded {} {} samples'.format(self.__len__(), split))

    def _get_random_trajectory(self, seq_length):

        assert self.input_size[0] == self.input_size[1]
        assert self.image_size[0] == self.image_size[1]

        canvas_size = self.input_size[0] - self.image_size[0]

        x = random.random()
        y = random.random()

        theta = random.random() * 2 * np.pi

        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        for i in range(seq_length):

            y += v_y * self.step_length
            x += v_x * self.step_length

            if x <= 0.: x = 0.; v_x = -v_x;
            if x >= 1.: x = 1.; v_x = -v_x
            if y <= 0.: y = 0.; v_y = -v_y;
            if y >= 1.: y = 1.; v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2,):

        data = np.zeros((self.num_frames_total, *self.input_size), dtype=np.float32)

        for n in range(num_digits):

            start_y, start_x = self._get_random_trajectory(self.num_frames_total)
            ind = np.random.randint(0, self.__len__())
            digit_image = self.datas[ind]

            for i in range(self.num_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size[0]
                right = left + self.image_size[1]
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]

        return data

    def __getitem__(self, item):

        num_digits = random.choice(self.num_objects)
        images = torch.from_numpy(self._generate_moving_mnist(num_digits)).permute(0, 3, 1, 2).contiguous()

        return images / 255.

    def __len__(self):
        return self.datas.shape[0]



class Moving_MNIST_test(Dataset):
    def __init__(self, args, test_data_dir):
        super().__init__()

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        self.dataset = np.load(test_data_dir)
        self.dataset = self.dataset[..., np.newaxis]

        print('Loaded {} {} samples'.format(self.__len__(), 'test'))

    def __getitem__(self, index):
        images =  torch.from_numpy(self.dataset[:, index, ...]).permute(0, 3, 1, 2).contiguous()

        return images / 255.

    def __len__(self):
        return len(self.dataset[1])


class MovingMNIST(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = os.path.join(hparams.base_dir, "moving_mnist")


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

        parser.add_argument('--num_frames_input', default=10, type=int, help='Input sequence length')
        parser.add_argument('--num_frames_output', default=10, type=int, help='Output sequence length')
        parser.add_argument('--image_size', default=(28, 28), type=int, help='Original resolution')
        parser.add_argument('--input_size', default=(64, 64), help='Input resolution')
        parser.add_argument('--step_length', default=0.1, type=float)
        parser.add_argument('--num_objects', default=[2], type=int)
        parser.add_argument('--train_samples', default=[0, 10000], type=int,
                            help='Number of samples in training set')
        parser.add_argument('--valid_samples', default=[10000, 13000], type=int,
                            help='Number of samples in validation set')
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument(
            "--num_workers", type=int, default=multiprocessing.cpu_count()
        )

        return parser


    def setup(self, stage: str =None):
        if stage == "fit" or None:
            self.minist_train = Moving_MNIST_train(
                self.hparams,
                os.path.join(self.base_dir, "train-images-idx3-ubyte.gz"),
                split="train",
            )
            self.minist_val = Moving_MNIST_train(
                self.hparams,
                os.path.join(self.base_dir, "train-images-idx3-ubyte.gz"),
                split="val",
            )
        if stage == "test" or None:
            self.minist_test = Moving_MNIST_test(
                self.hparams,
                os.path.join(self.base_dir, "mnist_test_seq.npy"),
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.minist_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True, pin_memory=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.minist_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False, pin_memory=True, drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.minist_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

