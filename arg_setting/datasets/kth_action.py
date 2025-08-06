import argparse
import copy
import multiprocessing
import os
import re
from typing import Optional, Union

import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class KTHDataset(Dataset):
    """KTH Action Dataset with integrated data loading and preprocessing"""

    def __init__(self, base_dir, mode='train', context_length=10, target_length=20, image_width=128):
        super(KTHDataset, self).__init__()
        self.base_dir = base_dir
        self.mode = mode
        self.context_length = context_length
        self.target_length = target_length
        self.image_width = image_width
        self.seq_len = context_length + target_length

        # 类别定义
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.category = self.category_1 + self.category_2

        # 训练和测试人员ID
        self.train_person = [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
            '13', '14', '15', '16',
        ]
        self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

        # 加载数据和构建索引
        self.frames_paths, self.indices = self.load_data()

    def extract_number(self, file_name):
        match = re.search(r"image_(\d+)", file_name)
        if match:
            return int(match.group(1))
        else:
            return None  # 如果没有匹配到，返回 None
    def load_data(self):
        """加载数据并构建索引"""
        if self.mode == 'train':
            person_id = self.train_person
        elif self.mode == 'test':
            person_id = self.test_person
        else:
            raise ValueError("Mode must be 'train' or 'test'")

        frames_paths = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0

        for c_dir in self.category:
            if c_dir in self.category_1:
                frame_category_flag = 1
            elif c_dir in self.category_2:
                frame_category_flag = 2
            else:
                raise ValueError("Category error")

            c_dir_path = os.path.join(self.base_dir, "kth_action", c_dir)
            p_c_dir_list = os.listdir(c_dir_path)

            for p_c_dir in p_c_dir_list:
                if p_c_dir[6:8] not in person_id:
                    continue
                person_mark += 1

                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort()

                for cur_file in filelist:
                    if not cur_file.startswith('image'):
                        continue
                    frames_paths.append(os.path.join(dir_path, cur_file))
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)

        # 构建索引
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(self.extract_number(os.path.basename(frames_paths[index])))
                start = int(self.extract_number(os.path.basename(frames_paths[index - self.seq_len + 1])))
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.seq_len - 1
                    elif frames_category[index] == 2:
                        index -= 2
            index -= 1

        print(f"{self.mode} dataset: {len(frames_paths)} frames, {len(indices)} sequences")
        return frames_paths, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end = begin + self.seq_len

        data = []
        for idx in range(begin, end):
            frame_path = self.frames_paths[idx]
            frame = Image.open(frame_path)
            frame_np = np.array(frame)  # (H, W, C)
            frame_np = frame_np[:, :, 0]  # 只取单通道（假设是灰度图）

            # 调整大小和归一化
            resized_frame = cv2.resize(frame_np, (self.image_width, self.image_width))
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            data.append(normalized_frame)

        data = np.array(data)
        data = torch.tensor(data).unsqueeze(1).float()  # (seq_len, 1, H, W)

        return data


class KTH_action(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))

    @staticmethod
    def add_data_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--base_dir", type=str, default="models_pytorch/datasets/")
        parser.add_argument("--image_width", type=int, default=128)
        parser.add_argument('--context_length', default=10, type=int, help='Input sequence length')
        parser.add_argument('--target_length', default=20, type=int, help='Output sequence length for train')
        parser.add_argument('--val_length', default=20, type=int, help='Output sequence length for text')
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())

        return parser

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_set = KTHDataset(
                base_dir=self.hparams.base_dir,
                mode='train',
                context_length=self.hparams.context_length,
                target_length=self.hparams.target_length,
                image_width=self.hparams.image_width
            )

            self.val_set = KTHDataset(
                base_dir=self.hparams.base_dir,
                mode='test',
                context_length=self.hparams.context_length,
                target_length=self.hparams.val_length,
                image_width=self.hparams.image_width
            )
            print(f"train dataset: {len(self.train_set)} sequences, val dataset: {len(self.val_set)} sequences,")

        if stage == "test" or stage is None:
            self.test_set = KTHDataset(
                base_dir=self.hparams.base_dir,
                mode='test',
                context_length=self.hparams.context_length,
                target_length=self.hparams.val_length,
                image_width=self.hparams.image_width
            )
            print(f"test dataset: {len(self.test_set)} sequences,")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
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