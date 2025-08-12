import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class Vision(nn.Module):
    def __init__(self, save_dir, context_length, data_name):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.context_length = context_length
        self.data_name = data_name
    def forward(self, preds, batch):
        preds = torch.clip(preds, min=0, max=1)
        print(f"Before indexing: preds shape = {preds.shape}, batch shape = {batch.shape}")
        preds = preds[3]
        batch = batch[3]
        print(f"After indexing: preds shape = {preds.shape}, batch shape = {batch.shape}")
        targ = batch.detach().cpu()
        pred = preds.detach().cpu()

        targ_save_dir = os.path.join(self.save_dir, f"target_sequence")
        pred_save_dir = os.path.join(self.save_dir, f"pred_sequence")

        os.makedirs(targ_save_dir, exist_ok=True)
        os.makedirs(pred_save_dir, exist_ok=True)

        if self.data_name in ["moving_mnist", "kth"]:
            for t in range(targ.shape[0]):
                # 获取当前时间点的图像
                img = targ[t]
                # 判断通道数，确定保存图像的方式
                img = (img[0] * 255).byte()  # 只取第一个通道
                # 保存为灰度图像
                plt.imsave(f"{targ_save_dir}/target_{t}.png", img, cmap='gray')

            for t in range(pred.shape[0]):
                # 获取当前时间点的图像
                img = pred[t]

                img = (img[0] * 255).byte()  # 只取第一个通道
                # 保存为灰度图像
                plt.imsave(f"{pred_save_dir}/pred_{t}.png", img, cmap='gray')

        elif self.data_name == "human":
            error_save_dir = os.path.join(self.save_dir, f"error")
            os.makedirs(error_save_dir, exist_ok=True)
            error = torch.abs(targ - pred)
            for t in range(targ.shape[0]):
                # 获取当前时间点的图像
                img = targ[t].numpy()
                # 调整图像数据范围和数据类型
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 保存为 RGB 图像
                plt.imsave(f"{targ_save_dir}/target_{t}.jpg", img)


            for t in range(pred.shape[0]):
                # 获取当前时间点的图像
                img = pred[t].numpy()
                # 调整图像数据范围和数据类型
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 保存为 RGB 图像
                plt.imsave(f"{pred_save_dir}/target_{t}.jpg", img)

            for t in range(error.shape[0]):
                # 获取当前时间点的图像
                img = error[t].numpy()
                # 调整图像数据范围和数据类型
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 保存为 RGB 图像
                plt.imsave(f"{error_save_dir}/error_{t}.jpg", img)


        elif self.data_name == "taxibj":

            error_save_dir = os.path.join(self.save_dir, f"error")

            os.makedirs(error_save_dir, exist_ok=True)

            cmap = get_mpl_colormap('viridis')

            # 将张量转换为 NumPy 数组

            targ_numpy = targ.numpy()

            pred_numpy = pred.numpy()

            # 现在可以使用 NumPy 的 astype 方法

            error = np.abs(targ_numpy[:1, :].astype(np.float32) - pred_numpy.astype(np.float32))

            for t in range(targ_numpy.shape[0]):
                # 获取当前时间点的目标图像

                targ_img = targ_numpy[t, 0, :, :]  # 假设通道数为1

                # 将数据缩放到 [0, 1] 范围，并转换为 8 位整数

                targ_img = (targ_img - targ_img.min()) / (targ_img.max() - targ_img.min())

                targ_img = np.uint8(255 * targ_img)

                # 应用颜色映射表

                targ_img = cv2.applyColorMap(targ_img, cmap)

                # 保存目标图像

                cv2.imwrite(f"{targ_save_dir}/target_{t}.png", targ_img)

            for t in range(pred_numpy.shape[0]):
                # 获取当前时间点的预测图像

                pred_img = pred_numpy[t, 0, :, :]

                pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())

                pred_img = np.uint8(255 * pred_img)

                pred_img = cv2.applyColorMap(pred_img, cmap)

                # 保存预测图像

                cv2.imwrite(f"{pred_save_dir}/pred_{t}.png", pred_img)

            for t in range(error.shape[0]):
                error_img = error[t, 0, :, :]

                error_img = np.uint8(255 * error_img / error_img.max())

                error_img = cv2.applyColorMap(error_img, cmap)

                # 保存误差图像

                cv2.imwrite(f"{error_save_dir}/error_{t}.png", error_img)


def get_mpl_colormap(cmap_name):
    """mapping matplotlib cmap to cv2"""
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 1, 3)







