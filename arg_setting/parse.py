"""Utilities for parsing setting yamls

parse setting implements copying of global attributes to the particular subcategory.

"""
import os.path
from pathlib import Path
import yaml
import warnings

from datasets.data_selection import METRIC_CHECKPOINT_INFO


def Spatio_temporal_setting_s(setting_file, track=None):
    setting_file = Path(setting_file)
    # example: setting: moving_minist, architecture: win_lstm, feature: win_lstm6M, config: seed=24.yaml
    setting, architecture, feature, config = setting_file.parts[-4:]

    with open(setting_file, 'r') as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    # check if the information of the setting_file path correspond with the yaml file

    if setting_dict["Setting"] != setting:
        warnings.warn(
            f"Ambivalent definition of setting, found {setting_dict['Setting']} in yaml but {setting} in path. Using {setting_dict['Setting']}.")
        setting = setting_dict["Setting"]
        setting_dict["Task"]["data_name"] = setting
    else:
        setting_dict["Task"]["data_name"] = setting
    setting_dict["Logger"] = {
        "save_dir": str(Path("experiments/") / setting) if setting is not None else "experiments/",
        "name": setting_dict["Architecture"],
        "version": feature
    }

    setting_dict["Seed"] = setting_dict["Seed"] if "Seed" in setting_dict else 42

    setting_dict["Trainer"]["precision"] = setting_dict["Trainer"]["precision"] if "precision" in setting_dict[
        "Trainer"] else 32  # binary floating-point computer number format

    setting_dict["Checkpointer"] = {**setting_dict["Checkpointer"], **METRIC_CHECKPOINT_INFO[
        setting_dict["Setting"]]} if "Checkpointer" in setting_dict else METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]

    if setting_dict["ckpt_path"] == "train":
        setting_dict["ckpt_path"] = os.path.join(setting_dict["Logger"]["save_dir"],
                                                 setting_dict["Logger"]["name"],
                                                 setting_dict["Logger"]["version"],
                                                 "checkpoints",
                                                 "last.ckpt", )
    elif setting_dict["ckpt_path"] == "test":
        setting_dict["ckpt_path"] = os.path.join(setting_dict["Logger"]["save_dir"],
                                                 setting_dict["Logger"]["name"],
                                                 setting_dict["Logger"]["version"],
                                                 "checkpoints",
                                                 "best.ckpt", )
    else:
        setting_dict["ckpt_path"] = None

    bs = setting_dict["Data"]["train_batch_size"]
    gpus = setting_dict["Trainer"]["gpus"] if "gpus" in setting_dict["Trainer"] else setting_dict["Trainer"]["devices"]
    ddp = (setting_dict["Trainer"]["strategy"] == "ddp") if "strategy" in setting_dict["Trainer"] else 1

    optimizers = setting_dict["Task"]["optimization"]["optimizer"]
    for optimizer in optimizers:
        if "lr_per_sample" in optimizer:
            lr_per_sample = optimizer["lr_per_sample"]
            if isinstance(gpus, list):
                lr = bs * (len(gpus) * ddp + (1 - ddp)) * lr_per_sample
            else:
                lr = bs * (gpus * ddp + (1 - ddp)) * lr_per_sample
            print('learning rate', lr)
            optimizer["args"]["lr"] = lr

    if track is not None:
        setting_dict["Data"]["test_track"] = track

        if "pred_dir" not in setting_dict["Task"]:
            setting_dict["Task"]["pred_dir"] = Path(setting_dict["Logger"]["save_dir"]) / setting_dict["Logger"][
                "name"] / setting_dict["Logger"]["version"] / "preds" / track


    # Cpy information for others modules
    setting_dict["Model"]["context_length"] = setting_dict["Task"]["context_length"]
    setting_dict["Model"]["target_length"] = setting_dict["Task"]["target_length"]
    if "target" in setting_dict["Data"]:
        setting_dict["Model"]["target"] = setting_dict["Data"]["target"]

    setting_dict["Task"]["loss"]["target_length"] = setting_dict["Task"]["target_length"]

    setting_dict["Task"]["train_batch_size"] = setting_dict["Data"]["train_batch_size"]
    setting_dict["Task"]["val_batch_size"] = setting_dict["Data"]["val_batch_size"]
    setting_dict["Task"]["test_batch_size"] = setting_dict["Data"]["test_batch_size"]

    if "metric_kwargs" not in setting_dict["Task"]:
        setting_dict["Task"]["metric_kwargs"] = {}
    setting_dict["Task"]["metric_kwargs"]["target_length"] = setting_dict["Task"]["target_length"]

    if "model_shedules" in setting_dict["Task"]["metric_kwargs"]:
        setting_dict["Task"]["metric_kwargs"]["shedulers"] = setting_dict["Task"]["metric_kwargs"]["model_shedules"]

    return setting_dict

def Spatio_temporal_setting_m(setting_file, track=None):
    setting_file = Path(setting_file)
    # example: setting: moving_minist, architecture: win_lstm, feature: win_lstm6M, config: seed=24.yaml
    setting, architecture, feature, config = setting_file.parts[-4:]

    with open(setting_file, 'r') as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    # check if the information of the setting_file path correspond with the yaml file

    if setting_dict["Setting"] != setting:
        warnings.warn(
            f"Ambivalent definition of setting, found {setting_dict['Setting']} in yaml but {setting} in path. Using {setting_dict['Setting']}.")
        setting = setting_dict["Setting"]
        setting_dict["Task"]["dataname"] = setting
    setting_dict["Logger"] = {
        "save_dir": str(Path("experiments/") / setting) if setting is not None else "experiments/",
        "name": setting_dict["Architecture"],
        "version": feature
    }

    setting_dict["Seed"] = setting_dict["Seed"] if "Seed" in setting_dict else 42

    setting_dict["Trainer"]["precision"] = setting_dict["Trainer"]["precision"] if "precision" in setting_dict[
        "Trainer"] else 32  # binary floating-point computer number format
    setting_dict["Data"]["fp16"] = (setting_dict["Trainer"]["precision"] == 16)  # binary floating-point computer number format
    setting_dict["Checkpointer"] = {**setting_dict["Checkpointer"], **METRIC_CHECKPOINT_INFO[
        setting_dict["Setting"]]} if "Checkpointer" in setting_dict else METRIC_CHECKPOINT_INFO[setting_dict["Setting"]]

    if setting_dict["ckpt_path"] == "train":
        setting_dict["ckpt_path"] = os.path.join(setting_dict["Logger"]["save_dir"],
                                                 setting_dict["Logger"]["name"],
                                                 setting_dict["Logger"]["version"],
                                                 "checkpoints",
                                                 "last.ckpt", )
    elif setting_dict["ckpt_path"] == "test":
        setting_dict["ckpt_path"] = os.path.join(setting_dict["Logger"]["save_dir"],
                                                 setting_dict["Logger"]["name"],
                                                 setting_dict["Logger"]["version"],
                                                 "checkpoints",
                                                 "best.ckpt", )
    else:
        setting_dict["ckpt_path"] = None

    bs = setting_dict["Data"]["train_batch_size"]
    gpus = setting_dict["Trainer"]["gpus"] if "gpus" in setting_dict["Trainer"] else setting_dict["Trainer"]["devices"]
    ddp = (setting_dict["Trainer"]["strategy"] == "ddp") if "strategy" in setting_dict["Trainer"] else 1

    optimizers = setting_dict["Task"]["optimization"]["optimizer"]
    for optimizer in optimizers:
        if "lr_per_sample" in optimizer:
            lr_per_sample = optimizer["lr_per_sample"]
            if isinstance(gpus, list):
                lr = bs * (len(gpus) * ddp + (1 - ddp)) * lr_per_sample
            else:
                lr = bs * (gpus * ddp + (1 - ddp)) * lr_per_sample
            print('learning rate', lr)
            optimizer["args"]["lr"] = lr

    if track is not None:
        setting_dict["Data"]["test_track"] = track

        if "pred_dir" not in setting_dict["Task"]:
            setting_dict["Task"]["pred_dir"] = Path(setting_dict["Logger"]["save_dir"]) / setting_dict["Logger"][
                "name"] / setting_dict["Logger"]["version"] / "preds" / track

    # Cpy information for others modules
    setting_dict["Model"]["context_length"] = setting_dict["Task"]["context_length"]
    setting_dict["Model"]["target_length"] = setting_dict["Task"]["target_length"]
    if "target" in setting_dict["Data"]:
        setting_dict["Model"]["target"] = setting_dict["Data"]["target"]

    setting_dict["Task"]["loss"]["context_length"] = setting_dict["Task"]["context_length"]
    setting_dict["Task"]["loss"]["target_length"] = setting_dict["Task"]["target_length"]

    setting_dict["Task"]["train_batch_size"] = setting_dict["Data"]["train_batch_size"]
    setting_dict["Task"]["val_batch_size"] = setting_dict["Data"]["val_batch_size"]
    setting_dict["Task"]["test_batch_size"] = setting_dict["Data"]["test_batch_size"]

    setting_dict["Task"]["lc_min"] = setting_dict["Task"]["loss"]["lc_min"]
    setting_dict["Task"]["lc_max"] = setting_dict["Task"]["loss"]["lc_max"]

    setting_dict["Task"]["loss"]["setting"] = setting_dict["Setting"]

    if "metric_kwargs" not in setting_dict["Task"]:
        setting_dict["Task"]["metric_kwargs"] = {}
    setting_dict["Task"]["metric_kwargs"]["context_length"] = setting_dict["Task"]["context_length"]
    setting_dict["Task"]["metric_kwargs"]["target_length"] = setting_dict["Task"]["target_length"]

    setting_dict["Task"]["metric_kwargs"]["lc_min"] = setting_dict["Task"]["lc_min"]
    setting_dict["Task"]["metric_kwargs"]["lc_max"] = setting_dict["Task"]["lc_max"]

    if "model_shedules" in setting_dict["Task"]["metric_kwargs"]:
        setting_dict["Task"]["metric_kwargs"]["shedulers"] = setting_dict["Task"]["metric_kwargs"]["model_shedules"]

    return setting_dict