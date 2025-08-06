
import os
import time
from argparse import ArgumentParser
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import TQDMProgressBar
import warnings

from arg_setting.parse_setting import Parameter_setting
from datasets.data_selection import DATASETS
from models.model_selection import MODELS
from task.trainer_selection import Trainer

warnings.filterwarnings("ignore", category=UserWarning)


def train_model(setting_dict: dict, setting_file: str = None):
    start = time.time()

    pl.seed_everything(setting_dict["Seed"])
    # Data
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

    # Model
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)

    # Logger
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    # Task
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = Trainer[setting_dict["Setting"]].add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = Trainer[setting_dict["Setting"]](model=model, hparams=task_params, logger=logger)

    if (
        setting_file is not None
        and type(logger.experiment).__name__ != "DummyExperiment"
    ):
        print("Copying setting yaml.")
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(os.path.join(logger.log_dir, "setting.yaml"), "w") as fp:
            yaml.dump(setting_dict, fp)

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**setting_dict["Checkpointer"])

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    ckpt_path = setting_dict["ckpt_path"]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        **trainer_dict,
    )
    if setting_dict["ckpt_path"] in [None, "None"]:
        trainer.fit(task, dm)
    else:
        trainer.fit(task, dm, ckpt_path=ckpt_path)  # Recovery checkpoint

    print(
        f"Best argument {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}"
    )

    end = time.time()

    print(f"Calculation done in {end - start} seconds.")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "setting",
        type=str,
        metavar="path/to/setting.yaml",
        help="yaml with all settings",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="moving_mnist",
        help="The code needs to know the training task based on the datasets",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/ZY/dataset",
        metavar="path/to/datasets",
        help="Path where datasets is located",
    )

    args = parser.parse_args()

    setting_dict = Parameter_setting[args.data_name](args.setting)

    setting_dict["Data"]["base_dir"] = args.data_dir

    train_model(setting_dict, args.setting)