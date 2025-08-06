from datasets.bair import BAIR
from datasets.greenearthnet import GreenEarthNet
from datasets.human import Human
from datasets.kth import KTH
from datasets.kth_action import KTH_action
from datasets.movingmnist import MovingMNIST
from datasets.taxibj import Taxibj

DATASETS = {
    "greenearthnet": GreenEarthNet,
    "moving_mnist": MovingMNIST,
    "kth": KTH,
    "kth_action":KTH_action,
    "taxibj": Taxibj,
    "human": Human,
    "bair": BAIR,

}


METRIC_CHECKPOINT_INFO = {
    "greenearthnet": {
        "monitor": "RMSE_Veg",
        "filename": "{epoch:02d}-{RMSE_Veg:.4f}",
        "mode": "min",
    },

    "moving_mnist": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },

    "kth": {
        "monitor": "PSNR",
        "filename": "{epoch:02d}-{PSNR:.4f}",
        "mode": "max",
    },
    "kth_action": {
        "monitor": "PSNR",
        "filename": "{epoch:02d}-{PSNR:.4f}",
        "mode": "max",
    },
    "taxibj": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },
    "human": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },
    "bair": {
        "monitor": "PSNR",
        "filename": "{epoch:02d}-{PSNR:.4f}",
        "mode": "max",
    },
}