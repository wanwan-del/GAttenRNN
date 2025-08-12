from task.st_multi import SpatioTemporal_multi
from task.st_single import SpatioTemporal_single
from task.st_variable import SpatioTemporal_variable

Trainer = {
    "moving_mnist": SpatioTemporal_single,
    "greenearthnet": SpatioTemporal_multi,
    "kth": SpatioTemporal_variable,
    "taxibj": SpatioTemporal_single,
    "human": SpatioTemporal_single,
}