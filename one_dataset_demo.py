
import argparse

from demo_functions import *
from model.distribution2d import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "data_size",
    help="number of data point for training",
    type=float
)

parser.add_argument(
    "batch_size",
    help="display a square of a given number",
    type=int,
    default=4000
)

# args = parser.parse_args()
dis = SPIRALS(flip_var_order=True)
demo(
    dis,
    data_size=3000,
    batch_size=200,
    ttl_iter=8000,
    lr=0.005,
    sapw=0.5,
    learnable_sapw=True,
    stat_size=30,
    use_scheduler=False
)

