
import argparse

from demo_functions import *
from model.distribution2d import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_size",
    help="number of data points in original data set",
    type=int,
    default=3000
)

parser.add_argument(
    "--batch_size",
    help="batch size for training",
    type=int,
    default=200
)

parser.add_argument(
    "--ttl_iter",
    help="total backprop update for training",
    type=int,
    default=8000
)

parser.add_argument(
    "--lr",
    help="learning rate",
    type=int,
    default=0.005
)

parser.add_argument(
    "--learnable_sapw",
    help="specify if sapw is learnable",
    type=bool,
    default=False
)

parser.add_argument(
    "--stat_size",
    help="number of iteration for showing one training stats",
    type=int,
    default=30
)

parser.add_argument(
    "--use_scheduler",
    help="reduce learning rate if loss does not decrease",
    type=bool,
    default=True
)

parser.add_argument(
    "--sapw",
    help="gate weight from 0(flow estimator) to 1(pure gaussian like estimator)",
    type=float,
    default=0.5
)

if __name__ == "__main__":
    args = parser.parse_args()
    dis = GAUSSIANS(flip_var_order=True)
    demo(
        dis,
        data_size=args.data_size,
        batch_size=args.batch_size,
        ttl_iter=args.ttl_iter,
        lr=args.lr,
        sapw=args.sapw,
        learnable_sapw=args.learnable_sapw,
        stat_size=args.stat_size,
        use_scheduler=args.use_scheduler
    )

