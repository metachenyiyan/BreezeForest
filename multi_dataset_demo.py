
import argparse
import matplotlib.pyplot as plt
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
    default=False
)

if __name__ == "__main__":
    args = parser.parse_args()

    data_set_list = [
        MOONS(),
        BUTTERFLY(),
        BLOBS(),
        CIRCLE(),
        CHECKERBOARD(),
        SPIRALS(),
        GAUSSIANS()
    ]
    sap_weights = [0, 0.01, 0.1, 0.5,  0.9, 0.99, 1]
    # sap_weights = [0.5, 0.7, 0.9, 0.95, 0.99]
    cols = ["Original"] + ['Sap Weight:{}'.format(sapw) for sapw in sap_weights]
    figures = []
    fig = plt.figure()

    for i in range(len(data_set_list)):
        for j in range(len(cols)):
            print("demo: " + data_set_list[i].__class__.__name__)
            figures.append(fig.add_subplot(len(data_set_list), len(cols), i * len(cols) + j + 1))
            col_title = None
            row_title = None
            if i == 0:
                col_title = cols[j]
            if j == 0:
                view_init_dis_sample2(data_set_list[i], 3000, col_title)
            else:
                demo(
                    data_set_list[i],
                    data_size=args.data_size,
                    batch_size=args.batch_size,
                    ttl_iter=args.ttl_iter,
                    lr=args.lr,
                    sapw=sap_weights[j-1],
                    learnable_sapw=args.learnable_sapw,
                    stat_size=args.stat_size,
                    multiplot=True,
                    col_title=col_title,
                    use_scheduler=args.use_scheduler
                )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
