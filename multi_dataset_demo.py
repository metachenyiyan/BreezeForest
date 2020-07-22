from demo_functions import *
from model.distribution2d import *
import numpy as np
import matplotlib.pyplot as plt

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
                data_size=3000,
                batch_size=200,
                ttl_iter=8000,
                lr=0.005,
                sapw=sap_weights[j-1],
                learnable_sapw=False,
                stat_size=30,
                multiplot=True,
                col_title=col_title,
                use_scheduler=False
            )

fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
