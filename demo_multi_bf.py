
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot

from model.MultiBF import MultiBF
from model.distribution2d import *

MARKERSIZE = 0.5


def demo_multi_bf(
        distribution,
        n_components=3,
        data_size=3000,
        batch_size=200,
        ttl_iter=5000,
        lr=0.005,
        sapw=0.5,
        learnable_sapw=True,
        stat_size=30,
        use_scheduler=False
):
    # Estimate mean and std for normalization
    data_loader = DataLoader(distribution, batch_size=batch_size * 10, shuffle=True)
    data_iter = iter(data_loader)
    ttl, _ = next(data_iter)
    std = torch.std(ttl, dim=0)
    mean = torch.mean(ttl, dim=0)

    data_loader = DataLoader(distribution, batch_size=batch_size, shuffle=True)
    data_iter = iter(data_loader)

    use_mask = False
    if sapw == 0.0 or sapw == 1.0:
        use_mask = True

    mbf = MultiBF(
        n_components=n_components,
        dim=2,
        shapes=[[1, 8, 16, 32, 32, 1]],
        sap_w=sapw,
        trainable_sapw=learnable_sapw,
        inc_mode="no strict",
        use_mask=use_mask
    )

    if sapw == 1.0:
        for bf in mbf.components:
            bf.sap_mask = torch.ones(1, 2)

    # ActiNorm init
    batch, _ = next(data_iter)
    batch = (batch - mean) / std
    with torch.no_grad():
        mbf.forward(batch)

    optimizer = optim.Adam(mbf.parameters(), weight_decay=1e-5, lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.95, patience=1,
        threshold=0.0001, threshold_mode='abs', min_lr=0.001
    )

    cur_index = 0
    cur_loss_sum = 0
    avgloss2pack = []

    for index in range(ttl_iter):
        try:
            batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch, _ = next(data_iter)
        batch = (batch - mean) / std

        log_prob = mbf.train_forward(batch)
        loss = -log_prob
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cur_loss_sum += loss.detach().item()
        cur_index += 1
        if cur_index >= stat_size:
            avg_loss = cur_loss_sum / stat_size
            avgloss2pack.append(avg_loss)
            if use_scheduler:
                scheduler.step(metrics=avg_loss)
            weights = mbf.get_mixture_weights().detach()
            print(
                'progress: {:.0f}%\tLoss: {:.6f}\tWeights: {}'.format(
                    index * 100.0 / ttl_iter, avg_loss,
                    [f'{w:.3f}' for w in weights.tolist()]
                )
            )
            cur_index = 0
            cur_loss_sum = 0

    # Plot loss
    pyplot.figure(figsize=(10, 4))
    pyplot.plot(avgloss2pack, markersize=MARKERSIZE)
    pyplot.title("Training Loss")
    pyplot.xlabel("Stat window")
    pyplot.ylabel("Neg log-likelihood")
    pyplot.show()

    # Plot original data
    view_loader = DataLoader(distribution, batch_size=data_size, shuffle=True)
    batch, _ = next(iter(view_loader))
    pyplot.figure(figsize=(6, 6))
    pyplot.plot(batch[:, 0].numpy(), batch[:, 1].numpy(), ".", markersize=MARKERSIZE)
    pyplot.title("Original Data")
    pyplot.show()

    # Generate samples from mixture
    mbf.eval()
    with torch.no_grad():
        samples = mbf.inverse_map(n_samples=data_size)
        samples = samples * std + mean
        samples = samples.numpy()
        pyplot.figure(figsize=(6, 6))
        pyplot.plot(samples[:, 0], samples[:, 1], ".", markersize=MARKERSIZE)
        pyplot.title(f"MultiBF Generated ({n_components} components)")
        pyplot.show()

    print(f"\nFinal mixture weights: {mbf.get_mixture_weights().detach()}")


parser = argparse.ArgumentParser()
parser.add_argument("--n_components", type=int, default=3)
parser.add_argument("--data_size", type=int, default=3000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--ttl_iter", type=int, default=8000)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--sapw", type=float, default=0.5)
parser.add_argument("--learnable_sapw", type=bool, default=False)
parser.add_argument("--stat_size", type=int, default=30)
parser.add_argument("--use_scheduler", type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    dis = GAUSSIANS(flip_var_order=True)
    demo_multi_bf(
        dis,
        n_components=args.n_components,
        data_size=args.data_size,
        batch_size=args.batch_size,
        ttl_iter=args.ttl_iter,
        lr=args.lr,
        sapw=args.sapw,
        learnable_sapw=args.learnable_sapw,
        stat_size=args.stat_size,
        use_scheduler=args.use_scheduler
    )
