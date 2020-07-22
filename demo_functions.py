
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions import uniform
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot

from model.BreezeForest import BreezeForest
MARKERSIZE = 0.5
def demo(
        distribution,
        data_size=3000,
        batch_size=200,
        ttl_iter=5000,
        lr=0.005,
        sapw=0.5,
        learnable_sapw=True,
        stat_size=30,
        multiplot=False,
        col_title=None,
        use_scheduler=False
):

    #estimaing mean and var for the dataset
    data_loader = DataLoader(distribution, batch_size=batch_size*10, shuffle=True)
    data_iter = iter(data_loader)
    ttl, _ = next(data_iter)
    std = torch.std(ttl, dim=0)
    mean = torch.mean(ttl, dim=0)

    #init data loader for training
    data_loader = DataLoader(distribution, batch_size=batch_size, shuffle=True)
    data_iter = iter(data_loader)

    #init model
    use_mask = False
    if sapw == 0.0 or sapw == 1.0:
        use_mask = True

    bf = BreezeForest(
        dim=2,
        shapes=[
            [1, 8, 16, 32, 32, 1],
        ],
        sap_w=sapw,
        trainable_sapw=learnable_sapw,
        inc_mode="no strict",
        use_mask=use_mask
    )
    if sapw == 1.0:
        bf.sap_mask = torch.ones(1, 2)
    #init actinorm's scale and bias parameters
    batch, _ = next(data_iter)
    batch.add_(-mean)
    batch.mul_(1 / std)

    with torch.no_grad():
        _ = bf.forward(batch)

    scheduler_params = dict(
        mode='min',
        factor=0.95,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=0.001,
        eps=1e-08
    )

    # init optimizer
    optimizer = optim.Adam([
        {'params': [p for p in bf.parameters()]}
    ], weight_decay=1e-5, lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)

    cur_index = 0
    cur_loss_sum = 0
    avgloss2pack = []

    for index in range(ttl_iter):

        try:
            batch, _ = next(data_iter)
            batch.add_(-mean)
            batch.mul_(1 / std)
        except StopIteration:
            data_iter = iter(data_loader)
            batch, _ = next(data_iter)
            batch.add_(-mean)
            batch.mul_(1 / std)

        z, log_det = bf.train_forward(batch)
        loss = (-log_det)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cur_loss_sum += loss.detach().item()
        cur_index += 1
        if cur_index >= stat_size:
            avgloss2pack.append(cur_loss_sum / stat_size)
            if use_scheduler:
                scheduler.step(metrics=cur_loss_sum / stat_size)
            print(
                'progress: {:.0f}%\tLoss: {:.6f}'.format(
                    index * 100.0 / ttl_iter,
                    cur_loss_sum / stat_size
                )
            )

            cur_index = 0
            cur_loss_sum = 0


    if not multiplot:
        pyplot.plot(avgloss2pack, markersize=MARKERSIZE)
        pyplot.show()
        view_init_dis_sample(distribution, data_size)

    generate_sample(bf, std, mean, data_size, multiplot, col_title)


def view_init_dis_sample(distribution, data_size):
    view_loader = DataLoader(distribution, batch_size=data_size, shuffle=True)
    for batch, _ in view_loader:
        pyplot.plot(batch[:, 0].cpu().numpy(), batch[:, 1].cpu().numpy(), ".", markersize=MARKERSIZE)
        pyplot.show()
        break

def view_init_dis_sample2(distribution, data_size, col_title=None):
    view_loader = DataLoader(distribution, batch_size=data_size, shuffle=True)
    for batch, _ in view_loader:
        pyplot.plot(batch[:, 0].cpu().numpy(), batch[:, 1].cpu().numpy(), ".", markersize=MARKERSIZE)
        frame = pyplot.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        if col_title is not None:
            frame.axes.set_title(col_title)
        break

def generate_sample(model, std, mean, sample_size, multiplot, col_title):
    model.eval()
    with torch.no_grad():

        distribution = uniform.Uniform(torch.tensor(0.01), torch.tensor(0.99))
        # distribution = normal.Normal(mean_latent, std_latent) #std_latent mean_latent
        seeds = distribution.sample(torch.Size([sample_size, 2]))
        # z = bf.inverse_map(seeds)
        generated = model.inverse_map(seeds)
        generated.mul_(std)
        generated.add_(mean)
        generated = generated.cpu().numpy()
        pyplot.plot(generated[:, 0], generated[:, 1], ".", markersize=MARKERSIZE)
        if multiplot:
            frame = pyplot.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            if col_title is not None:
                frame.axes.set_title(col_title)


        if not multiplot:
            pyplot.show()
