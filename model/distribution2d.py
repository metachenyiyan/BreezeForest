import torch
import torch.distributions as D
from torch.utils.data import Dataset
from sklearn.datasets import make_moons, make_circles, make_blobs
import numpy as np


torch.set_default_tensor_type('torch.cuda.FloatTensor')

class BUTTERFLY(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):
        x, self.y = make_butterfly(n_samples=dataset_size, shuffle=True)
        self.x = torch.Tensor(x)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class MOONS(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):
        x, self.y = make_moons(n_samples=dataset_size, shuffle=True, noise=0.05)
        self.x = torch.Tensor(x)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class CIRCLE(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):
        x, self.y = make_circles(n_samples=dataset_size, shuffle=True, noise=0.01)
        self.x = torch.Tensor(x)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class BLOBS(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):
        x, self.y = make_blobs(n_samples=dataset_size, shuffle=True)
        self.x = torch.Tensor(x)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class CHECKERBOARD(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):

        self.x = torch.from_numpy(sample2d('checkerboard', batch_size=dataset_size)).type(torch.cuda.FloatTensor)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.x[i]

class GAUSSIANS(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):

        self.x = torch.from_numpy(sample2d('8gaussians', batch_size=dataset_size)).type(torch.cuda.FloatTensor)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.x[i]


class SPIRALS(Dataset):
    def __init__(self, dataset_size=5000, flip_var_order=False):

        self.x = torch.from_numpy(sample2d('2spirals', batch_size=dataset_size)).type(torch.cuda.FloatTensor)
        if flip_var_order:
            self.x = self.x.flip([1])
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.x[i]


def sample2d(data, batch_size=200):
    rng = np.random.RandomState()

    if data == '8gaussians':
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset)
        dataset /= 1.414
        return dataset

    elif data == '2spirals':
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == 'checkerboard':
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    else:
        raise RuntimeError

class ButterFlyDis(D.Distribution):
    def __init__(self, flip_var_order):
        super().__init__()
        self.flip_var_order = flip_var_order
        self.p_x2 = D.Normal(0, 4)
        self.p_x1 = lambda x2: D.Normal(0.25 * x2**2, abs(x2))

    def rsample(self, sample_shape=torch.Size()):
        x2 = self.p_x2.sample(sample_shape)
        x1 = self.p_x1(x2).sample()
        if self.flip_var_order:
            return torch.stack((x2, x1), dim=-1).squeeze(), torch.ones(sample_shape)
        else:
            return torch.stack((x1, x2), dim=-1).squeeze(), torch.ones(sample_shape)

    def log_prob(self, value):
        if self.flip_var_order:
            value = value.flip(1)
        return self.p_x1(value[:,1]).log_prob(value[:,0]) + self.p_x2.log_prob(value[:,1])

def make_butterfly(n_samples, shuffle=True):
    dis = ButterFlyDis(shuffle)
    return dis.rsample(torch.Size([n_samples]))