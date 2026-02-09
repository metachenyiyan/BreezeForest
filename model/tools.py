
from math import sqrt
import torch
import torch.nn as nn
from torch.distributions import normal

# torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Commented out for CPU compatibility
torch.set_default_tensor_type('torch.FloatTensor')

class Logit(nn.Module):
    def forward(self, x):
        return torch.log(x/(1-x))

def logit(x,max_v=1.0):
    y = x / max_v
    return torch.log(y / (1 - y))

def sigmoid(x, max_v=1.0):
    # 2*max_v -
    # Use differentiable operations instead of floor_divide
    # torch.sign(x) returns {-1, 0, 1}, we want {0, 1, 1}
    # Replace floor division with clamped float division
    sign = torch.clamp((torch.sign(x) + 1), min=0, max=1)  # {-1,0,1} -> {0,1,2} -> {0,1,1}
    x_abs = torch.abs(x)
    res = max_v/(1 + torch.exp(-x_abs))
    res = res * sign + (1 - sign) * (max_v - res)
    return res

class ActiId(nn.Module):
    def forward(self, x):
        return x
    def reverse(self, x):
        return x
    def derivative(self, x):
        return torch.ones_like(x)

class Relu(nn.Module):
    def forward(self, x):
        return x.clamp(min=0.0)
    def derivative(self, x):
        return torch.sign(x).clamp(min=0.0)

class Sigmoid(nn.Module):
    __constants__ = ['max_k']

    def __init__(self, max_k=1):
        super(Sigmoid, self).__init__()
        self.coeff = max_k * 4

    def forward(self, x):
        x = x * self.coeff
        return sigmoid(x)

    def derivative(self, x):
        return self.coeff * self.forward(x) * (1 - self.forward(x))

    def reverse(self, x):
        return torch.log(x / (1 - x))/self.coeff

class CdfGaussian(nn.Module):
    def __init__(self):
        super(CdfGaussian, self).__init__()

    def forward(self, x):
        return 0.5*(1+torch.erf(x/sqrt(2.0)))

    def reverse(self, y):
        return torch.erfinv(2*y - 1) * sqrt(2.0)


def bisection(target, inc_func,  distribution=None, gap_dis=0.1, gap_real=0.001, anomaly_dis=1-0.001):
    """
    tensor([
    [0.0800],
    [-0.2402],
    [0.3205],
    [0.0400],
    [-0.1040],
    [-0.1601]
    ])
    :param inc_func:  Rn -> Rn piece-wise increasing function, dimensions are independent as a batch
    :param guess_L: batch_size
    :param guess_U: batch_size
    :param gap_epsilon:
    :return:
    """

    if distribution is None:
        distribution = normal.Normal(0, 1)


    guess_U = torch.ones_like(target)
    guess_L = torch.zeros_like(target)
    # Use differentiable comparison instead of floor_divide
    done_mask = (guess_L + gap_dis >= guess_U).float()

    while(not torch.all(done_mask.bool())):
        guess = (guess_L + guess_U) / 2
        # print(guess)
        res = torch.sign(inc_func(distribution.icdf(guess)) - target)
        # Use differentiable operations: sign gives {-1,0,1}, we want masks for >= 0 and <= 0
        geq_mask = (res >= 0).float()
        leq_mask = (res <= 0).float()
        guess_U = guess * geq_mask + guess_U*(-geq_mask + 1)
        guess_L = guess * leq_mask + guess_L*(-leq_mask + 1)
        done_mask = (guess_L + gap_dis >= guess_U).float()

    guess_U.clamp_(max=anomaly_dis)
    guess_L.clamp_(min=1-anomaly_dis)

    guess_U = distribution.icdf(guess_U)
    guess_L = distribution.icdf(guess_L)


    guess = (guess_L + guess_U) / 2
    done_mask = (guess_L + gap_real >= guess_U).float()
    while (not torch.all(done_mask.bool())):
        k = inc_func(guess)
        res = torch.sign(k - target)
        geq_mask = (res >= 0).float()
        leq_mask = (res <= 0).float()
        gp = guess_U
        guess_U = guess * geq_mask + guess_U*(-geq_mask + 1)
        guess_L = guess * leq_mask + guess_L*(-leq_mask + 1)
        done_mask = (guess_L + gap_real >= guess_U).float()
        guess = (guess_L + guess_U) / 2
    return guess


def get_epsilons(max_epsilon, dim, decay=1.0):
    cur_eps = max_epsilon / pow(decay, dim - 1)
    epsilons = []
    for _ in range(dim):
        epsilons.append(cur_eps)
        cur_eps *= decay
    return epsilons


def actinorm_init_bias(param, x, offset=0, dim=0):
    if param is None:
        shape = list(x.shape)
        shape[dim] = 1
        mean = torch.mean(x.data, dim=dim).view(*shape)
        return nn.Parameter(mean + offset)
    else:
        return param

def actinorm_init_scale(param, x, var=1.0, func=None, dim=0, min_std=1.0):
    if param is None:
        if x.shape[0] == 1:
            std=torch.tensor(min_std)
        else:
            shape = list(x.shape)
            shape[dim] = 1
            std = torch.std(x.data, dim=dim).view(*shape).clamp(min=min_std)
        if func is None:
            func = torch.sqrt
        return nn.Parameter(func(abs(var)/std))
    else:
        return param
