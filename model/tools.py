
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


def bisection(target, inc_func, distribution=None, gap_dis=0.1, gap_real=0.001, anomaly_dis=1-0.001):
    """
    Find x such that inc_func(x) = target via two-stage bisection.
    Stage 1: coarse search in distribution CDF space [0, 1].
    Stage 2: fine search in real space with tighter tolerance.

    :param target: target values (batch_size, 1)
    :param inc_func: piece-wise increasing function, batched over dim 0
    :param distribution: reference distribution (default: standard normal)
    :param gap_dis: convergence tolerance for CDF-space stage
    :param gap_real: convergence tolerance for real-space stage
    :param anomaly_dis: CDF clamp boundary to avoid infinite icdf values
    :return: x such that inc_func(x) ≈ target
    """
    if distribution is None:
        distribution = normal.Normal(0, 1)

    def _bisect(lo, hi, eval_fn, gap):
        while not torch.all(lo + gap >= hi):
            mid = (lo + hi) / 2
            too_high = (eval_fn(mid) >= target)
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(~too_high, mid, lo)
        return lo, hi

    # Stage 1: coarse search in CDF space
    lo, hi = _bisect(
        torch.zeros_like(target), torch.ones_like(target),
        lambda m: inc_func(distribution.icdf(m)), gap_dis,
    )
    # Map to real space, clamping to avoid icdf blowup at boundaries
    lo = distribution.icdf(lo.clamp(min=1 - anomaly_dis))
    hi = distribution.icdf(hi.clamp(max=anomaly_dis))

    # Stage 2: fine search in real space
    lo, hi = _bisect(lo, hi, inc_func, gap_real)
    return (lo + hi) / 2


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
