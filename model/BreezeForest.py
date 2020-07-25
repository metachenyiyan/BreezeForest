from __future__ import print_function
import torch
import numpy as np
from torch import nn
from torch.distributions import normal
from model.TreeLayer import TreeLayer
from model.tools import bisection, Sigmoid, logit, sigmoid


torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""
ht-1 to zt, conditional is universal or only gaussian

"""

class BreezeForest(torch.nn.Module):
    def __init__(
            self,
            dim,
            shapes,
            out_func=None,
            sap_w=0.1,
            trainable_sapw=True,
            inc_mode="no strict",
            max_eps=0.0005,
            use_mask=False
    ):
        """
        :param dim:
        :param shapes:
        list
        [
            [1, 4, 8, 16, 1],  # first dim 0
            [1, 8, 16, 32, 1]  # second dim 1
        ]
        input_shape = 2,1,1
        output_shape = 4, 8
        :param sap_w:
        :param acti_func:
        """

        super(BreezeForest, self).__init__()
        self.depth = len(shapes[0])
        self.dim = dim
        self.shapes = shapes
        while len(shapes) < self.dim:
            shapes.append(shapes[-1])
        shapes = np.array(shapes, dtype=np.int16)
        self.treeLayers = nn.ModuleList([])

        if type(sap_w) is list:
            while len(sap_w) < self.dim:
                sap_w.append(sap_w[-1])
            self.saplingWeights = nn.Parameter(logit(torch.tensor(sap_w).view(1,self.dim)), requires_grad=trainable_sapw)

        else:
            self.saplingWeights = nn.Parameter(logit(torch.ones(1, self.dim) * sap_w), requires_grad=trainable_sapw)

        for d_i in range(self.depth-1):
            self.treeLayers.append(
                TreeLayer(
                    shapes[:, d_i],
                    shapes[:, d_i+1],
                    inc_mode=inc_mode,
                    acti_func=Sigmoid()
                )
            )
        if out_func is not None:
            self.treeLayers[-1].set_acti_func(out_func)

        self.batch_example = None
        self.distributions = None
        self.dim_mask = torch.ones([1, self.dim])
        self.epsilon = torch.ones([1, self.dim]) * max_eps
        self.sap_mask = 1 - torch.ones([1, self.dim])
        self.use_mask = use_mask

    def set_max_eps(self, max_eps):
        self.epsilon = torch.ones([1, self.dim]) * max_eps

    def explain(self):
        # if self.distributions is not None:
        #     for dis in self.distributions:
        #         print(dis)
        print("sapw")
        print(sigmoid(self.saplingWeights).detach())
        print("sap_mask")
        print(self.sap_mask)
        print("tree layers")
        for tree_layer in self.treeLayers:
            tree_layer.explain()

    def forward(self, x, breeze_list=None):
        x = x * self.dim_mask
        sapw = self.get_sapw()
        x_init = x * sapw
        x = x * (1 - sapw)

        for i in range(len(self.treeLayers)):
            if (i < len(self.treeLayers) - 1):
                x = self.treeLayers[i].forward(x, x_init=None, breeze_list=breeze_list)
            else:
                x = self.treeLayers[i].forward(x, x_init=x_init, breeze_list=breeze_list)

        return x * self.dim_mask

    def get_sapw(self):
        if self.use_mask:
            return self.sap_mask
        else:
            return sigmoid(self.saplingWeights)

    def breeze_forward(self, x, breeze_list):
        x = x * self.dim_mask
        sapw = self.get_sapw()
        x_init = x * sapw
        x = x * (1 - sapw)

        assert len(breeze_list) == len(self.treeLayers)
        for i in range(len(self.treeLayers)):
            if (i < len(self.treeLayers) - 1):
                x = self.treeLayers[i].breeze_forward(x, x_init=None, breeze_bias=breeze_list[i])
            else:
                x = self.treeLayers[i].breeze_forward(x, x_init=x_init, breeze_bias=breeze_list[i])
        return x * self.dim_mask

    def train_forward(self, x, light=False):
        """
        :param x:
        :param conditions: list of tensor
        :return:
        """

        self.batch_example = x
        epsilons = self.epsilon
        assert x.size(1) == epsilons.size(1)

        if light:
            x_deltas = x + epsilons
        else:
            x_deltas = torch.cat([
                (x - epsilons).view(1, -1, x.size(1)),
                (x + epsilons).view(1, -1, x.size(1))
            ],
                dim=0
            )
        breeze_list = []
        x = self.forward(x, breeze_list)
        x_deltas = self.breeze_forward(x_deltas, breeze_list)

        if light:
            delta_u = (x_deltas - x) * self.dim_mask + 1 - self.dim_mask
            x_logDet = torch.sum(torch.log(torch.abs(delta_u))) / x.size(0) - torch.sum(torch.log(epsilons))
        else:
            du_dx = (x_deltas[1] - x_deltas[0])/(2*epsilons)
            du_dx = torch.abs(du_dx * self.dim_mask + 1 - self.dim_mask).clamp(min=0.001)
            x_logDet = torch.sum(torch.mean(torch.log(du_dx), dim=0))

        return x * self.dim_mask, x_logDet

    def func(self, x, bias_breezes, tree_weights, tree_bias, tree_scale, sapw):

        x_init = x * sapw
        x = x * (1-sapw)
        new_x = [x]

        for i in range(len(self.treeLayers)):
            init = None
            if i == len(self.treeLayers) - 1:
                init = x_init
            x, _, _ = self.treeLayers[i].forward_helper(
                x,
                tree_weights[i],
                bias_breezes[i],
                tree_bias[i],
                tree_scale[i],
                init
            )
            new_x.append(x)
        return x, new_x

    def get_one_dim_func_param(self, dim, conditions):
        bias_breezes = []
        tree_bias = []
        tree_scale = []
        tree_weights = []
        sapw = self.get_sapw()[0][dim]
        for i in range(len(self.treeLayers)):
            tree_layer = self.treeLayers[i]
            tree_weights.append(tree_layer.getTreeWeights2dim(dim))
            begin_index = sum(tree_layer.output_shape[:dim])
            end_index = begin_index + tree_layer.output_shape[dim]

            bias_breezes.append(
                    conditions[i] @ tree_layer.getBreezeWeights2dim(tree_layer.breezeBiasWeights, dim) if dim > 0 else None
            )

            tree_bias.append(tree_layer.treeBias[:, begin_index:end_index])
            tree_scale.append(tree_layer.treeScale[:, begin_index:end_index])

        return bias_breezes, tree_weights, tree_bias, tree_scale, sapw

    def compute_dis(self):
        assert self.batch_example is not None
        std = torch.std(self.batch_example, dim=0).clamp(min=0.01)
        mean = torch.mean(self.batch_example, dim=0)
        self.distributions = []
        for i in range(self.dim):
            self.distributions.append(normal.Normal(mean[i].item(), std[i].item()))

    def inverse_map(self, z, max_gap=1e-3, decay_ratio=1.0):
        """
        compute inverse of z from dimension 0 to z.size(1)
        :param z: tensor with size = batch size * dimension
        :param max_gap:
        :param decay_ratio:
        :return:
        """
        assert z.size(1) == self.dim
        cur_gap = max_gap / pow(decay_ratio, self.dim - 1)
        """
        store values of previously calculated dimensions of each depth, 
        these will be used to compute the breeze weight for the current dimension.  
        """
        previous_conditions = None
        res = []
        if self.batch_example is not None:
            self.compute_dis()

        for dim in range(self.dim):
            bias_breezes, tree_weights, tree_bias, tree_scale, saplingWeights = self.get_one_dim_func_param(dim, previous_conditions)
            if self.dim_mask[0][dim] < 0.5:
                x = torch.zeros(z.size(0), 1)
            else:

                dis = None
                if self.distributions is not None:
                    dis = self.distributions[dim]
                x = bisection(
                    target=z[:, dim].view(-1, 1),
                    inc_func=lambda y: self.func(y, bias_breezes, tree_weights, tree_bias, tree_scale, saplingWeights)[0],
                    gap_real=cur_gap,
                    distribution=dis,
                )
                cur_gap *= decay_ratio
            _, new_conditions = self.func(x, bias_breezes, tree_weights, tree_bias, tree_scale, saplingWeights)
            res.append(x)
            if dim == 0:
                previous_conditions = new_conditions
            else:
                for i in range(len(previous_conditions)):
                    previous_conditions[i] = torch.cat((previous_conditions[i], new_conditions[i]), dim=1)

        return torch.cat(res, dim=1)

    def acti_norm_init(self, batch):
        return self.forward(batch)[0]

if __name__ == "__main__":
    bf = BreezeForest(
        dim=1,
        shapes=[
            [1, 3, 1],
        ],
        sap_w=0.5,
        inc_mode="no strict"
    )

    bf.forward(torch.randn(100, 1))
    optimizer = torch.optim.Adam([
        {'params': [p for p in bf.parameters()]}
    ], lr=0.01)

    A = torch.randn(10, 1)
    u, lgdet = bf.train_forward(A)
    (-lgdet).backward()
    print(u)
    print(lgdet)
    optimizer.step()
    optimizer.zero_grad()
    u, lgdet = bf.train_forward(A)
    print(lgdet)



