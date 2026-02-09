
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
from model.tools import Sigmoid, actinorm_init_bias, actinorm_init_scale

# torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Commented out for CPU compatibility
torch.set_default_tensor_type('torch.FloatTensor')

class TreeLayer(torch.nn.Module):
    """
    One layer of breeze forest
    """
    def __init__(
            self,
            input_shape,
            output_shape,
            acti_func=None,
            inc_mode="no strict"
    ):
        """

        :param input_shape: list, indicating input width for each dimension [32,16,16,16]
        :param output_shape: list, indicating output width for each dimension [32,16,16,16]
        :param sap_w: Float, initial weight for the skip connection from the input of breezeforest to it's output
        :param acti_func: by default sigmoid
        :param inc_mode:
            "strict": each conditional cumulative density function is strictly increasing
            "no strict": each conditional cumulative density function is increasing, zero density is allowed
            "any": no monotone is imposed, can be seen as normal normalizing flow
        """

        super(TreeLayer, self).__init__()
        self.inc_mode = inc_mode

        self.input_shape = input_shape
        self.dim2endrow = np.cumsum(input_shape)
        self.output_shape = output_shape
        self.dim2endcol = np.cumsum(output_shape)

        self.in_size = sum(input_shape)
        self.out_size = sum(output_shape)
        self.strict = True
        self.inc = True
        self.z_dim = len(input_shape)

        if inc_mode == "strict":
            self.treeWeights = nn.Parameter(
                torch.log(torch.abs(torch.randn(self.in_size, self.out_size))/sqrt(self.in_size/self.z_dim))
            )
        elif inc_mode == "no strict":
            self.treeWeights = nn.Parameter(
                torch.sqrt(torch.abs(torch.randn(self.in_size, self.out_size))/sqrt(self.in_size/self.z_dim))
            )
            self.strict = False
        else:
            self.treeWeights = nn.Parameter(torch.randn(self.in_size, self.out_size)/sqrt(self.in_size/self.z_dim))
            self.inc = False
            self.strict = False

        self.breezeBiasWeights = nn.Parameter(torch.zeros(self.in_size, self.out_size))
        self.init_mask()

        if acti_func is None:
            self.acti_func = torch.nn.Sigmoid()
        else:
            self.acti_func = acti_func

        self.scaleBreezeScale = None
        self.scaleBreezeBias = None
        self.treeBias = None
        self.treeScale = None


    def init_mask(self):
        """
        Init mask for block-wise diagonal weight matrix
        """
        self.treeMask = torch.zeros(self.in_size, self.out_size)
        begin_row = 0
        begin_col = 0
        for in_size, out_size in zip(self.input_shape, self.output_shape):
            self.treeMask[begin_row:(begin_row + in_size), begin_col: (begin_col + out_size)] = 1.0
            begin_row += in_size
            begin_col += out_size

        """
        Init mask for block-wise triangular weight matrix  
        """

        self.breezeMask = torch.zeros(self.in_size, self.out_size)
        begin_row = 0
        begin_col = self.output_shape[0]
        for in_size, out_size in zip(self.input_shape, self.output_shape[1:]):
            self.breezeMask[0:begin_row+in_size, begin_col:begin_col+out_size] = 1.0
            self.breezeBiasWeights.data[
                0:begin_row+in_size,
                begin_col:begin_col+out_size
            ] = torch.randn(begin_row+in_size, out_size)/sqrt(begin_row+in_size)

            begin_row += in_size
            begin_col += out_size


    def set_acti_func(self, acti_func):
        self.acti_func = acti_func

    def breeze_forward(
            self,
            x,
            breeze_bias,
            x_init
    ):
        tree_matrix = self.getRealTreeWeights(self.treeWeights) * self.treeMask
        tree_out, self.treeBias, self.treeScale = self.forward_helper(
            x,
            tree_matrix,
            breeze_bias,
            self.treeBias,
            self.treeScale,
            x_init
        )
        return tree_out

    def forward(
            self,
            x,
            x_init,
            breeze_list=None
    ):
        tree_matrix = self.getRealTreeWeights(self.treeWeights) * self.treeMask
        breeze_bias = x @ (self.breezeBiasWeights * self.breezeMask)
        tree_out, self.treeBias, self.treeScale = self.forward_helper(
            x,
            tree_matrix,
            breeze_bias,
            self.treeBias,
            self.treeScale,
            x_init
        )
        if breeze_list is not None:
            breeze_list.append(breeze_bias)
        return tree_out

    def forward_helper(
            self,
            x,
            tree_matrix,
            breeze_bias,
            tree_bias,
            tree_scale,
            x_init=None
    ):
        """
        conditional cdf for one dimension or parallel inference of conditional cdfs for all dimensions
        :param x_init: the original input of breezeforest, if not None, add to the output with gate saplingWeights
        """
        x = x @ tree_matrix
        if x_init is not None:
            x = x + x_init

        if breeze_bias is not None:
            x = x + breeze_bias

        tree_bias = actinorm_init_bias(tree_bias, x)
        if self.strict:
            tree_scale = actinorm_init_scale(tree_scale, x, func=torch.log)
        else:
            tree_scale = actinorm_init_scale(tree_scale, x)

        x = x - tree_bias
        if self.strict:
            x = x * torch.exp(tree_scale)
        else:
            x = x * torch.pow(tree_scale, 2)
        return self.acti_func.forward(x), tree_bias, tree_scale

    def getTreeWeights2dim(self, dim):
        end_out = self.dim2endcol[dim]
        begin_out = end_out - self.output_shape[dim]
        end_in = self.dim2endrow[dim]
        begin_in = end_in - self.input_shape[dim]
        matrix = self.getRealTreeWeights(self.treeWeights[begin_in:end_in, begin_out:end_out])
        return matrix

    def getBreezeWeights2dim(self, weights, dim):
        assert dim > 0
        end_out = self.dim2endcol[dim]
        begin_out = end_out - self.output_shape[dim]
        end_in = self.dim2endrow[dim-1]
        return weights[:end_in, begin_out:end_out]

    def getRealTreeWeights(self, treeWeights):
        if self.strict:
            return torch.exp(treeWeights)
        if self.inc:
            return torch.pow(treeWeights, 2)
        else:
            return treeWeights



    def simple_forward(self, x_input, x_init=None):
        tree_matrix = self.getRealTreeWeights(self.treeWeights) * self.treeMask
        breeze_bias = x_input @ (self.breezeBiasWeights * self.breezeMask)
        x_out, self.treeBias, self.treeScale = self.forward_helper(
            x_input,
            tree_matrix,
            breeze_bias,
            self.treeBias,
            self.treeScale,
            x_init
        )
        return x_out

    def explain(self):
        """
        :return:
        """

        print("====TreeLayer====")
        print("treeScale")
        print(self.treeScale.data)
        print("treeBias")
        print(self.treeBias.data)
        print("tree matrix")
        print((self.getRealTreeWeights(self.treeWeights) * self.treeMask))
        print("breezeBiasWeights")
        print((self.breezeBiasWeights * self.breezeMask))
