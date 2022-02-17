import torch
from torch import nn
from torch.nn import functional as F

from learning.utils import make_layers

class MLP(nn.Module):
    """
    Implements a MLP that makes 1-D predictions in [0,1]
    """

    def __init__(self, n_in, n_hidden, n_layers):
        super(MLP, self).__init__()
        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32

        self.n_in, self.hidden, self.n_layers = n_in, n_hidden, n_layers
        n_out = 1
        self.N = make_layers(n_in, n_out, n_hidden, n_layers)

    def forward(self, input):
        N, n_in = input.shape
        out = self.N(input).view(N)
        return torch.sigmoid(out)
