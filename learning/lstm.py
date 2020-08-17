import torch
from torch import nn
from torch.nn import functional as F

class TowerLSTM(nn.Module):
    def __init__(self, n_in, n_hidden):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        """
        super(TowerLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_in,
                            hidden_size=n_hidden,
                            num_layers=2,
                            batch_first=True)



        self.O = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               #nn.Linear(n_hidden, n_hidden),
                               #nn.ReLU(),
                               nn.Linear(n_hidden, 1))
        
        self.n_in, self.n_hidden = n_in, n_hidden

    def forward(self, towers, k):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K, _ = towers.shape
        x = torch.flip(towers, dims=[1])
        x, _ = self.lstm(x)

        x = torch.mean(x, dim=1)
        x = self.O(x).view(N)
        return torch.sigmoid(x)

        