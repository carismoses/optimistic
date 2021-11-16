import torch
from torch import nn


def make_layers(n_input, n_output, n_hidden, n_layers):
    if n_layers == 1:
        modules = [nn.Linear(n_input, n_output)]
    else:
        modules = [nn.Linear(n_input, n_hidden)]
        n_units = (n_layers-1)*[n_hidden] + [n_output]
        for n_in, n_out in zip(n_units[:-1], n_units[1:]):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*modules)


class GNN(nn.Module):
    def __init__(self, n_of_in, n_ef_in, n_hidden, n_layers):
        """
        :param n_of_in: Dimensionality of object features
        :param n_ef_in: Dimensionality of edge features
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super().__init__()
        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32
        self.n_of_in, self.n_ef_in, self.n_hidden = n_of_in, n_ef_in, n_hidden

        # Initial embedding of node features into latent state
        self.Ni = make_layers(n_of_in, n_hidden, n_hidden, 3)

        # Message function that compute relation between two nodes and outputs a message vector.
        self.E = make_layers(2*n_hidden, n_hidden, n_hidden, n_layers)

        # Update function that updates a node based on the sum of its messages.
        self.N = make_layers(n_hidden, n_hidden, n_hidden, n_layers)

    def embed_node(self, input):
        """
        :param input: Network input. First item is always
                        object_features (N, K, n_of_in)
        """
        object_features = input[0]
        N, K, n_of_in = object_features.shape

        # Pass each object feature from encoder
        # x.shape = (N*K, n_of_in)
        x = object_features.view(-1, n_of_in)

        # Calculate the hidden state for each node
        # hn.shape = (N*K, n_hidden) --> (N, K, n_hidden)
        hn = self.Ni(x).view(N, K, self.n_hidden)
        return hn

    def node_fn(self, he):
        """
        :param he: Hidden edge features (N, K, K, n_hidden)
        """
        N, K, K, n_hidden = he.shape

        # sum all edge features going into each node
        # he_sum.shape (N, K, n_hidden) --> (N*K, n_hidden)
        he_sum = torch.sum(he, dim=2).view(-1, n_hidden)

        # Calculate the updated node features.
        # hn.shape = (N, K, n_hidden)
        hn = self.N(he_sum).view(N, K, n_hidden)
        return hn

    def edge_fn(self, hn):
        """
        :param hn: Node hidden states (N, K, n_hidden)
        """
        N, K, n_hidden = hn.shape

        # Get features between all nodes and mask with edge features.
        # x.shape = (N, K, K, n_hidden)
        # xx.shape = (N, K, K, 2*n_hidden) --> (N*K*K, 2*n_hidden)
        x = hn[:, :, None, :].expand(-1, -1, K, -1)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)

        # Calculate the hidden edge state for each edge
        # he.shape = (N, K, K, n_hidden)
        he = self.E(xx).view(N, K, K, self.n_hidden)
        return he

    def forward(self, input):
        """
        :param input: network inputs. First element in list is always object_features
        """

        # Calculate initial node and edge hidden states
        hn = self.embed_node(input)
        he = self.embed_edge(hn, input)

        I = 1
        for i in range(I):
            # Calculate node hidden state
            hn = self.node_fn(he)

            # Calculate edge hidden state
            he = self.edge_fn(hn)

        y = self.final_pred(he)
        return y


class TransitionGNN(GNN):
    def __init__(self, n_of_in, n_ef_in, n_af_in, n_hidden, n_layers):
        """ This network is given three inputs of size (N, K, n_of_in), (N, K, K, n_ef_in), and (N, n_af_in).
        N is the batch size, K is the number of objects (including a table)
        :param n_of_in: Dimensionality of object features
        :param n_ef_in: Dimensionality of edge features
        :param n_af_in: Dimensionality of action features
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super(TransitionGNN, self).__init__(n_of_in, n_ef_in, n_hidden, n_layers)

        # Initial embedding of edge features and action into latent state
        self.Ei = make_layers(n_ef_in+n_af_in+2*n_hidden, n_hidden, n_hidden, 1)

        # Final function to get next state edge predictions
        self.F = make_layers(n_hidden, 1, n_hidden, 3)

        self.n_af_in = n_af_in

    def embed_edge(self, hn, input):
        """
        :param hn: Hidden node state (N, K, n_hidden)
        :param input: Network inputs. Elements are
            object_features (N, K, n_hidden)
            edge_features (N, K, K, n_ef_in)
            action (N, n_af_in)
        """
        object_features, edge_features, action = input
        N, K, n_hidden = hn.shape
        N, K, n_of_in = object_features.shape
        N, K, K, n_ef_in = edge_features.shape
        N, n_af_in = action.shape

        # Append edge features, action, and hidden node states
        # a.shape = (N, K, K, n_af_in)
        # hn_exp.shape = (N, K, K, n_hidden)
        # hnhn.shape = (N, K, K, 2*n_hidden)
        # xahnhn.shape = (N, K, K, n_ef_in+n_af_in+2*n_hidden) --> (N*K*K, n_ef_in+n_af_in+2*n_hidden)
        a = action[:, None, None, :].expand(-1, K, K, -1)
        hn_exp = hn[:, :, None, :].expand(-1, -1, K, -1)
        hnhn = torch.cat([hn_exp, hn_exp.transpose(1, 2)], dim=3)
        xahnhn = torch.cat([edge_features, a, hnhn], dim=3)
        xahnhn = xahnhn.view(-1, n_ef_in+n_af_in+2*n_hidden)

        # Calculate the hidden edge state for each node
        # he.shape = (N*K*K, n_hidden) --> (N, K, K, n_hidden)
        he = self.Ei(xahnhn).view(N, K, K, self.n_hidden)
        return he

    def final_pred(self, he):
        N, K, K, n_hidden = he.shape

        # Calculate the final edge predictions
        # he.shape = (N, K, K, n_hidden)
        # x.shape = (N, n_hidden)
        # y.shape = (N, 1) --> (N)
        x = torch.mean(he, dim=(1,2))
        y = self.F(x).view(N)
        return torch.sigmoid(y)
