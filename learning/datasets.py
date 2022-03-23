import numpy as np
import torch

from torch.utils.data import Dataset


class OptDictDataset:
    def __init__(self, objects=['yellow_block', 'blue_block']):
        # object in {yellow_block, blue_block}
        self.datasets = {}
        for action in ['pick', 'push-poke', 'push-push_pull']:
            for obj in objects:
                if action not in self.datasets:
                    self.datasets[action] = {}
                self.datasets[action][obj] = OptDataset()


    def __getitem__(self, action_type, object):
        return self.datasets[action_type][object]


    def __len__(self):
        dlen = 0
        for action, action_dict in self.datasets.items():
            for object, dataset in action_dict.items():
                dlen += len(dataset)
        return dlen


    def add_to_dataset(self, action_type, object, x, y):
        self.datasets[action_type][object].add_to_dataset(x, y)


class OptDataset(Dataset):
    def __init__(self):
        self.xs = torch.tensor([], dtype=torch.float64)
        self.ys = torch.tensor([], dtype=torch.float64)


    def __getitem__(self, ix):
        return [self.xs[ix],
                self.ys[ix]]


    def __len__(self):
        """
        The total number of datapoints in the entire dataset.
        """
        return len(self.xs)


    def add_to_dataset(self, x, y):
        if not isinstance(x, torch.Tensor):
            self.xs = torch.cat([self.xs, torch.tensor([x])])
            self.ys = torch.cat([self.ys, torch.tensor([y])])
        else:
            self.xs = torch.cat([self.xs, x.unsqueeze(dim=0)])
            self.ys = torch.cat([self.ys, y.unsqueeze(dim=0)])
