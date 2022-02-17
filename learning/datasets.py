import numpy as np
import torch

from torch.utils.data import Dataset
from learning.models.gnn import TransitionGNN


class MoveContactDataset(Dataset):
    # inputs to the MoveContactDataset are:
    #   object_ids (2-d)
    #   rel_pose (3-d)
    #   goal_pose (3-d)
    # output is feasibility: 1-d
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


class TransDataset(Dataset):
    def __init__(self):
        self.object_features = torch.tensor([], dtype=torch.float64)
        self.edge_features = torch.tensor([], dtype=torch.float64)
        self.actions = torch.tensor([], dtype=torch.float64)
        self.delta_edge_features = torch.tensor([], dtype=torch.float64)
        self.next_edge_features = torch.tensor([], dtype=torch.float64)
        self.optimistic_accuracy = torch.tensor([], dtype=torch.float64)

    def __getitem__(self, ix, full_info=False):
        if full_info:
            return [self.object_features[ix],
                    self.edge_features[ix],
                    self.actions[ix],
                    self.next_edge_features[ix],
                    self.delta_edge_features[ix],
                    self.optimistic_accuracy[ix]]
        else:
            return [self.object_features[ix],
                    self.edge_features[ix],
                    self.actions[ix]], \
                    self.optimistic_accuracy[ix]


    def __len__(self):
        """
        The total number of datapoints in the entire dataset.
        """
        return len(self.object_features)

    def remove_elements(self, remove_list):
        mask = np.ones(len(self.edge_features), dtype=bool)
        mask[remove_list] = False
        self.object_features = self.object_features[mask]
        self.edge_features = self.edge_features[mask]
        self.actions = self.actions[mask]
        self.next_edge_features = self.next_edge_features[mask]
        self.delta_edge_features = self.delta_edge_features[mask]
        self.optimistic_accuracy = self.optimistic_accuracy[mask]

    def add_to_dataset(self, object_features, edge_features, action, next_edge_features, delta_edge_features, optimistic_accuracy):
        if not isinstance(object_features, torch.Tensor):
            self.object_features = torch.cat([self.object_features, torch.tensor([object_features])])
            self.edge_features = torch.cat([self.edge_features, torch.tensor([edge_features])])
            self.actions = torch.cat([self.actions, torch.tensor([action])])
            self.next_edge_features = torch.cat([self.next_edge_features, torch.tensor([next_edge_features])])
            self.delta_edge_features = torch.cat([self.delta_edge_features, torch.tensor([delta_edge_features])])
            self.optimistic_accuracy = torch.cat([self.optimistic_accuracy, torch.tensor([optimistic_accuracy])])
        else:
            self.object_features = torch.cat([self.object_features, object_features.unsqueeze(dim=0)])
            self.edge_features = torch.cat([self.edge_features, edge_features.unsqueeze(dim=0)])
            self.actions = torch.cat([self.actions, action.unsqueeze(dim=0)])
            self.next_edge_features = torch.cat([self.next_edge_features, next_edge_features.unsqueeze(dim=0)])
            self.delta_edge_features = torch.cat([self.delta_edge_features, delta_edge_features.unsqueeze(dim=0)])
            self.optimistic_accuracy = torch.cat([self.optimistic_accuracy, optimistic_accuracy.unsqueeze(dim=0)])

    # balance so equal labels and balanced actions within labels
    # NOTE: only filtering on actions since that's all that matters in simple block domain
    def balance(self):
        # collect unique actions and their indices
        unique_actions = {}
        for i in range(len(self)):
            x, y = self[i]
            str_action = ','.join([str(int(a)) for a in x[2]])
            if (str_action, int(y)) not in unique_actions:
                unique_actions[(str_action, int(y))] = [i]
            else:
                unique_actions[(str_action, int(y))].append(i)

        # first remove extra elements
        remove_list = []
        for (action, label), indices in unique_actions.items():
            remove_list += indices[1:]
        self.remove_elements(remove_list)

        # collect new indices in reduced dataset
        for i in range(len(self)):
            x, y, = self[i]
            str_action = ','.join([str(int(a)) for a in x[2]])
            unique_actions[(str_action, int(y))] = i

        ## make new dataset with one of each of whichever label has more
        ## then do the same for the other label and randomly sample until they are even
        pos = np.sum([ay[1] for ay in unique_actions])
        neg = len(self) - pos
        if pos < neg:
            min_label = 1
            n_min = pos
            n_max = neg
        else:
            min_label = 0
            n_min = neg
            n_max = pos
        assert n_min!=0, 'Cannot balance dataset, no %i labels exist' % min_label
        assert n_max!=0, 'Cannot balance dataset, no %i labels exist' % max_label
        while n_min < n_max:
            match = False
            while not match:
                random_action = random.choice(list(unique_actions))
                if random_action[1] == min_label:
                    dataset_i = unique_actions[random_action]
                    new_sample = self.__getitem__(dataset_i, full_info=True)
                    self.add_to_dataset(*new_sample)
                    match = True
            n_min += 1
        print('Done balancing dataset.')


# for testing
def preprocess(args, dataset, type='successful_actions'):
    xs, ys = dataset[:]
    remove_list = []
    # only keep samples with successful actions/edge changes
    distinct_actions = []
    actions_counter = {}
    for i, ((object_features, edge_features, action), next_edge_features) in enumerate(dataset):
        a = tuple(action.numpy())
        if a not in distinct_actions:
            distinct_actions.append(a)
            actions_counter[a] = [i]
        else:
            actions_counter[a] += [i]
    min_distinct_actions = min([len(counter) for counter in actions_counter.values()])
    for a in distinct_actions:
        remove_list += actions_counter[a][min_distinct_actions:]

    dataset.remove_elements(remove_list)
