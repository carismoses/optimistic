from torch import nn
import torch
from torch.utils.data import random_split, ConcatDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import pb_robot

from learning.models.ensemble import Ensemble, Ensembles
from learning.train import train


class MLP(nn.Module):
    """
    Implements a MLP that makes 1-D predictions in [0,1]
    """

    def __init__(self, n_in, n_hidden, n_layers, winit_sd):
        super(MLP, self).__init__()
        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32

        self.n_in, self.hidden, self.n_layers = n_in, n_hidden, n_layers
        self.winit_sd = winit_sd
        n_out = 1
        self.N = make_layers(n_in, n_out, n_hidden, n_layers)

    def forward(self, input):
        N, n_in = input.shape
        out = self.N(input).view(N)
        return torch.sigmoid(out)

    def reset(self):
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, self.winit_sd)
            m.bias.data.normal_(0, self.winit_sd)


def model_forward(model, inputs, action, obj_name, single_batch=False):
    if single_batch:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.unsqueeze(0)
        else:
            inputs = torch.tensor(inputs, dtype=torch.float64).unsqueeze(0)

    if torch.cuda.is_available():
        model.ensembles[action][obj_name].cuda()
        inputs = inputs.cuda()

    output = model.ensembles[action][obj_name].forward(inputs)
    if torch.cuda.is_available():
        output = output.cpu()
    return output.detach().numpy()


def add_trajectory_to_dataset(domain, dataset_logger, trajectory, world):
    datapoints = []
    if len(trajectory) > 0:
        print('Adding trajectory to dataset.')
        dataset, n_actions = dataset_logger.load_trans_dataset('', ret_i=True)
        n_actions += len(trajectory)
        for (state, pddl_action, next_state, opt_accuracy) in trajectory:
            if (pddl_action.name in ['move_contact', 'pick', 'move_holding'] and domain == 'tools') or \
                (pddl_action.name in ['place', 'pickplace'] and domain == 'ordered_blocks'):
                if (pddl_action.name == 'pick' and pddl_action.args[0].readableName == 'tool') or \
                    (pddl_action.name == 'move_holding' and pddl_action.args[0].readableName == 'tool'):
                    continue
                x = world.action_to_vec(pddl_action)
                if pddl_action.name == 'move_contact':
                    contact_type = pddl_action.args[5].type
                    action = '%s-%s' % ('move_contact', contact_type)
                    obj = pddl_action.args[2].readableName
                elif pddl_action.name == 'pick':
                    action = pddl_action.name
                    obj = pddl_action.args[0].readableName
                elif pddl_action.name == 'move_holding':
                    action = pddl_action.name
                    obj = pddl_action.args[0].readableName
                dataset.add_to_dataset(action, obj, x, opt_accuracy)
                dataset_logger.save_trans_dataset(dataset, '', n_actions)
                datapoints.append((pddl_action.name, x, opt_accuracy))
    return datapoints


def make_layers(n_input, n_output, n_hidden, n_layers):
    if n_layers == 1:
        modules = [nn.Linear(n_input, n_output)]
    else:
        modules = [nn.Linear(n_input, n_hidden)]
        n_units = (n_layers-1)*[n_hidden] + [n_output]
        for n_in, n_out in zip(n_units[:-1], n_units[1:]):
            modules.append(nn.Sigmoid())
            modules.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*modules)


def train_model(model, dataset, args, plot=False):
    def inner_loop(dataset, ensemble):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        all_losses = []
        for model_e in ensemble.models:
            losses = train(dataloader,
                            model_e,
                            n_epochs=args.n_epochs,
                            loss_fn=F.binary_cross_entropy,
                            early_stop=args.early_stop)
            all_losses.append(losses)
        if plot:
            fig, ax = plt.subplots()
            for losses in all_losses:
                ax.plot(losses)
            #plt.show()
        return all_losses

    for action, action_dict in dataset.datasets.items():
        for obj, action_object_dataset in action_dict.items():
            if len(action_object_dataset) > 0:
                print('Training %s %s ensemble with |dataset| = %i' % (action, obj, len(action_object_dataset)))
                losses = inner_loop(action_object_dataset, model.ensembles[action][obj])
