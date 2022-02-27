from torch import nn
import torch
from torch.utils.data import random_split, ConcatDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pb_robot

from learning.models.ensemble import Ensemble, Ensembles
from learning.train import train


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


def model_forward(contact_type, model, inputs, single_batch=False):
    if single_batch:
        single_inputs = inputs
        inputs = torch.tensor(inputs, dtype=torch.float64).unsqueeze(0)

    if torch.cuda.is_available():
        model.ensembles[contact_type].cuda()
        inputs = inputs.cuda()

    output = model.ensembles[contact_type].forward(inputs)
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
            if (pddl_action.name in ['move_contact', 'pick'] and domain == 'tools') or \
                (pddl_action.name in ['place', 'pickplace'] and domain == 'ordered_blocks'):
                if pddl_action.name == 'pick' and pddl_action.args[0].readableName == 'tool':
                    continue
                x = world.action_to_vec(pddl_action)
                if dataset_logger.args.goal_type == 'push':
                    contact_type = pddl_action.args[5].type
                    dataset[contact_type].add_to_dataset(x, opt_accuracy)
                elif dataset_logger.args.goal_type == 'pick':
                    dataset.add_to_dataset(x, opt_accuracy)
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
            modules.append(nn.ReLU())
            modules.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*modules)


def initialize_model(args, base_args, types=None):
    if args.goal_type == 'push':
        return Ensembles(MLP, base_args, args.n_models, types)
    elif args.goal_type == 'pick':
        return Ensemble(MLP, base_args, args.n_models)


def train_model(model, dataset, args, types=None, plot=False):
    if args.goal_type == 'push':
        for type in types:
            if len(dataset[type]) > 0:
                print('Training %s ensemble with |dataset| = %i' % (type, len(dataset)))
                dataloader = DataLoader(dataset[type], batch_size=args.batch_size, shuffle=True)
                all_losses = []
                for model_e in model.ensembles[type].models:
                    losses = train(dataloader, model_e, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)
                    all_losses.append([losses])
                if plot:
                    fig, ax = plt.subplots()
                    ax.plot(np.array(all_losses).mean(axis=0).squeeze())
                    ax.set_title('Training Loss for %s' % type)
    elif args.goal_type == 'pick':
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            all_losses = []
            for model_e in model.models:
                losses = train(dataloader, model_e, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)
                all_losses.append([losses])
            if plot:
                fig, ax = plt.subplots()
                ax.plot(np.array(all_losses).mean(axis=0).squeeze())
                ax.set_title('Training Loss for %s' % type)
