from torch import nn
import torch
from torch.utils.data import random_split, ConcatDataset

import pb_robot

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
            if (pddl_action.name == 'move_contact' and domain == 'tools') or \
                (pddl_action.name in ['place', 'pickplace'] and domain == 'ordered_blocks'):
                x = world.state_and_action_to_vec(state, pddl_action)
                contact_type = pddl_action.args[5].type
                dataset[contact_type].add_to_dataset(x, opt_accuracy)
                dataset_logger.save_trans_dataset(dataset, '', n_actions)
                datapoints.append((pddl_action.name, contact_type, x, opt_accuracy))
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
