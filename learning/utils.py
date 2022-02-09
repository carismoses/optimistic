from torch import nn
import torch

import pb_robot

def model_forward(model, inputs, single_batch=False):
    if single_batch:
        single_inputs = inputs
        inputs = [torch.tensor(input[None, :], dtype=torch.float64) \
                                            for input in single_inputs]
    if torch.cuda.is_available():
        model.cuda()
        inputs = [inpi.cuda() for inpi in inputs]

    if len(inputs) == 1:
        inputs = inputs[0]
    output = model.forward(inputs)
    if torch.cuda.is_available():
        output = output.cpu()
    return output.detach().numpy()


def add_goal_to_dataset(dataset_logger, trajectory, goal_vec):
    if len(trajectory) > 0: print('Adding goal to dataset.')
    goal_dataset, n_actions = dataset_logger.load_dataset('goal', ret_i=True)
    last_i = len(trajectory)-1
    for i in range(len(trajectory)):
        if i == last_i:
            goal_dataset.add_to_dataset(goal_vec, float(trajectory[-1][-1]))
        n_actions += 1
        dataset_logger.save_dataset(goal_dataset, 'goal', i=n_actions)


def add_trajectory_to_dataset(domain, dataset_logger, trajectory, world):
    if len(trajectory) > 0: print('Adding trajectory to dataset.')
    dataset, n_actions = dataset_logger.load_dataset('trans', ret_i=True)
    for (state, pddl_action, next_state, opt_accuracy) in trajectory:
        if (pddl_action.name == 'move_contact' and domain == 'tools') or \
            (pddl_action.name in ['place', 'pickplace'] and domain == 'ordered_blocks'):
            object_features, edge_features = world.state_to_vec(state)
            action_features = world.action_to_vec(pddl_action)
            # assume object features don't change for now
            _, next_edge_features = world.state_to_vec(next_state)
            delta_edge_features = next_edge_features-edge_features
            dataset.add_to_dataset(object_features,
                                    edge_features,
                                    action_features,
                                    next_edge_features,
                                    delta_edge_features,
                                    opt_accuracy)
        n_actions += 1
        dataset_logger.save_dataset(dataset, 'trans', i=n_actions)
    return n_actions


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
