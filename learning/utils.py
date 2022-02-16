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

    output = model.forward(inputs)
    if torch.cuda.is_available():
        output = output.cpu()
    return output.detach().numpy()


def add_trajectory_to_dataset(domain, dataset_logger, trajectory, world):
    if len(trajectory) > 0:
        print('Adding trajectory to dataset.')
        dataset, n_curr_actions = dataset_logger.load_trans_dataset('curr', ret_i=True)
        if n_curr_actions != 0:
            dataset_logger.remove_dataset('curr', i=n_curr_actions)
        n_curr_actions += len(trajectory)
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
                dataset_logger.save_trans_dataset(dataset, 'curr', n_curr_actions)


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


def split_and_move_data(logger, val_ratio):
    train_dataset, n_actions = logger.load_trans_dataset('train', ret_i=True)
    val_dataset = logger.load_trans_dataset('val')
    new_dataset, n_new_actions = logger.load_trans_dataset('curr', ret_i=True)
    val_len = round(len(new_dataset)*val_ratio)
    assert val_len >= 1, \
            'Not enough data in current dataset to split into val and train datasets. ' \
            'Try making train freq larger.'
    train_dataset.merge_datasets(new_dataset, [0, val_len])
    val_dataset.merge_datasets(new_dataset, [val_len, len(new_dataset)])
    logger.save_trans_dataset(train_dataset, 'train', i=n_actions+n_new_actions)
    logger.save_trans_dataset(val_dataset, 'val', i=n_actions+n_new_actions)
    logger.remove_dataset('curr', i=n_new_actions)
    return train_dataset, val_dataset
