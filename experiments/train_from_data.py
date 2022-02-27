import numpy as np
import argparse
import matplotlib.pyplot as plt

from domains.tools.world import ToolsWorld, CONTACT_TYPES
from experiments.utils import ExperimentLogger
from learning.utils import initialize_model, train_model


def train_step(args, base_args, i):
    dataset, i = logger.load_trans_dataset('', i=i, ret_i=True)
    print('Training for action step %i' % i)
    model = initialize_model(args, base_args, types=CONTACT_TYPES)
    train_model(model, dataset, args, types=CONTACT_TYPES)

    # save model and accuracy plots
    logger.save_trans_model(model, i=i)


def train_from_data(args, logger, start_i):
    # get model params
    base_args = {'n_in': MODEL_INPUT_DIMS[args.goal_type],
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    if args.single_train_step:
        train_step(args, base_args, None)
    else:
        largest_dataset, max_actions = logger.load_trans_dataset('', ret_i=True)
        for dataset, i in logger.get_dataset_iterator(''):
            n_dataset_actions = len(dataset)
            if not n_dataset_actions % args.train_freq:
                if i > start_i:
                    train_step(args, base_args, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data collection args
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--dataset-exp-paths',
                        type=str,
                        nargs='+',
                        help='paths to datasets that need to be trained')
    parser.add_argument('--train-freq',
                        type=int,
                        default=10,
                        help='number of actions between model training')
    parser.add_argument('--single-train-step',
                        action='store_true',
                        help='use when just want to train a single model from the last dataset step')

    # Training args
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='training batch size')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=300,
                        help='training epochs')
    parser.add_argument('--n-hidden',
                        type=int,
                        default=48,
                        help='number of hidden units in network')
    parser.add_argument('--n-layers',
                        type=int,
                        default=5,
                        help='number of layers in GNN node and edge networks')
    parser.add_argument('--n-models',
                        type=int,
                        default=1,
                        help='number of models in ensemble')

    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    for dataset_exp_path in args.dataset_exp_paths:
        print('Training models on path: ', dataset_exp_path)
        logger = ExperimentLogger(dataset_exp_path)

        # check if models already in logger
        _, indices = logger.get_dir_indices('models')
        models_exist = len(indices) > 0
        if models_exist:
            print('Adding to models already in logger')
        else:
            logger.add_model_args(args)

        start_i = 0 if not models_exist else max(indices)
        train_from_data(args, logger, start_i)
