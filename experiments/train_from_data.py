import torch
import numpy as np
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from pprint import pformat
import matplotlib.pyplot as plt

from learning.models.mlp import MLP
from learning.models.ensemble import Ensembles
from learning.utils import train_move_contact
from domains.utils import init_world
from domains.tools.world import ToolsWorld


def train_from_data(args, logger, start_i):
    # get model params
    n_mc_in = ToolsWorld.get_model_params()
    base_args = {'n_in': n_mc_in,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    largest_dataset, max_actions = logger.load_trans_dataset('', ret_i=True)

    for i in range(0, max_actions+1, args.train_freq):
        if i > start_i:
            dataset = logger.load_trans_dataset('', i=i)
            if len(dataset) > 0:
                ensembles = Ensembles(MLP, base_args, n_models, contact_types)
                for type in contact_types:
                    print('Training %s ensemble.' % type)
                    dataloader = DataLoader(dataset[type], batch_size=batch_size, shuffle=True)
                    for model in ensembles.ensembles[type].models:
                        train(dataloader, model, n_epochs=n_epochs, loss_fn=F.binary_cross_entropy)
                print('Training model from |dataset| = %i' % len(dataset))

            # save model and accuracy plots
            logger.save_trans_model(models, i=i)


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

    # Training args
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=300,
                        help='training epochs')
    parser.add_argument('--n-hidden',
                        type=int,
                        default=32,
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
