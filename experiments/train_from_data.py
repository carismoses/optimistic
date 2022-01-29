import torch
import numpy as np
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from pprint import pformat
import matplotlib.pyplot as plt

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger, add_trajectory_to_dataset
from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble, OptimisticEnsemble
from learning.train import train
from domains.utils import init_world
from domains.tools.world import ToolsWorld
from experiments.strategies import collect_trajectory_wrapper


def train_from_data(args, logger):
    # get model params
    n_of_in, n_ef_in, n_af_in = ToolsWorld.get_model_params()
    base_args = {'n_of_in': n_of_in,
                'n_ef_in': n_ef_in,
                'n_af_in': n_af_in,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    largest_dataset = logger.load_trans_dataset()
    max_actions = len(largest_dataset)
    for i in range(0, max_actions+1, args.train_freq):
        if i > 0:
            ensemble = Ensemble(TransitionGNN, base_args, args.n_models)
            dataset = logger.load_trans_dataset(i=i)
            print('Training model from |dataset| = %i' % len(dataset))
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            for model in ensemble.models:
                train(dataloader, model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

            # save model and accuracy plots
            logger.save_trans_model(ensemble, i=i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data collection args
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--dataset-exp-path',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
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

    logger = ExperimentLogger(args.dataset_exp_path, add_args=args)
    train_from_data(args, logger)
