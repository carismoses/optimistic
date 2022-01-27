import torch
import numpy as np
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from pprint import pformat
import matplotlib.pyplot as plt
import os
import pickle
import subprocess

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger, add_trajectory_to_dataset
from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble, OptimisticEnsemble
from learning.train import train
from domains.utils import init_world
from domains.tools.world import ToolsWorld
from experiments.strategies import collect_trajectory


def run_curric(args, logger):
    # get model params
    n_of_in, n_ef_in, n_af_in = ToolsWorld.get_model_params()
    base_args = {'n_of_in': n_of_in,
                'n_ef_in': n_ef_in,
                'n_af_in': n_af_in,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    # save initial fully optimistic model
    ensemble = Ensemble(TransitionGNN, base_args, 1)
    logger.save_trans_model(ensemble, i=0)
    dataset = TransDataset()
    logger.save_trans_dataset(dataset, i=0)

    n_actions = 0

    while n_actions < args.max_actions:
        print('|dataset| = %i' % n_actions)

        # collect data in world with learned model with progress goal
        pddl_model_type = 'optimistic'
        world = init_world(args.domain,
                            args.domain_args,
                            pddl_model_type,
                            args.vis,
                            logger)

        world.change_goal_space(args.progress)
        trajectory = collect_trajectory(world, logger, 'random-goals-opt')

        # if trajectory returned, add to dataset
        if trajectory:
            print('Successfully collected trajectory.')
            if all([t_seg[3] for t_seg in trajectory]):
                print('All feasible actions.')
            else:
                print('Infeasible action attempted.')
            add_trajectory_to_dataset(args.domain, dataset, trajectory, world)
            n_actions += 1
            logger.save_trans_dataset(dataset, i=n_actions)

            # if at training freq, train model
            if not n_actions % args.train_freq:
                print('Training ensemble.')
                # want to train first model from random initialization, not optimistic
                # model (it doesn't actually have any parameters)
                ensemble = Ensemble(TransitionGNN, base_args, 1)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                for model in ensemble.models:
                    train(dataloader, model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)
                # save model and accuracy plots
                logger.save_trans_model(ensemble, i=n_actions)
                print('Saved dataset, model, and accuracy plot to %s' % logger.exp_path)
        else:
            print('Trajectory collection failed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data collection args
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--progress',
                        type=float)
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks', 'tools'],
                        default='tools',
                        help='domain to generate data from')
    parser.add_argument('--domain-args',
                        nargs='+',
                        help='arguments to pass into desired domain')
    parser.add_argument('--max-actions',
                        type=int,
                        default=400,
                        help='max number of actions for the robot to attempt')
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--train-freq',
                        type=int,
                        default=10,
                        help='number of actions between model training')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--n-ensembles',
                        type=int,
                        default=5,
                        help='number of ensembles to train')

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

    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    paths = []
    for _ in range(args.n_ensembles):
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        run_curric(args, logger)
        paths.append(logger.exp_path)
    print('Runs saved to:')
    for path in paths:
        print(path)
