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
from experiments.utils import ExperimentLogger, add_trajectory_to_dataset
from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble, OptimisticEnsemble
from learning.train import train
from domains.utils import init_world
from domains.tools.world import ToolsWorld
from experiments.strategies import collect_trajectory


def run_curric(args, logger, n_actions):
    # get model params
    n_of_in, n_ef_in, n_af_in = ToolsWorld.get_model_params()
    base_args = {'n_of_in': n_of_in,
                'n_ef_in': n_ef_in,
                'n_af_in': n_af_in,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    # save initial fully optimistic model
    if n_actions == 0:
        ensemble = OptimisticEnsemble(TransitionGNN, base_args, args.n_models)
        logger.save_trans_model(ensemble, i=0)

    num_curric_levels = args.max_actions // args.actions_per_curric
    corr_max_actions = num_curric_levels * args.actions_per_curric

    while n_actions < corr_max_actions:
        print('|dataset| = %i' % n_actions)

        # calculate curriculum level
        curric_level = n_actions // args.actions_per_curric

        # get model to pass into world
        planning_model_i = curric_level*args.actions_per_curric

        # collect data in world with learned model with progress goal
        pddl_model_type = 'learned'
        world = init_world(args.domain,
                            args.domain_args,
                            pddl_model_type,
                            args.vis,
                            logger,
                            planning_model_i=planning_model_i)
        progress = curric_level / num_curric_levels
        world.change_goal_space(progress)
        trajectory = collect_trajectory(world, logger, args.data_collection_mode)

        # if trajectory returned, add to dataset
        if trajectory:
            print('Successfully collected trajectory.')
            if all([t_seg[3] for t_seg in trajectory]):
                print('All feasible actions.')
            else:
                print('Infeasible action attempted.')
            # start new dataset if starting new curriculum level else add to last dataset
            if not n_actions % args.actions_per_curric:
                dataset = TransDataset()
            else:
                dataset = logger.load_trans_dataset(i=n_actions)
            inital_len_dataset = len(dataset)
            add_trajectory_to_dataset(args.domain, dataset, trajectory, world)
            n_actions += len(dataset) - inital_len_dataset
            logger.save_trans_dataset(dataset, i=n_actions)


            # if at training freq, train model
            if not n_actions % args.train_freq:
                print('Training ensemble.')
                # want to train first model from random initialization, not optimistic
                # model (it doesn't actually have any parameters)
                if planning_model_i == 0:
                    ensemble = Ensemble(TransitionGNN, base_args, args.n_models)
                else:
                    ensemble = logger.load_trans_model(i=planning_model_i)
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
    # if restarting an experiment, only need to set the following 2 arguments
    # (all other args will be taken from the initial run)
    parser.add_argument('--restart',
                        action='store_true',
                        help='use if want to restart from a crash (must also pass in exp-path)')
    parser.add_argument('--exp-path',
                        type=str,
                        help='the exp-path to restart from')

    # Data collection args
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
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
    parser.add_argument('--data-collection-mode',
                        type=str,
                        choices=['curriculum'],
                        default='curriculum',
                        help='method of data collection')
    parser.add_argument('--actions-per-curric',
                        type=int,
                        default=10,
                        help='how many actions to execute for each curriculum level')
    parser.add_argument('--train-freq',
                        type=int,
                        default=10,
                        help='number of actions between model training')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')

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
                        default=5,
                        help='number of models in ensemble')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.restart:
        assert args.exp_path, 'Must set the --exp-path to restart experiment'
        logger = ExperimentLogger(args.exp_path)
        n_actions = logger.get_action_count()
        args = logger.args
    else:
        assert args.exp_name, 'Must set the --exp-name to start a new experiment'
        assert args.data_collection_mode, 'Must set the --data-collection-mode when starting a new experiment'
        condition = args.train_freq > args.actions_per_curric
        assert condition, 'Train frequency must be <= actions per curriculum'
        condition = args.actions_per_curric % args.train_freq
        assert condition, 'train freq must be a divisor of actions per curric'
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        n_actions = 0

    run_curric(args, logger, n_actions)
    print('Run saved to %s' % logger.exp_path)
