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
from learning.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble
from learning.train import train
#from domains.utils import init_world
from domains.tools.world import ToolsWorld
from experiments.strategies import collect_trajectory

def train_class(args, logger, n_actions, last_train_count):
    pddl_model_type = 'learned' if 'learned' in args.data_collection_mode else 'optimistic'

    # get model params
    n_of_in, n_ef_in, n_af_in = ToolsWorld.get_model_params()
    base_args = {'n_of_in': n_of_in,
                'n_ef_in': n_ef_in,
                'n_af_in': n_af_in,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    # save initial (empty) dataset
    if n_actions == 0:
        trans_dataset = TransDataset()
        logger.save_trans_dataset(trans_dataset, i=0)
        ensemble = Ensemble(TransitionGNN,
                                base_args,
                                args.n_models)
        logger.save_trans_model(ensemble, i=0)
    else:
        trans_dataset = logger.load_trans_dataset(n_actions)

    while n_actions < args.max_actions:
        print('|dataset| = %i, # actions = %i' % (len(trans_dataset), n_actions))
        #world = init_world(args.domain,
        #                    args.domain_args,
        #                    pddl_model_type,
        #                    args.vis,
        #                    logger)
        #if world.use_panda:
        #    world.panda.add_text('|dataset| = %i, # actions = %i' % (len(trans_dataset), n_actions),
        #                        position=(0, -1.15, 1.1),
        #                        size=1,
        #                        counter=True)
        progress = n_actions/args.max_actions

        # write solver args to file (remove if one is there)
        tmp_dir = 'temp'
        os.makedirs(tmp_dir, exist_ok=True)
        in_pkl = '%s/solver_args.pkl' % tmp_dir
        out_pkl = '%s/solver_solution.pkl' % tmp_dir
        if os.path.exists(in_pkl):
            os.remove(in_pkl)
        with open(in_pkl, 'wb') as handle:
            pickle.dump([args, pddl_model_type, logger, progress, n_actions], handle)

        # call planner with pickle file
        print('Collecting trajectory.')
        proc = subprocess.run(["python3", "-m", "experiments.strategies", \
                            "--in-pkl", in_pkl, \
                            "--out-pkl", out_pkl], stdout=subprocess.PIPE)

        # read results from pickle file
        with open(out_pkl, 'rb') as handle:
            trajectory, n_actions = pickle.load(handle)

        if not trajectory:
            print('Trajectory collection failed.')
        else:
            print('Successfully collected trajectory.')
        #if len(trans_dataset) > 0:
        #    world.plot_datapoint(len(trans_dataset)-1)

        # check that at training step and there is data in the dataset
        trans_dataset = logger.load_trans_dataset()
        if (n_actions-last_train_count) > args.train_freq and len(trans_dataset) > 0:
            last_train_count = n_actions

            # initialize and train new model
            ensemble = Ensemble(TransitionGNN,
                                    base_args,
                                    args.n_models)
            print('Training ensemble.')
            trans_dataloader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=True)
            for model in ensemble.models:
                train(trans_dataloader, model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

            # save model and accuracy plots
            #world.plot_model_accuracy(n_actions, ensemble)
            logger.save_trans_model(ensemble, i=n_actions)
            print('Saved dataset, model, and accuracy plot to %s' % logger.exp_path)



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
                        default='ordered_blocks',
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
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned', \
                                'sequential-plans', 'sequential-goals', 'engineered-goals-dist', \
                                'engineered-goals-size'],
                        help='method of data collection')
    parser.add_argument('--train-freq',
                        type=int,
                        default=10,
                        help='number of actions between model training')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--n-seq-plans',
                        type=int,
                        default=100,
                        help='number of plans used to generate search space for sequential methods')
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
        if not args.exp_path:
            assert 'Must set the --exp-path to restart experiment'
        logger = ExperimentLogger(args.exp_path)
        n_actions, last_train_count = logger.get_action_count()
        args = logger.args
    else:
        if not args.exp_name:
            assert 'Must set the --exp-name to start a new experiments'
        if not args.data_collection_mode:
            assert 'Must set the --data-collection-mode when starting a new experiment'
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        n_actions, last_train_count = 0, 0

    train_class(args, logger, n_actions, last_train_count)
    print('Run saved to %s' % logger.exp_path)
