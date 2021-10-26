import torch
import numpy as np
import argparse
from argparse import Namespace
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from pprint import pformat

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from learning.train import train
from tamp.utils import execute_random, execute_plan
from domains.ordered_blocks.world import OrderedBlocksWorld


def train_class(args, trans_dataset, logger):
    pddl_model_type = 'learned' if 'learned' in args.data_collection_mode else 'optimistic'

    # save initial (empty) dataset
    logger.save_trans_dataset(trans_dataset, i=0)

    # initialize and save model
    trans_model = TransitionGNN(n_of_in=1,
                                n_ef_in=1,
                                n_af_in=2,
                                n_hidden=args.n_hidden,
                                pred_type=args.pred_type)
    logger.save_trans_model(trans_model, i=0)

    i = 0
    while len(trans_dataset) < args.max_transitions:
        print('Iteration %i |dataset| = %i' % (i, len(trans_dataset)))
        if args.domain == 'ordered_blocks':
            world, opt_pddl_info, pddl_info = OrderedBlocksWorld.init(args.domain_args,
                                                                    pddl_model_type,
                                                                    args.vis,
                                                                    logger)
        else:
            raise NotImplementedError
        goal = world.generate_random_goal() # ignored if execute_random()
        print('Init: ', world.init_state)
        print('Goal: ', goal)
        if 'goals' in args.data_collection_mode:
            if world.use_panda:
                world.panda.add_text('Planning')
            # generate plan (using PDDLStream) to reach random goal
            problem = tuple([*pddl_info, world.init_state, goal])
            ic = 2 if world.use_panda else 0
            pddl_plan, cost, init_expanded = solve_focused(problem,
                                                success_cost=INF,
                                                max_skeletons=2,
                                                search_sample_ratio=1.0,
                                                max_time=INF,
                                                verbose=False,
                                                unit_costs=True,
                                                initial_complexity=ic)  # don't set complexity=2 in simple (non-robot) domain
            print('Plan: ', pddl_plan)
            if pddl_plan is not None and len(pddl_plan) > 0:
                if world.use_panda:
                    world.panda.add_text('Executing plan')
                trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
            else:
                # if plan not found, execute random actions
                if world.use_panda:
                    world.panda.add_text('Planning failed. Executing random action')
                trajectory = execute_random(world, opt_pddl_info)
        elif args.data_collection_mode == 'random-actions':
            if world.use_panda:
                world.panda.add_text('Executing random actions')
            trajectory = execute_random(world, opt_pddl_info)

        # disconnect from world
        world.disconnect()

        # add to dataset and save
        print('Adding trajectory to dataset.')
        add_trajectory_to_dataset(args, trans_dataset, trajectory, world)

        # initialize and train new model
        trans_model = TransitionGNN(n_of_in=1,
                                    n_ef_in=1,
                                    n_af_in=2,
                                    n_hidden=args.n_hidden,
                                    pred_type=args.pred_type)
        trans_dataset.set_pred_type(args.pred_type)
        print('Training model.')
        trans_dataloader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=True)
        train(trans_dataloader, trans_model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

        # save new model and dataset
        i += 1
        logger.save_trans_dataset(trans_dataset, i=i)
        logger.save_trans_model(trans_model, i=i)
        print('Saved model to %s' % logger.exp_path)


def add_trajectory_to_dataset(args, trans_dataset, trajectory, world):
    for (state, pddl_action, next_state, opt_accuracy) in trajectory:
        if 'place' in pddl_action.name:
            object_features, edge_features = world.state_to_vec(state)
            action_features = world.action_to_vec(pddl_action)
            # assume object features don't change for now
            _, next_edge_features = world.state_to_vec(next_state)
            delta_edge_features = next_edge_features-edge_features
            trans_dataset.add_to_dataset(object_features,
                                            edge_features,
                                            action_features,
                                            next_edge_features,
                                            delta_edge_features,
                                            opt_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data collection args
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks'],
                        default='ordered_blocks',
                        help='domain to generate data from')
    parser.add_argument('--domain-args',
                        nargs='+',
                        help='arguments to pass into desired domain')
    parser.add_argument('--max-transitions',
                        type=int,
                        default=100,
                        help='max number of transitions to save to transition dataset')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save datasets and models to')
    parser.add_argument('--data-collection-mode',
                        type=str,
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned'],
                        required=True,
                        help='method of data collection')
    parser.add_argument('--N',
                        type=int,
                        default=1,
                        help='number of data collection/training runs to perform')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    # Training args
    parser.add_argument('--pred-type',
                        type=str,
                        choices=['delta_state', 'full_state', 'class'],
                        default='class')
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
                        default=16,
                        help='number of hidden units in network')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if not args.domain == 'ordered_blocks':
        NotImplementedError('Only Ordered Blocks world works')

    if not args.pred_type == 'class':
        NotImplementedError('Prediction types != class have not been tested in a while')

    paths = []
    for n in range(args.N):
        print('Run %i/%i' % (n+1, args.N))
        trans_dataset = TransDataset()
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        train_class(args, trans_dataset, logger)
        paths.append(logger.exp_path)

    # print out all paths
    if len(paths) > 1:
        print('%i runs saved to :' % args.N)
        print('['+pformat(paths[0])+',')
        [print(pformat(path)+',') for path in paths[1:-1]]
        print(pformat(paths[-1])+']')
    else:
        print('%i run saved to :' % args.N)
        print('['+pformat(paths[0])+']')
