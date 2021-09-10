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
from pddlstream.algorithms.downward import fact_from_fd, is_applicable

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from learning.train import train
from tamp.utils import postprocess_plan, task_from_problem, get_fd_action
from domains.ordered_blocks.world import OrderedBlocksWorld

# TODO: this doesn't make len(dataset) == args.max_transitions exactly
# because trajectories are added in chunks that might push it past the limit
# but will be close

def train_class(args, trans_dataset, logger):
    if args.domain == 'ordered_blocks':
        world = OrderedBlocksWorld(args.domain_args)
        if args.data_collection_mode in ['random-goals-opt', 'random-actions']:
            pddl_info = world.get_pddl_info('optimistic')
        elif args.data_collection_mode == 'random-goals-learned':
            pddl_info = world.get_pddl_info('learned', logger)
    else:
        raise NotImplementedError

    world.panda.step_simulation()

    init_state = world.get_init_state()
    # NOTE: the goal is ignored if execute_random is called
    goal = world.generate_random_goal()
    print('Goal: ', goal)
    problem = tuple([*pddl_info, init_state, goal])

    # save initial (empty) dataset
    logger.save_trans_dataset(trans_dataset, i=0)

    # initialize and save model
    trans_model = TransitionGNN(n_of_in=1,
                                n_ef_in=1,
                                n_af_in=2,
                                n_hidden=16,
                                pred_type=args.pred_type)
    logger.save_trans_model(trans_model, i=0)

    i = 0
    while len(trans_dataset) < args.max_transitions:
        print('Iteration %i |dataset| = %i' % (i, len(trans_dataset)))
        if 'random-goals' in args.data_collection_mode:
            # generate plan (using PDDLStream) to reach random goal
            pddl_plan, cost, _ = solve_focused(problem,
                                                success_cost=INF,
                                                max_skeletons=2,
                                                search_sample_ratio=1.0,
                                                max_time=INF,
                                                verbose=False,
                                                unit_costs=True)
            print('Plan: ', pddl_plan)
            if pddl_plan is not None:
                trajectory = execute_plan(world, problem, pddl_plan)
            else:
                # if plan not found, execute random actions
                trajectory = execute_random(world, problem)
        elif args.data_collection_mode == 'random-actions':
            trajectory = execute_random(world, problem)

        # add to dataset and save
        print('Adding trajectory to dataset.')
        add_trajectory_to_dataset(args, trans_dataset, trajectory, world)

        # initialize and train new model
        if args.train:
            trans_model = TransitionGNN(n_of_in=1,
                                        n_ef_in=1,
                                        n_af_in=2,
                                        n_hidden=16,
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

def execute_plan(world, problem, pddl_plan):
    task, fd_plan = postprocess_plan(problem, pddl_plan)
    fd_state = set(task.init)
    trajectory = []
    for (fd_action, pddl_action) in zip(fd_plan, pddl_plan):
        assert is_applicable(fd_state, fd_action), 'Something wrong with planner. An invalid action is in the plan.'
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_pddl_state, new_fd_state, valid_transition = world.transition(pddl_state, fd_state, pddl_action, fd_action)
        trajectory.append((pddl_state, pddl_action, new_pddl_state, valid_transition))
        fd_state = new_fd_state
    return trajectory

def execute_random(world, problem):
    task = task_from_problem(problem)
    fd_state = set(task.init)
    pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
    trajectory = []
    valid_actions = True
    while valid_actions:
        pddl_action = world.random_action(pddl_state)
        fd_action = get_fd_action(task, pddl_action)
        new_pddl_state, new_fd_state, valid_transition = world.transition(pddl_state, fd_state, pddl_action, fd_action)
        trajectory.append((pddl_state, pddl_action, new_pddl_state, valid_transition))
        fd_state = new_fd_state
        pddl_state = new_pddl_state
        valid_actions = world.valid_actions_exist(pddl_state)
    return trajectory

def add_trajectory_to_dataset(args, trans_dataset, trajectory, world):
    for (state, pddl_action, next_state, opt_accuracy) in trajectory:
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
                        help='set to run in debug mode')
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
    parser.add_argument('--train',
                        type=bool,
                        default=True,
                        help='Sometimes just want to collect dataset, dont want to train')
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
