import torch
import numpy as np
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from pprint import pformat
import matplotlib.pyplot as plt

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from learning.train import train
from tamp.utils import execute_random, execute_plan
from domains.utils import init_world

def train_class(args, trans_dataset, logger):
    pddl_model_type = 'learned' if 'learned' in args.data_collection_mode else 'optimistic'

    world, opt_pddl_info, pddl_info = init_world(args.domain,
                                                    args.domain_args,
                                                    pddl_model_type,
                                                    args.vis,
                                                    logger)

    # save initial (empty) dataset
    logger.save_trans_dataset(trans_dataset, i=0)

    # initialize and save model
    trans_model = TransitionGNN(n_of_in=world.n_of_in,
                                n_ef_in=world.n_ef_in,
                                n_af_in=world.n_af_in,
                                n_hidden=args.n_hidden,
                                n_layers=args.n_layers)
    logger.save_trans_model(trans_model, i=0)

    # NOTE: just made a world to get model params
    world.disconnect()
    n_actions = 0
    last_train_count = 0
    while n_actions < args.max_actions:
        i = len(trans_dataset)
        print('|dataset| = %i, # actions = %i' % (i, n_actions))
        world, opt_pddl_info, pddl_info = init_world(args.domain,
                                                        args.domain_args,
                                                        pddl_model_type,
                                                        args.vis,
                                                        logger)
        goal = world.generate_random_goal() # ignored if execute_random()
        print('Init: ', world.init_state)
        if world.use_panda:
            world.panda.add_text('|dataset| = %i, # actions = %i' % (i, n_actions),
                                position=(0, -1.15, 1.1),
                                size=1,
                                counter=True)
        pddl_plan = None
        if 'goals' in args.data_collection_mode:
            pddl_plan, problem, init_expanded = plan_wrapper(goal, world, pddl_model_type, pddl_info)

        if not pddl_plan and args.data_collection_mode == 'random-goals-learned':
            # if plan not found for learned model use optimistic model
            pddl_plan, problem, init_expanded = plan_wrapper(goal, world, 'optimistic', opt_pddl_info)

        if pddl_plan:
            trajectory = execute_plan_wrapper(world, problem, pddl_plan, init_expanded)
        else:
            # execute random actions
            print('Planning random actions.')
            if world.use_panda:
                world.panda.add_text('Planning random actions',
                                    position=(0, -1, 1),
                                    size=1.5)
            trajectory = execute_random(world, opt_pddl_info)
        n_actions += len(trajectory)

        # add to dataset and save
        if trajectory:
            print('Adding trajectory to dataset.')
            add_trajectory_to_dataset(args, trans_dataset, trajectory, world)

        # check that at training step and there is data in the dataset
        if (n_actions-last_train_count) > args.train_freq and len(trans_dataset) > 0:
            last_train_count = n_actions
            # initialize and train new model
            trans_model = TransitionGNN(n_of_in=world.n_of_in,
                                        n_ef_in=world.n_ef_in,
                                        n_af_in=world.n_af_in,
                                        n_hidden=args.n_hidden,
                                        n_layers=args.n_layers)
            print('Training model.')
            trans_dataloader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=True)
            train(trans_dataloader, trans_model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

            # save dataset, model, and accuracy plots
            logger.save_trans_dataset(trans_dataset, i=n_actions)
            world.plot_model_accuracy(n_actions, trans_model, logger)
            logger.save_trans_model(trans_model, i=n_actions)
            print('Saved dataset, model, and accuracy plot to %s' % logger.exp_path)

        # disconnect from world
        world.disconnect()


def plan_wrapper(goal, world, pddl_model_type, pddl_info):
    print('Goal: ', goal)
    print('Planning with %s model'%pddl_model_type)
    if world.use_panda:
        world.panda.add_text('Planning with %s model'%pddl_model_type,
                            position=(0, -1, 1),
                            size=1.5)

    # generate plan (using PDDLStream) to reach random goal
    problem = tuple([*pddl_info, world.init_state, goal])
    ic = 2 if world.use_panda else 0
    pddl_plan, cost, init_expanded = solve_focused(problem,
                                        success_cost=INF,
                                        max_skeletons=2,
                                        search_sample_ratio=1.0,
                                        max_time=120,
                                        verbose=False,
                                        unit_costs=True,
                                        initial_complexity=ic,
                                        max_iterations=2)
    print('Plan: ', pddl_plan)

    if not pddl_plan and world.use_panda:
        world.panda.add_text('Planning failed.',
                            position=(0, -1, 1),
                            size=1.5)
        time.sleep(.5)

    return pddl_plan, problem, init_expanded


def execute_plan_wrapper(world, problem, pddl_plan, init_expanded):
    if world.use_panda:
        world.panda.add_text('Executing found plan',
                            position=(0, -1, 1),
                            size=1.5)
    trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
    return trajectory


def add_trajectory_to_dataset(args, trans_dataset, trajectory, world):
    for (state, pddl_action, next_state, opt_accuracy) in trajectory:
        if (pddl_action.name == 'move_contact' and args.domain == 'tools') or \
            (pddl_action.name in ['place', 'pickplace'] and args.domain == 'ordered_blocks'):
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
                        choices=['ordered_blocks', 'tools'],
                        default='ordered_blocks',
                        help='domain to generate data from')
    parser.add_argument('--domain-args',
                        nargs='+',
                        help='arguments to pass into desired domain')
    parser.add_argument('--max-actions',
                        type=int,
                        default=100,
                        help='max number of actions for the robot to attempt')
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
    parser.add_argument('--train-freq',
                        type=int,
                        default=1,
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
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

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
