import argparse
import os
import time

from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt

from experiments.utils import ExperimentLogger
from tamp.utils import execute_plan
from domains.tools.world import ToolsWorld
from experiments.strategies import collect_trajectory_wrapper
from domains.tools.primitives import get_contact_gen
from domains.tools.world import CONTACT_TYPES

def remove_last_datapoint(dataset_logger, trajectory, n_actions):
    print('Removing last trajectory added to dataset.')
    dataset_logger.remove_dataset('', i=n_actions)
    n_actions -= len(trajectory)
    return n_actions

# first try to get through expert goals (should be feasible)
def gen_dataset(args, n_actions, dataset_logger, model_logger):
    dataset = dataset_logger.load_trans_dataset('')

    if args.goal_type == 'push':
        types = CONTACT_TYPES
    elif args.goal_type == 'pick':
        types = ['pick']

    while n_actions < len(types):
        print('# actions = %i, |dataset| = %i' % (n_actions, len(dataset)))
        pddl_model_type = 'optimistic'
        goal_progress = None
        trajectory = collect_trajectory_wrapper(args,
                                                pddl_model_type,
                                                dataset_logger,
                                                goal_progress,
                                                separate_process=not args.single_process,
                                                model_logger=model_logger,
                                                save_to_dataset=True)
        if len(trajectory) > 0:
            n_actions += len(trajectory)
            dataset = dataset_logger.load_trans_dataset('')
            for type in types:
                if args.goal_type == 'push':
                    opt_dataset = dataset[type]
                else:
                    opt_dataset = dataset
                if args.goal_type == 'push' and args.goal_obj == 'blue_block' and type == 'hook':
                    n_actions = remove_last_datapoint(dataset_logger, trajectory, n_actions)
                else:
                    num_pos_datapoints = sum([y for x,y in opt_dataset])
                    num_neg_datapoints = len(opt_dataset) - num_pos_datapoints
                    print('Positive %s: %i' % (type, num_pos_datapoints))
                    print('Negative %s: %i' % (type, num_neg_datapoints))
                    if num_neg_datapoints > 0:
                        n_actions = remove_last_datapoint(dataset_logger, trajectory, n_actions)
                    if num_pos_datapoints > 1:
                        n_actions = remove_last_datapoint(dataset_logger, trajectory, n_actions)
                dataset = dataset_logger.load_trans_dataset('')

    return dataset_logger

if __name__ == '__main__':
    # Data collection args
    parser = argparse.ArgumentParser()
    # if restarting an experiment, only need to set the following 2 arguments
    # (all other args will be taken from the initial run)
    # NOTE: you can only finish a single dataset at a time with this method
    parser.add_argument('--restart',
                        action='store_true',
                        help='use if want to restart from a crash (must also pass in exp-path)')
    parser.add_argument('--exp-path',
                        type=str,
                        help='the exp-path to restart from')


    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks', 'tools'],
                        default='tools',
                        help='domain to generate data from')
    parser.add_argument('--domain-args',
                        nargs='+',
                        help='arguments to pass into desired domain')
    parser.add_argument('--single-process',
                        action='store_true')
    parser.add_argument('--goal-obj',
                        required=True,
                        type=str,
                        choices=['yellow_block', 'blue_block'])
    parser.add_argument('--goal-type',
                        required=True,
                        type=str,
                        choices=['push', 'pick'])
    args = parser.parse_args()
    args.data_collection_mode = 'random-goals-opt'
    if args.debug:
        import pdb; pdb.set_trace()

    assert args.exp_name, 'Must set the --exp-name arg to start new run'
    n_actions = 0
    model_logger = None
    dataset_logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
    gen_dataset(args, n_actions, dataset_logger, model_logger)
