import argparse
import os
import time

from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt

from experiments.utils import ExperimentLogger
from tamp.utils import execute_plan
from experiments.strategies import collect_trajectory_wrapper
from learning.domains import OptDictDataset


expert_feasible_goals = []

# first try to get through expert goals (should be feasible)
def gen_dataset(args, n_actions, dataset_logger, model_logger):
    if n_actions == 0:
        dataset = OptDictDataset(args.actions, args.objects)
        dataset_logger.save_trans_dataset(dataset, '', i=n_actions)
    else:
        dataset = dataset.logger.load_trans_dataset('')

    if args.goal_type == 'push':
        types = args.contact_types
    elif args.goal_type == 'pick':
        types = ['pick']

    if args.balanced:
        condition = len(dataset) < len(types)*args.max_type_size
    else:
        condition = n_actions < args.max_actions
    while condition:
        print('# actions = %i, |dataset| = %i' % (n_actions, len(dataset)))
        if model_logger is None:
            pddl_model_type = 'optimistic'
        else:
            pddl_model_type = 'learned'

        trajectory = collect_trajectory_wrapper(args,
                                                pddl_model_type,
                                                dataset_logger,
                                                separate_process=not args.single_process,
                                                model_logger=model_logger,
                                                save_to_dataset=True)
        if len(trajectory) > 0:
            n_actions += len(trajectory)
            dataset = dataset_logger.load_trans_dataset('')
            if args.balanced:
                # balance dataset by removing added element if makes it unbalanced
                for type in types:
                    if args.goal_type == 'push':
                        opt_dataset = dataset[type]
                    else:
                        opt_dataset = dataset
                    num_pos_datapoints = sum([y for x,y in opt_dataset])
                    num_neg_datapoints = len(opt_dataset) - num_pos_datapoints
                    print('Positive %s: %i' % (type, num_pos_datapoints))
                    print('Negative %s: %i' % (type, num_neg_datapoints))
                    if num_pos_datapoints > 0 or num_neg_datapoints > args.max_type_size:
                        print('Removing last trajectory added to dataset.')
                        dataset_logger.remove_dataset('', i=n_actions)
                        n_actions -= len(trajectory)
                    dataset = dataset_logger.load_trans_dataset('')

        if args.balanced:
            condition = len(dataset) < len(types)*args.max_type_size
        else:
            condition = n_actions < args.max_actions

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

    # World args
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks', 'tools'],
                        default='tools',
                        help='domain to generate data from')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--contact-types',
                        type=str,
                        nargs='+',
                        default=['poke', 'push_pull'])
    parser.add_argument('--goal-obj',
                        required=True,
                        type=str,
                        choices=['yellow_block', 'blue_block'])
    parser.add_argument('--goal-type',
                        required=True,
                        type=str,
                        choices=['push', 'pick'])

    # Data collection args
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--max-type-size',
                        type=int,
                        help='max number of actions IN DATASET for each class in balanced case')
    parser.add_argument('--max-actions',
                        type=int,
                        help='max number of ALL actions total for unbalanced case')
    # random-actions: sample random rollouts
    # random-goals-opt: plan to achieve random goals with the optimistic model
    # random-goals-learned: plan to achieve random goals from a learned model
    # (DIFFERENT from the model being learned. set model_path with --model-paths)
    parser.add_argument('--data-collection-mode',
                        type=str,
                        default='random-goals-opt',
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned'],
                        help='method of data collection')
    parser.add_argument('--single-process',
                        action='store_true')
    parser.add_argument('--balanced',
                        type=str,
                        default='False',
                        choices=['False', 'True'],
                        help='use if want balanced feasible/infeasible dataset')
    parser.add_argument('--n-datasets',
                        type=int,
                        default=1,
                        help='number of datasets to generate')
    # for now this assumes that you always want to use the most trained model on the path for planning
    # (as opposed to a different i) -- only needed for data-collection-mode == 'random-goals-learned'
    parser.add_argument('--model-paths',
                        type=str,
                        nargs='+',
                        help='list of model paths to use for planning')

    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.restart:
        assert args.exp_path, 'Must set the --exp-path to restart experiment'
        dataset_logger = ExperimentLogger(args.exp_path)
        _, n_actions = dataset_logger.load_trans_dataset('', ret_i=True)
        dataset_args = dataset_logger.args
        model_logger = ExperimentLogger(dataset_args.data_model_path) \
                    if dataset_args.data_model_path else None
        gen_dataset(dataset_args, n_actions, dataset_logger, model_logger)
        print('Finished dataset path: %s' % args.exp_path)
    else:
        assert args.exp_name, 'Must set the --exp-name arg to start new run'
        if args.data_collection_mode == 'random-goals-learned':
            assert args.model_paths, 'Must pass in models to learn from in random-goals-learned mode'
        n_actions = 0
        args.balanced = args.balanced == 'True'
        if args.model_paths:
            if len(args.model_paths) > 1:
                assert len(args.model_paths) == args.n_datasets, 'If using multiple models to generate datasets \
                                then should generate the same number of datasets'

            if len(args.model_paths) > 0:
                assert 'learned' in args.data_collection_mode, 'Must use learned model if passing in model paths'
        else:
            args.model_paths = []

        dataset_paths = []
        for di in range(args.n_datasets):
            if len(args.model_paths) == 0:
                model_path = None
                model_logger = None
            elif len(args.model_paths) == 1:
                model_path = args.model_paths[0]
                model_logger = ExperimentLogger(model_path)
            else:
                model_path = args.model_paths[di]
                model_logger = ExperimentLogger(model_path)
            args.data_model_path = model_path
            dataset_logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
            gen_dataset(args, n_actions, dataset_logger, model_logger)
            dataset_paths.append(dataset_logger.exp_path)

        print('---Dataset paths---')
        print(dataset_paths)
