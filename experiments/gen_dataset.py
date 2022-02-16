import argparse

from learning.datasets import TransDataset
from experiments.utils import ExperimentLogger
from tamp.utils import execute_plan
from domains.utils import init_world
from experiments.strategies import collect_trajectory_wrapper
import matplotlib.pyplot as plt

def gen_dataset(args, n_actions, dataset_logger, model_logger):
    dataset = dataset_logger.load_trans_dataset('')
    while len(dataset) < args.max_dataset_size:
        print('|dataset| = %i' % len(dataset))
        if model_logger is None:
            pddl_model_type = 'optimistic'
        else:
            pddl_model_type = 'learned'
        trajectory = collect_trajectory_wrapper(args,
                                                pddl_model_type,
                                                dataset_logger,
                                                args.goal_progress,
                                                separate_process=not args.single_process,
                                                model_logger=model_logger,
                                                save_to_dataset=True)
        n_actions += len(trajectory)
        if args.balanced:
            # balance dataset by removing added element if makes it unbalanced
            num_per_class = args.max_dataset_size // 2
            num_pos_datapoints = sum([y for x,y in dataset])
            num_neg_datapoints = len(dataset) - num_pos_datapoints
            if num_pos_datapoints > num_per_class or num_neg_datapoints > num_per_class:
                print('Removing last trajectory added to dataset.')
                for _ in range(len(trajectory)):
                    dataset_logger.remove_dataset(i=n_actions)
                    n_actions -= 1

        dataset = dataset_logger.load_trans_dataset()

        # optionally replay with pyBullet
        '''
        if args.vis_performance:
            answer = input('Replay with pyBullet (r) or not (ENTER)?')
            plt.close()
            if answer == 'r':
                # make new world to visualize plan execution
                world.disconnect()
                vis = True
                world = init_world('tools',
                                    None,
                                    'optimistic',
                                    vis,
                                    dataset_logger)
                trajectory = execute_plan(world, *plan_data)
                world.disconnect()
        '''
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
    parser.add_argument('--max-dataset-size',
                        type=int,
                        default=400,
                        help='max number of actions for the robot to attempt')
    parser.add_argument('--balanced',
                        type=str,
                        default='False',
                        choices=['False', 'True'],
                        help='use if want balanced feasible/infeasible dataset')
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--goal-progress',
                        type=float,
                        help='what fraction of the maximum goal region is acceptable')
    parser.add_argument('--vis-performance',
                        action='store_true',
                        help='use to visualize success/failure of robot executions and optionally replay with pyBullet.')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--n-seq-plans',
                        type=int,
                        default=100,
                        help='number of plans used to generate search space for sequential methods')
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks', 'tools'],
                        default='tools',
                        help='domain to generate data from')
    parser.add_argument('--domain-args',
                        nargs='+',
                        help='arguments to pass into desired domain')
    # random-actions: sample random rollouts
    # random-goals-opt: plan to achieve random goals with the optimistic model
    # random-goals-learned: plan to achieve random goals from a learned model
    parser.add_argument('--data-collection-mode',
                        type=str,
                        default='random-goals-opt',
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned'],
                        help='method of data collection')
    parser.add_argument('--n-datasets',
                        type=int,
                        default=1,
                        help='number of datasets to generate')
    parser.add_argument('--single-process',
                        action='store_true')
    # for now this assumes that you always want to use the most trained model on the path for planning
    # (as opposed to a different i) -- only needed for data-collection-mode == 'random-goals-learned'
    parser.add_argument('--model-paths',
                        type=str,
                        nargs='+',
                        help='list of model paths to use for planning')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.restart:
        assert args.exp_path, 'Must set the --exp-path to restart experiment'
        dataset_logger = ExperimentLogger(args.exp_path)
        n_actions = dataset_logger.get_action_count()
        dataset_args = dataset_logger.args
        if args.max_dataset_size > dataset_args.max_dataset_size:
            print('Adding %i to previous max dataset size' % (args.max_dataset_size - dataset_args.max_dataset_size))
            dataset_args.max_dataset_size = args.max_dataset_size

        model_logger = ExperimentLogger(dataset_args.data_model_path) \
                            if 'data_model_path' in vars(dataset_args) else None
        gen_dataset(dataset_args, n_actions, dataset_logger, model_logger)
        print('Finished dataset path: %s' % args.exp_path)
    else:
        assert args.exp_name, 'Must set the --exp-name arg to start new run'
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
