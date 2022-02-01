import argparse

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger, add_trajectory_to_dataset
from tamp.utils import execute_plan
from domains.utils import init_world
from experiments.strategies import collect_trajectory_wrapper
import matplotlib.pyplot as plt

def gen_dataset(args, model_path):
    #plt.ion()
    n_actions = 0
    logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
    dataset = TransDataset()
    logger.save_trans_dataset(dataset, i=n_actions)

    while n_actions < args.max_actions:
        print('|dataset| = %i' % n_actions)

        if model_path is None:
            pddl_model_type = 'optimistic'
            model_logger = None
        else:
            pddl_model_type = 'learned'
            model_logger = ExperimentLogger(model_path)
        trajectory, n_actions = collect_trajectory_wrapper(args,
                                                    pddl_model_type,
                                                    logger,
                                                    args.goal_progress,
                                                    n_actions,
                                                    separate_process=True,
                                                    model_logger=model_logger)

        # if trajectory returned, visualize and add to dataset
        if trajectory:
            # visualize goal and success
            #world.plot_datapoint(i=n_actions-1, show=args.vis_performance)

            if args.balanced:
                # balance dataset by removing added element if makes it unbalanced
                num_per_class = args.max_actions // 2

                dataset = logger.load_trans_dataset()
                n_datapoints = len(dataset)
                num_pos_datapoints = sum([y for x,y in dataset])
                num_neg_datapoints = n_datapoints - num_pos_datapoints
                if num_pos_datapoints > num_per_class or num_neg_datapoints > num_per_class:
                    print('Removing last added dataset.')
                    logger.remove_dataset(i=n_actions)
                    n_actions -= 1

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
                                        logger)
                    trajectory = execute_plan(world, *plan_data)
                    success = all([t_seg[3] for t_seg in trajectory])
                    color = 'g' if success else 'r'
                    world.plot_datapoint(i=len(dataset)-1, color=color, show=True)
                    world.disconnect()
            '''
    return logger

if __name__ == '__main__':
    # Data collection args
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--max-actions',
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
    parser.add_argument('--data-collection-mode',
                        type=str,
                        default='random-goals-opt',
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned', \
                                'sequential-plans', 'sequential-goals', 'engineered-goals-dist', \
                                'engineered-goals-size'],
                        help='method of data collection')
    parser.add_argument('--n-datasets',
                        type=int,
                        default=1,
                        help='number of datasets to generate')
    # for now this assumes that you always want to use the most trained model on the path for planning
    # (as opposed to a different i)
    parser.add_argument('--model-paths',
                        type=str,
                        nargs='+',
                        help='list of model paths to use for planning')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    args.balanced = args.balanced == 'True'

    if len(args.model_paths) > 1:
        assert len(args.model_paths) == args.n_datasets, 'If using multiple models to generate datasets \
                        then should generate the same number of datasets'

    if len(args.model_paths) > 0:
        assert 'learned' in args.data_collection_mode, 'Must use learned model if passing in model paths'

    dataset_paths = []
    for di in range(args.n_datasets):
        if len(args.model_paths) == 0:
            model_path = None
        elif len(args.model_paths) == 1:
            model_path = args.model_paths[0]
        else:
            model_path = args.model_paths[di]
        logger = gen_dataset(args, model_path)
        dataset_paths.append(logger.exp_path)

    print('---Dataset paths---')
    print(dataset_paths)
