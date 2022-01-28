import argparse

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger, add_trajectory_to_dataset
from tamp.utils import execute_plan
from domains.utils import init_world
from experiments.strategies import collect_trajectory_wrapper
import matplotlib.pyplot as plt

def gen_dataset(args):
    #plt.ion()
    n_actions = 0
    logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
    dataset = TransDataset()
    logger.save_trans_dataset(dataset, i=n_actions)

    while n_actions < args.max_actions:
        print('|dataset| = %i' % n_actions)

        pddl_model_type = 'optimistic'
        args.domain = 'tools'
        args.domain_args = None
        args.vis = False
        args.data_collection_mode = 'random-goals-opt'
        args.n_seq_plans = None
        trajectory, n_actions = collect_trajectory_wrapper(args,
                                                    pddl_model_type,
                                                    logger,
                                                    args.goal_progress,
                                                    n_actions,
                                                    separate_process=True)

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
                        action='store_true',
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
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_dataset(args)
