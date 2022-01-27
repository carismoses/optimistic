import argparse

from learning.datasets import TransDataset
from learning.utils import ExperimentLogger, add_trajectory_to_dataset
from tamp.utils import execute_plan
from domains.utils import init_world
from experiments.strategies import collect_trajectory
import matplotlib.pyplot as plt

def gen_dataset(args):
    plt.ion()
    logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
    dataset = TransDataset()
    while len(dataset) < args.max_actions:
        print('|dataset| = %i' % len(dataset))
        vis = False
        world = init_world('tools',
                            None,
                            'optimistic',
                            vis,
                            logger)

        # change goal space, plan for and execute trajectory
        world.change_goal_space(args.goal_progress)
        trajectory, plan_data = collect_trajectory(world, logger, 'random-goals-opt', ret_plan=True)

        # if trajectory returned, visualize and add to dataset
        if trajectory:
            # add to dataset
            add_trajectory_to_dataset('tools', dataset, trajectory, world)
            logger.save_trans_dataset(dataset, i=len(dataset))

            # visualize goal and success
            world.plot_datapoint(i=len(dataset)-1, show=args.vis_performance)

            # optionally replay with pyBullet
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

    gen_dataset(args)
