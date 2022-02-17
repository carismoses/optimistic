import argparse

from domains.utils import init_world
from learning.utils import ExperimentLogger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--exp-name',
                        type=str,
                        help='the exp_name to save the goal set to')
    parser.add_argument('--n-goals',
                        type=int,
                        help='number of plans per datapoint')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')

    world = init_world('tools',
                        [],
                        'optimistic',
                        False,
                        None)
    goals = []
    for _ in range(args.n_goals):
        goal = world.generate_goal()
        goals.append(goal)


    # For calculating plan success
    logger.save_goals(goals)
