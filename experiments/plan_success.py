import argparse

from experiments.strategies import collect_trajectory_wrapper
from learning.utils import ExperimentLogger
args, pddl_model_type, dataset_logger, progress, \
                            goal_xy=None, separate_process=False, model_logger=None, \
                            save_to_dataset=True

def solve_trajectory(model, logger, mi, planner_args, test_goals):
    trajectories = []
    for goal in test_goals:
        trajectory = collect_trajectory_wrapper(planner_args,
                                                'learned',
                                                None,
                                                None,
                                                separate_process=True,
                                                model_logger=logger,
                                                save_to_dataset=False)
        trajectories.append(trajectory)

    # save trajectory data
    logger.save_trajectories(trajectories, i=mi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--exp-path',
                        type=str,
                        help='the exp-path to plan for')
    parser.add_argument('--actions-step',
                        type=int,
                        help='number of actions between each datapoint')
    parser.add_argument('--single-action-step',
                        type=int,
                        help='use if want to just calculate plan success for a single action step')
    parser.add_argument('--test-goals-path',
                        type=str,
                        help='path to test goals')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    logger = ExperimentLogger(args.exp_path)
    if args.actions_step:
        assert not args.actions_step % logger.args.train_freq, 'actions-step arg must be divisible by train_freq'

    goals_logger = ExperimentLogger(args.test_goals_path)
    test_goals = goals_logger.load_goals()

    planner_args = argparse.Namespace(domain='tools',
                                        domain_args=[],
                                        vis=False,
                                        data_collection_mode='random-goals-learned')

    if args.single_action_step:
        model = logger.load_trans_model(i=args.single_action_step)
        solve_trajectory(model, logger, args.single_action_step, planner_args, test_goals)
    else:
        for model, mi in logger.get_model_iterator():
            if not mi % args.actions_step:
                solve_trajectory(model, logger, mi, planner_args, test_goals)
