import argparse

from experiments.strategies import collect_trajectory_wrapper
from learning.utils import ExperimentLogger

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
    parser.add_argument('--test-goals-path',
                        type=str,
                        help='path to test goals')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    logger = ExperimentLogger(args.exp_path)
    assert not args.actions_step % logger.args.train_freq, 'actions-step arg must be divisible by train_freq'

    goals_logger = ExperimentLogger(args.test_goals_path)
    test_goals = goals_logger.load_goals()

    planner_args = argparse.Namespace(domain='tools',
                                        domain_args=[],
                                        vis=False,
                                        data_collection_mode='random-goals-learned')
    for model, mi in logger.get_model_iterator():
        if not mi % args.actions_step:
            trajectories = []
            for goal in test_goals:
                trajectory = collect_trajectory_wrapper(planner_args,
                                                        'learned',
                                                        None,
                                                        None,
                                                        separate_process=False,
                                                        model_logger=logger,
                                                        save_to_dataset=False,
                                                        goal=goal)
                trajectories.append(trajectory)

            # save trajectory data
            logger.save_trajectories(trajectories, i=mi)
