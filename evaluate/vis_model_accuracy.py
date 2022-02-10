import argparse

from experiments.utils import ExperimentLogger
from domains.utils import init_world


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--exp-path',
                        type=str,
                        help='experiment path to visualize results for')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    world = init_world('tools',
                        None,
                        'optimistic',
                        False,
                        None)

    logger = ExperimentLogger(args.exp_path)
    for model, mi in logger.get_model_iterator('trans'):
        print(mi)
        world.visualize_bald(None,
                            None,
                            model,
                            None,
                            logger,
                            goal_from_state=True,
                            plot_bald_scores=False,
                            dataset_i=mi)
