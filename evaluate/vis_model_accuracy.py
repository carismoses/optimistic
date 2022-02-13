import time
import argparse
import matplotlib.pyplot as plt

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
    dir = 'accuracy'
    for model, mi in logger.get_model_iterator('trans'):
        print(mi)
        ts = time.strftime('%Y%m%d-%H%M%S')
        axes = world.vis_model_accuracy(model, goal_from_state=True)
        world.vis_dataset(logger, axes=axes, dataset_i=mi, goal_from_state=True)
        for ci in axes:
            fname = 'acc_%s_%i.svg' % (ts, ci)
            logger.save_figure(fname, dir=dir)
            plt.close()
