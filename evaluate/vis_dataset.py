import time
import argparse
import matplotlib.pyplot as plt

from experiments.utils import ExperimentLogger
from domains.utils import init_world
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen


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
    dir = 'dataset'
    dataset = logger.load_trans_dataset('')
    all_axes = {}

    contacts_fn = get_contact_gen(world.panda.planning_robot)
    contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

    all_axes = {}
    for type, dataset in dataset.datasets.items():
        fig, axes = plt.subplots(2, figsize=(5,10))
        world.vis_dataset(axes[0], dataset)
        for contact in contacts:
            cont = contact[0]
            if cont.type == type:
                world.vis_tool_ax(cont, axes[1])

        axes[0].set_title('%s Dataset' % type)
        axes[1].set_title('Contact Configuration')

        axes[0].set_aspect('equal')
        axes[0].set_xlim([world.min_x, world.max_x])
        axes[0].set_ylim([world.min_y, world.max_y])

        all_axes[type] = axes

        fname = 'dataset_%s.svg' % type
        logger.save_figure(fname, dir=dir)
        plt.close()
