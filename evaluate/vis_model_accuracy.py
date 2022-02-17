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
    parser.add_argument('--plot-freq',
                        type=int,
                        default=4,
                        help='number of actions taken between each generated figure')
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

    # plot functions
    for dataset, di in logger.get_dataset_iterator('train'):
        if not di % args.plot_freq:
            dataset = logger.load_trans_dataset('train', i=di)
            val_dataset = logger.load_trans_dataset('val', i=di)
            curr_dataset = logger.load_trans_dataset('curr')
            model = logger.load_trans_model(i=di)
            print('Generating figures for action step %i' % di)
            ts = time.strftime('%Y%m%d-%H%M%S')

            # make a plot for each contact type (subplot for mean, std, and tool vis)
            contacts_fn = get_contact_gen(world.panda.planning_robot)
            contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

            mean_fn = get_model_accuracy_fn(model, 'mean')
            std_fn = get_model_accuracy_fn(model, 'std')
            seq_fn = get_seq_fn(model)

            all_axes = {}
            for ci, contact in enumerate(contacts):
                cont = contact[0]
                fig, axes = plt.subplots(4, figsize=(8,15))
                world.vis_dense_plot(cont, axes[0], [world.min_x, world.max_x], \
                                [world.min_y, world.max_y], 0, 1, value_fn=mean_fn)
                world.vis_dense_plot(cont, axes[1], [world.min_x, world.max_x], \
                                [world.min_y, world.max_y], None, None, value_fn=std_fn)
                world.vis_dense_plot(cont, axes[2], [world.min_x, world.max_x], \
                                [world.min_y, world.max_y], None, None, value_fn=seq_fn)
                for ai in range(3):
                    world.vis_failed_trajes(cont, axes[ai], logger)
                    world.vis_dataset(cont, axes[ai], dataset, linestyle='-')
                    world.vis_dataset(cont, axes[ai], val_dataset, linestyle='--')
                    world.vis_dataset(cont, axes[ai], curr_dataset, linestyle=':')

                world.vis_tool_ax(cont, axes[3])

                axes[0].set_title('Mean Ensemble Predictions')
                axes[1].set_title('Std Ensemble Predictions')
                axes[2].set_title('Sequential Score')
                all_axes[ci] = axes

                fname = 'acc_%s_%i_%i.svg' % (ts, ci, di)
                logger.save_figure(fname, dir=dir)
                plt.close()
