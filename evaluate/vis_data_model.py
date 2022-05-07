import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import pb_robot

from domains.tools.world import ToolsWorld
from experiments.utils import ExperimentLogger
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen


dir = 'accuracy'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def indiv_plot(args, contact_info, action, obj, grasp, world, mean_fn, std_fn, dataset, ts, mi, model_logger, dataset_logger):
    if 'move_contact' in action:
        if args.plot_std:
            n_axes = 3
        else:
            n_axes = 2
    else:
        if args.plot_std:
            n_axes = 2
        else:
            n_axes = 1
    fig, axes = plt.subplots(n_axes, figsize=(5, 3.3*n_axes))
    contact = None
    for ctype in contact_info:
        if ctype in action:
            contact = contact_info[ctype]
            # plot the tool
            if 'move_contact' in action:
                ax = axes if n_axes==1 else axes[n_axes-1]
                world.vis_tool_ax(contact, obj, action, ax, frame='world')
    x_axes, y_axes = world.get_world_limits(obj, action, contact)

    if not args.just_dataset:
        ax = axes if n_axes==1 else axes[0]
        world.vis_dense_plot(action, obj, grasp, ax, x_axes, y_axes, 0, 1, value_fn=mean_fn, cell_width=args.cell_width)
        if args.plot_std:
            ax = axes if n_axes==1 else axes[1]
            world.vis_dense_plot(action, obj, grasp, ax, x_axes, y_axes, None, None, value_fn=std_fn, cell_width=args.cell_width)

    for ai in range(n_axes):
        print(action, obj, len(dataset.datasets[action][obj][grasp]))
        ax = axes if n_axes==1 else axes[ai]
        world.vis_dataset(ax, action, obj, dataset.datasets[action][obj][grasp])
        ax.set_xlim(*x_axes)
        ax.set_ylim(*y_axes)

    ax = axes if n_axes==1 else axes[0]
    ax.set_title('Mean Ensemble Predictions')
    if args.plot_std:
        ax = axes if n_axes==1 else axes[1]
        ax.set_title('Std Ensemble Predictions')
    if 'move_contact' in action:
        ax = axes if n_axes==1 else axes[n_axes-1]
        ax.set_title('Tool Contact')

    if grasp is not None:
        fname = 'acc_%s_%s_g%s_%s_%i.png' % (ts, action, grasp, obj, mi)
    else:
        fname = 'acc_%s_%s_%s_%i.png' % (ts, action, obj, mi)
    if args.just_dataset:
        dataset_logger.save_figure(fname, dir=dir)
    else:
        model_logger.save_figure(fname, dir=dir)
    plt.close()


def gen_plots(args):
    if args.just_dataset:
        model_logger = None
    else:
        model_logger = ExperimentLogger(args.model_exp_path)
        if args.action_step:
            _, txs = model_logger.get_dir_indices('models')
            mi = find_nearest(txs, args.action_step)
            ensembles = model_logger.load_trans_model(i=mi)
        else:
            ensembles, mi = model_logger.load_trans_model(ret_i=True)
        print('Generating figures for models on path %s step %i' % (args.model_exp_path, mi))
    world = ToolsWorld()

    if args.dataset_exp_path:
        dataset_logger = ExperimentLogger(args.dataset_exp_path)
    else:
        dataset_logger = model_logger
    if args.action_step:
        _, txs = dataset_logger.get_dir_indices('datasets')
        di = find_nearest(txs, args.action_step)
        dataset = model_logger.load_trans_dataset('', i=di)
    else:
        dataset, di = dataset_logger.load_trans_dataset('', ret_i=True)

    print('Plotting dataset on path %s step %i' % (dataset_logger.exp_path, di))

    ts = time.strftime('%Y%m%d-%H%M%S')

    # make a plot for each contact type (subplot for mean, std, and tool vis)
    contacts_fn = get_contact_gen(world.panda.planning_robot)
    contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
    contact_info = {}
    for contact in contacts:
        contact_info[contact[0].type] = contact[0]

    if not args.just_dataset:
        mean_fn = get_model_accuracy_fn(ensembles, 'mean')
        std_fn = get_model_accuracy_fn(ensembles, 'std')
    else:
        mean_fn, std_fn = None, None
        mi = di

    for obj in ['yellow_block', 'blue_block']:
        for action in ['pick', 'move_contact-push_pull', 'move_contact-poke', 'move_holding']:
            if 'move_contact' in action:
                for grasp in ['p1', 'n1']:
                    if len(dataset.datasets[action][obj][grasp]) > 0:
                        indiv_plot(args, contact_info, action, obj, grasp, world, mean_fn, std_fn, dataset, ts, mi, model_logger, dataset_logger)
            else:
                grasp = 'None'
                if len(dataset.datasets[action][obj][grasp]) > 0:
                    indiv_plot(args, contact_info, action, obj, grasp, world, mean_fn, std_fn, dataset, ts, mi, model_logger, dataset_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--model-exp-path',
                        type=str,
                        help='experiment path to visualize results for')
    parser.add_argument('--dataset-exp-path',
                        type=str,
                        help='experiment path to visualize results for')
    parser.add_argument('--just-dataset',
                        action='store_true')
    parser.add_argument('--action-step',
                        type=int,
                        help='only generate plots for this action step (or the step closest to this value)')
    parser.add_argument('--plot-std',
                        action='store_true')
    parser.add_argument('--cell-width',
                        type=float,
                        default=0.01)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_plots(args)
