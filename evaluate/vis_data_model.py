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

def indiv_plot(contact_info, action, obj, world, mean_fn, std_fn, dataset, ts, mi, model_logger, dataset_logger, grasp=None):
    n_axes = 3
    fig, axes = plt.subplots(n_axes, figsize=(5, 10))
    contact = None
    for ctype in contact_info:
        if ctype in action:
            contact = contact_info[ctype]
            # plot the tool
            world.vis_tool_ax(contact, obj, action, axes[n_axes-1], frame='cont')
    x_axes, y_axes = world.get_cont_frame_limits(obj, action, contact)

    if not args.just_dataset:
        world.vis_dense_plot(action, obj, axes[0], x_axes, y_axes, 0, 1, value_fn=mean_fn, cell_width=0.1, grasp=grasp)
        world.vis_dense_plot(action, obj, axes[1], x_axes, y_axes, None, None, value_fn=std_fn, cell_width=0.1, grasp=grasp)

    for ai in range(n_axes):
        print(action, obj, len(dataset.datasets[action][obj]))
        world.vis_dataset(axes[ai], dataset.datasets[action][obj], grasp=grasp)

    axes[0].set_title('Mean Ensemble Predictions')
    axes[1].set_title('Std Ensemble Predictions')
    axes[2].set_title('Tool Contact')

    if grasp is not None:
        grasp_str = 'p1' if grasp == [.1,0] else 'n1'
        fname = fname = 'acc_%s_%s_g%s_%s_%i.png' % (ts, action, grasp_str, obj, mi)
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
        ensembles, mi = model_logger.load_trans_model(ret_i=True)
        print('Generating figures for models on path %s step %i' % (args.model_exp_path, mi))
    world = ToolsWorld()

    if args.dataset_exp_path:
        dataset_logger = ExperimentLogger(args.dataset_exp_path)
    else:
        dataset_logger = model_logger
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
        for action in ['pick', 'move_contact-push_pull', 'move_contact-poke']:
            if 'move_contact' in action:
                for grasp in [[-.1, 0.], [.1, 0.]]:
                    indiv_plot(contact_info, action, obj, world, mean_fn, std_fn, dataset, ts, mi, model_logger, dataset_logger, grasp=grasp)
            else:
                indiv_plot(contact_info, action, obj, world, mean_fn, std_fn, dataset, ts, mi, model_logger, dataset_logger)


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
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_plots(args)