import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import pb_robot

from domains.tools.world import ToolsWorld
from experiments.utils import ExperimentLogger
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen


def gen_plots(args):
    dir = 'accuracy'

    model_logger = ExperimentLogger(args.model_exp_path)
    world = ToolsWorld(False, None, model_logger.args.actions, model_logger.args.objects)
    ensembles, mi = model_logger.load_trans_model(ret_i=True)

    if args.dataset_exp_path:
        dataset_logger = ExperimentLogger(args.dataset_exp_path)
    else:
        dataset_logger = model_logger
    dataset, di = dataset_logger.load_trans_dataset('', ret_i=True)

    print('Generating figures for models on path %s step %i' % (args.model_exp_path, mi))
    print('Plotting dataset on path %s step %i' % (dataset_logger.exp_path, di))

    ts = time.strftime('%Y%m%d-%H%M%S')

    # make a plot for each contact type (subplot for mean, std, and tool vis)
    contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
    contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
    contact_info = {}
    for contact in contacts:
        contact_info[contact[0].type] = contact[0]


    mean_fn = get_model_accuracy_fn(ensembles, 'mean')
    std_fn = get_model_accuracy_fn(ensembles, 'std')
    #seq_fn = get_seq_fn(ensembles)

    for type in model_logger.args.contact_types:
        if model_logger.args.n_models == 1:
            n_axes = 2
        else:
            n_axes = 3
        fig, axes = plt.subplots(n_axes, figsize=(5, 10))

        x_axes, y_axes = world.get_cont_frame_limits(model_logger.args.goal_obj,
                                                    contact_info[type])

        world.vis_dense_plot(type, axes[0], x_axes, y_axes, 0, 1, value_fn=mean_fn, cell_width=0.01)
        if n_axes == 3:
            world.vis_dense_plot(type, axes[1], x_axes, y_axes, None, None, value_fn=std_fn, cell_width=0.01)
        #world.vis_dense_plot(type, axes[2], [-1, 1], [-1, 1], None, None, value_fn=seq_fn)
        for ai in range(n_axes):
            world.vis_dataset(axes[ai], dataset.datasets[type], model_logger.args.goal_type, linestyle='-')
            #world.vis_dataset(cont, axes[ai], val_dataset, linestyle='--')
            #world.vis_dataset(cont, axes[ai], curr_dataset, linestyle=':')
            #world.vis_failed_trajes(cont, axes[ai], logger)
        # visualize failed planning goals
        if args.dataset_exp_path == args.model_exp_path:
            failed_goals = dataset_logger.load_failed_plans()
            for _, contact_type, x, _ in failed_goals:
                if contact_type == type:
                    world.plot_block(axes[0], x, 'b')
                    world.plot_block(axes[1], x, 'b')


        world.vis_tool_ax(contact_info[type], axes[n_axes-1], frame='cont')

        axes[0].set_title('Mean Ensemble Predictions')
        if n_axes == 3:
            axes[1].set_title('Std Ensemble Predictions')
        #axes[2].set_title('Sequential Score')
        #all_axes[ci] = axes

        fname = 'acc_%s_%s_%i.png' % (ts, type, mi)
        model_logger.save_figure(fname, dir=dir)
        plt.close()

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
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_plots(args)
