import time
import argparse
import matplotlib.pyplot as plt

from experiments.utils import ExperimentLogger
from domains.utils import init_world
from domains.tools.world import CONTACT_TYPES
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen

test_dataset_path = 'logs/experiments/90_random_goals_balanced-20220219-170056'

def gen_plots(args):
    dir = 'accuracy'
    world = init_world('tools',
                        None,
                        'optimistic',
                        False,
                        None)

    model_logger = ExperimentLogger(args.model_exp_path)
    ensembles, mi = model_logger.load_trans_model(ret_i=True)

    if args.dataset_exp_path:
        dataset_logger = ExperimentLogger(args.dataset_exp_path)
    else:
        dataset_logger = model_logger
    dataset, di = dataset_logger.load_trans_dataset('', ret_i=True)

    print('Generating figures for models on path %s step %i' % (args.model_exp_path, mi))
    print('Plotting dataset on path %s step %i' % (args.dataset_exp_path, di))

    ts = time.strftime('%Y%m%d-%H%M%S')

    # make a plot for each contact type (subplot for mean, std, and tool vis)
    contacts_fn = get_contact_gen(world.panda.planning_robot)
    contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

    mean_fn = get_model_accuracy_fn(ensembles, 'mean')
    std_fn = get_model_accuracy_fn(ensembles, 'std')
    #seq_fn = get_seq_fn(ensembles)

    for type in CONTACT_TYPES:
        fig, axes = plt.subplots(3, figsize=(4, 12))
        world.vis_dense_plot(type, axes[0], [-1, 1], [-1, 1], 0, 1, value_fn=mean_fn)
        world.vis_dense_plot(type, axes[1], [-1, 1], [-1, 1], None, None, value_fn=std_fn)
        #world.vis_dense_plot(type, axes[2], [-1, 1], [-1, 1], None, None, value_fn=seq_fn)
        for ai in range(3):
            world.vis_dataset(axes[ai], dataset.datasets[type], linestyle='-')
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


        for contact in contacts:
            cont = contact[0]
            if cont.type == type:
                world.vis_tool_ax(cont, axes[2], frame='cont')

        axes[0].set_title('Mean Ensemble Predictions')
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
