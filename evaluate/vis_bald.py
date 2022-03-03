import time
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen

types=['push_pull']

def gen_plots(args):
    dir = 'bald'
    world = ToolsWorld(False, None, types)

    model_logger = ExperimentLogger(args.model_exp_path)
    dataset_lens = {}
    dataset_lens_set = False
    for dataset, di in model_logger.get_dataset_iterator(''):
        try:
            ensembles = model_logger.load_trans_model(i=di-4)
        except:
            continue
        if not dataset_lens_set:
            for type in types:
                dataset_lens[type] = len(dataset.datasets[type])
            dataset_lens_set = True
        print('Generating figures for models on path %s step %i' % (args.model_exp_path, di))
        ts = time.strftime('%Y%m%d-%H%M%S')

        # make a plot for each contact type (subplot for mean, std, and tool vis)
        contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
        contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

        mean_fn = get_model_accuracy_fn(ensembles, 'mean')
        std_fn = get_model_accuracy_fn(ensembles, 'std')
        seq_fn = get_seq_fn(ensembles)

        for type in types:
            fig, axes = plt.subplots(4, figsize=(5, 10))
            world.vis_dense_plot(type, axes[0], [-.5, .6], [-.25, .45], 0, 1, value_fn=mean_fn)
            world.vis_dense_plot(type, axes[1], [-.5, .6], [-.25, .45], 0, .4, value_fn=std_fn)
            world.vis_dense_plot(type, axes[2], [-.5, .6], [-.25, .45], 0, .4, value_fn=seq_fn)
            dlen = len(dataset.datasets[type])
            for ai in range(3):
                for dj, (x, y) in enumerate(dataset.datasets[type]):
                    if (dlen > dataset_lens[type]) and (dj == dlen-1) and (ai == 2):
                        color = 'b'
                        axes[ai].plot(*x, color+'.')
                        dataset_lens[type] += 1
                    elif (dlen > dataset_lens[type]) and (dj == dlen-1) and (ai != 2):
                        continue
                    else:
                        color = 'r' if y == 0 else 'g'
                        axes[ai].plot(*x, color+'.')

            for contact in contacts:
                cont = contact[0]
                if cont.type == type:
                    world.vis_tool_ax(cont, axes[3], frame='cont')
                axes[3].set_xlim([-.5, .6])
                axes[3].set_ylim([-.25, .45])

                # move over so aligned with colorbar images
                div = make_axes_locatable(axes[3])
                div.append_axes("right", size="10%", pad=0.5)

            axes[0].set_title('Mean Ensemble Predictions')
            axes[1].set_title('Std Ensemble Predictions')
            axes[2].set_title('BALD Score')

            #if type == 'poke':
            fname = 'acc_%s_%s_%i.png' % (ts, type, di)
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
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_plots(args)
