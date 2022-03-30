import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import pb_robot

from domains.tools.world import ToolsWorld
from experiments.utils import ExperimentLogger
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from evaluate.vis_skeletons import get_plan_name
from domains.tools.primitives import get_contact_gen
from experiments.skeletons import get_skeleton_fns


dir = 'plan_len'
max_l = 10

def gen_plots(args):
    logger = ExperimentLogger(args.dataset_exp_path)

    len_count = {l: [] for l in range(max_l)}
    prev_dl = 0
    for dataset, _ in logger.get_dataset_iterator(''):
        len_traj = len(dataset) - prev_dl
        prev_dl = len(dataset)
        for l in range(max_l):
            if len_traj == l:
                len_count[l] += [1]
            else:
                len_count[l] += [0]

    fig, ax = plt.subplots(figsize=(15,5))
    for l, counts in len_count.items():
        if sum(counts) > 0:
            ax.bar(np.arange(len(counts)), counts, width=1.0, label=l)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fname = 'lens.png'

    logger.save_figure(fname, dir=dir)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--dataset-exp-path',
                        type=str,
                        help='experiment path to visualize results for')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_plots(args)
