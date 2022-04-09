import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import pb_robot

from domains.tools.world import ToolsWorld
from experiments.utils import ExperimentLogger
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen
from experiments.skeletons import get_skeleton_fns


dir = 'dataset_info'

def gen_plots(args):
    logger = ExperimentLogger(args.dataset_exp_path)

    types = []
    for obj in ['yellow_block', 'blue_block']:
        for action in ['pick', 'move_contact-push_pull', 'move_contact-poke', 'move_holding']:
            if 'move_contact' in action:
                for grasp in ['p1', 'n1']:
                    types.append((action, obj, grasp))
            else:
                grasp = 'None'
                types.append((action, obj, grasp))

    type_count = {t: [[0], [0]] for t in types}
    dataset = logger.load_trans_dataset('')
    for ix in range(len(dataset)):
        print(ix)
        x, y, action, obj, grasp = dataset[ix]
        key = (action, obj, grasp)
        if y:
            type_count[key][1].append(type_count[key][1][-1]+1)
            type_count[key][0].append(ix)

    fig, ax = plt.subplots(figsize=(15,5))
    for t, (xs, ys) in type_count.items():
        print(len(xs))  
        print(xs, ys)
        ax.plot(xs, ys, label=t)

    # Put a legend to the right of the current axis
    ax.legend(loc='center left')#, bbox_to_anchor=(1.0, 0.5))
    fname = 'action_types.png'

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

