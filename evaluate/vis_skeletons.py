import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import pb_robot

from domains.tools.world import ToolsWorld
from experiments.utils import ExperimentLogger
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen


dir = 'skeletons'

def get_plan_name(plan):
    plan_name = ''
    for name, args in plan:
        if name not in ['move_holding', 'move_free']:
            if name == 'move_contact':
                plan_name += name+'_'
                obj = args[2].readableName
                ctype = args[5].type
                plan_name += obj+'_'
                plan_name += ctype+'_'
            elif name == 'pick' and args[0].readableName != 'tool':
                plan_name += name+'_'
                obj = args[0].readableName
                plan_name += obj+'_'
    return plan_name[:-1]


def gen_plots(args):
    logger = ExperimentLogger(args.dataset_exp_path)
    all_plans = []
    for plan, pi in logger.get_plan_iterator():
        plan_name = get_plan_name(plan)
        if plan_name not in all_plans:
            all_plans.append(plan_name)

    skel_count = {pn: [0] for pn in all_plans}
    for plan, pi in logger.get_plan_iterator():
        plan_name = get_plan_name(plan)
        for pn in all_plans:
            if plan_name == pn:

                skel_count[pn] += [1]
            else:
                skel_count[pn] += [0]

    fig, ax = plt.subplots()
    print(skel_count.keys())
    for plan_name, counts in skel_count.items():
        ax.bar(np.arange(len(counts)), counts, width=1.0, label=plan_name)
    ax.legend()
    plt.show()

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
