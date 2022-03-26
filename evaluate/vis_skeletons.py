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


dir = 'skeletons'


def get_plan_name(plan):
    plan_name = ''
    for name, args in plan:
        if name not in ['move_holding', 'move_free']:
            if name == 'move_contact':
                obj = args[2].readableName
                if plan_name == '':
                    plan_name += obj+'_'
                plan_name += name+'_'
                ctype = args[5].type
                plan_name += ctype+'_'
            elif name == 'pick' and args[0].readableName != 'tool':
                obj = args[0].readableName
                if plan_name == '':
                    plan_name += obj+'_'
                plan_name += name+'_'
    return plan_name[:-1]


def gen_plots(args):
    logger = ExperimentLogger(args.dataset_exp_path)

    # use skeleton fns instead of plans in logger so always in same order (and
    # use the same colors for comparing plots)
    all_plans = []
    for skel_fn in get_skeleton_fns():
        dummy_world = ToolsWorld()
        dummy_goal = dummy_world.generate_dummy_goal()
        skel = skel_fn(dummy_world, dummy_goal)
        for obj in ['yellow_block', 'blue_block']:
            new_plans = [obj]
            for name, skel_args in skel:
                if name == 'move_contact':
                    old_plans = new_plans
                    new_plans = []
                    for old_plan in old_plans:
                        for ctype in ['push_pull', 'poke']:
                            new_plans.append(old_plan+'_move_contact_'+ctype)
                elif name == 'pick' and skel_args[0].readableName != 'tool':
                    old_plans = new_plans
                    new_plans = []
                    for old_plan in old_plans:
                        new_plans.append(old_plan+'_pick_move_holding')
            all_plans += new_plans


    skel_count = {pn: [0] for pn in all_plans}
    for plan, pi in logger.get_plan_iterator():
        plan_name = get_plan_name(plan)
        for pn in all_plans:
            if plan_name == pn:
                #print('matching name!')
                skel_count[pn] += [1]
            else:
                skel_count[pn] += [0]

    fig, ax = plt.subplots(figsize=(15,5))
    for plan_name, counts in skel_count.items():
        if sum(counts) > 0:
            ax.bar(np.arange(len(counts)), counts, width=1.0, label=plan_name)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fname = 'skeletons.png'

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
