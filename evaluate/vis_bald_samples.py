import os
import dill as pickle
import numpy as np

import matplotlib.pyplot as plt
from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen

from experiments.strategies import sequential_bald
from experiments.gen_ss_skeleton import plot_action
from experiments.skeletons import get_skeleton_name
from experiments.utils import ExperimentLogger

###### PARAMETERS
path = 'logs/experiments/yellow_actions500_skel01467-20220329-183215'
import pdb; pdb.set_trace()
######

fig_dir = os.path.join(path, 'figures', 'sample_scores')
logger = ExperimentLogger(path)

world = ToolsWorld()
contacts_fn = get_contact_gen(world.panda.planning_robot)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
contact_preds = {}
for contact in contacts:
    if contact[0].type not in contact_preds:
        contact_preds[contact[0].type] = contact[0]

def get_grasp_indices(samples, grasp):
    grasp_ixs = []
    for pi, (plan,_,_) in enumerate(samples):
        for action in plan:
            if action.name == 'move_contact':
                grasp_xy = action.args[1].grasp_objF[:2,3]
                if np.allclose(grasp_xy, grasp):
                    grasp_ixs.append(pi)
    return grasp_ixs

def get_num_ax(plan):
    num = 0
    indices = []
    for ai, action in enumerate(plan):
        if action.name == 'move_contact':
            num += 1
            indices.append(ai)
        elif action.name == 'pick':
            if action.args[0].readableName != 'tool':
                num += 1
                indices.append(ai)
        elif action.name == 'move_holding':
            if action.args[0].readableName != 'tool':
                num += 1
                indices.append(ai)
    return num, indices

def gen_plot(world, mi_min, mi_max, ensembles, skel_num, skel_key, samples, grasp):
    # calc and normalize bald scores
    bald_scores = np.array([sequential_bald(pddl_plan, ensembles, world) for pddl_plan,_,_ in samples])
    bald_scores = np.clip(bald_scores, 0, 1)

    ## plot all sampled plans ##
    dummy_plan = samples[0][0] # all plans in list are the same length
    num_ax, plt_indices = get_num_ax(dummy_plan)
    fig_path = os.path.join(fig_dir, str(mi_max))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    mci = 0 # if more than 1 move_contact in a plan have to index through different push names

    # get indices that match this grasp
    if len(skel_key.ctypes) > 0:
        grasp_ixs = get_grasp_indices(samples, grasp)
    else:
        grasp_ixs = np.arange(len(samples))

    fig, axes = plt.subplots(2, num_ax, figsize=(15,10))
    for axi, ai in enumerate(plt_indices):
        ## plot all skeleton samples and bald scores ##
        ctype = None
        if samples[0][0][ai].name == 'move_contact':
            ctype = skel_key.ctypes[mci]
            mci += 1
        for ax_row in range(2):
            ax = axes[ax_row] if num_ax == 1 else axes[ax_row, axi]

            for pi in grasp_ixs:
                plan, _, _ = samples[pi]
                plot_action(world,
                            plan[ai],
                            ax,
                            contact_preds,
                            ctype,
                            color=str(bald_scores[pi]))
            if ctype is not None:
                ax.set_title(dummy_plan[ai].name+'_'+ctype)
            else:
                ax.set_title(dummy_plan[ai].name)
            ax.set_aspect('equal')

            ## plot all executed plans ##
            dataset = logger.load_trans_dataset('', i=mi_max)
            for ixs in dataset.ixs:
                x, y, action_type, object = dataset[ixs]
                if dummy_plan[ai].name in action_type and object == skel_key.goal_obj:
                    if ixs > mi_min:
                        color = 'b'
                    else:
                        color = 'g' if y == 1 else 'r'
                    if 'move_contact' in dummy_plan[ai].name:
                        if ctype in action_type:
                            ax.plot(*x[2:4], color=color, marker='o', markerfacecolor='none')
                    elif dummy_plan[ai].name in ['pick', 'move_holding']:
                        ax.plot(*x, color=color, marker='o', markerfacecolor='none')

    skeleton_name = get_skeleton_name(samples[0][0], skel_key)
    if grasp is not None:
        grasp_str = 'p1' if grasp == [.1,0] else 'n1'
    else:
        grasp_str = ''
    fname = '%s_%s.png' % (skeleton_name, grasp_str)
    plt.savefig(os.path.join(fig_path, fname))
    plt.close()

def main():
    mi_min = 0
    for ensembles, mi_max in logger.get_model_iterator():
        # plot samples colored by sequential score
        for skel_num in logger.args.skel_nums:
            filename = 'logs/all_skels/skel%i/ss_skeleton_samples.pkl'%skel_num
            with open(filename, 'rb') as handle:
                samples_dict = pickle.load(handle)
            skel_key = list(samples_dict.keys())[0]
            samples = samples_dict[skel_key]
            if len(skel_key.ctypes) > 0: # means a move_contact action is in the skeleton
                for grasp in [[-.1, 0.], [.1, 0.]]:
                    gen_plot(world, mi_min, mi_max, ensembles, skel_num, skel_key, samples, grasp=grasp)
            else:
                gen_plot(world, mi_min, mi_max, ensembles, skel_num, skel_key, samples, grasp=None)
        mi_min = mi_max

if __name__ == '__main__':
    main()
