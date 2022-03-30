import os
import dill as pickle
import numpy as np
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen
from evaluate.plot_value_fns import get_model_accuracy_fn
from experiments.strategies import sequential_bald
from experiments.gen_ss_skeleton import plot_action
from experiments.skeletons import get_skeleton_name
from experiments.utils import ExperimentLogger

###### PARAMETERS
path = 'logs/experiments/yellow_actions500_skel01467-20220329-183215'
import pdb; pdb.set_trace()

# plot parameters
sample_cmap = 'viridis'
model_cmap = 'binary'
sample_min = mean_min = std_min = 0
sample_max = mean_max = 1
std_max = 0.5
ms = 3              # marker size
model_step = 10     # if 1 then plot all models on path
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

# return x, y, c for scatter plot
def get_sample_points(samples, grasp_ixs, world, ai, bald_scores):
    xs = []
    ys = []
    cs = []
    for pi in grasp_ixs:
        plan, _, _ = samples[pi]
        x = plot_action(world,
                    plan[ai],
                    plot=False)
        xs.append(x[0])
        ys.append(x[1])
        cs.append(bald_scores[pi])
    #if len(cs) > 0:
    #    print(max(cs))
    return xs, ys, cs


def get_grasp_indices(samples, grasp):
    if grasp is None:
        return np.arange(len(samples))
    grasp_ixs = []
    for pi, (plan,_,_) in enumerate(samples):
        for action in plan:
            if action.name == 'move_contact':
                grasp_xy = action.args[1].grasp_objF[:2,3]
                if np.allclose(grasp_xy, grasp):
                    grasp_ixs.append(pi)
    return grasp_ixs


def get_num_ax(plan):
    indices = []
    for ai, action in enumerate(plan):
        if action.name == 'move_contact':
            indices.append(ai)
        elif action.name == 'pick':
            if action.args[0].readableName != 'tool':
                indices.append(ai)
        elif action.name == 'move_holding':
            if action.args[0].readableName != 'tool':
                indices.append(ai)
    return indices


def gen_plot(world, mi_min, mi_max, ensembles, skel_num, skel_key, samples, grasp):
    mean_fn = get_model_accuracy_fn(ensembles, 'mean')
    std_fn = get_model_accuracy_fn(ensembles, 'std')

    # calc and normalize bald scores
    bald_scores = np.array([sequential_bald(pddl_plan, ensembles, world) for pddl_plan,_,_ in samples])
    bald_scores = np.clip(bald_scores, sample_min, None) # clip with sample_max?

    dummy_plan = samples[0][0] # all plans in list are the same length
    plt_indices = get_num_ax(dummy_plan)
    fig_path = os.path.join(fig_dir, str(mi_max))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    mci = 0 # if more than 1 move_contact in a plan have to index through different push names

    # get indices that match this grasp
    grasp_ixs = get_grasp_indices(samples, grasp)

    fig, axes = plt.subplots(2, len(plt_indices), figsize=(20,10))
    for axi, ai in enumerate(plt_indices):
        ctype = None
        action = dummy_plan[ai].name
        contact = None
        obj = skel_key.goal_obj
        if samples[0][0][ai].name == 'move_contact':
            ctype = skel_key.ctypes[mci]
            mci += 1
            contact = contact_preds[ctype]
            action += '-'+ctype
        for ax_row in range(2):
            ax = axes[ax_row] if len(plt_indices) == 1 else axes[ax_row, axi]

            ## plot mean/std ##
            x_axes, y_axes = world.get_world_limits(obj, action, contact)
            if ax_row == 0:
                world.vis_dense_plot(action, obj, ax, x_axes, y_axes, mean_min, mean_max, value_fn=mean_fn, cell_width=0.01, grasp=grasp, cmap=model_cmap)
            else:
                world.vis_dense_plot(action, obj, ax, x_axes, y_axes, std_min, std_max, value_fn=std_fn, cell_width=0.01, grasp=grasp, cmap=model_cmap)

            ## plot all skeleton samples and bald scores ##
            xs, ys, cs = get_sample_points(samples, grasp_ixs, world, ai, bald_scores)
            ax.scatter(xs, ys, s=ms, c=cs, cmap=sample_cmap, vmin=sample_min, vmax=sample_max)
            ax.set_title(action)
            ax.set_aspect('equal')

            ## plot all executed plans ##
            dataset = logger.load_trans_dataset('', i=mi_max)
            #if len(list(dataset.ixs.keys())) > 3:
            #    import pdb; pdb.set_trace()
            for ixs in dataset.ixs:
                x, y, action_type, object = dataset[ixs]
                if dummy_plan[ai].name in action_type and object == skel_key.goal_obj:
                    #if ixs > mi_min:
                    #    color = 'b'
                    #else:
                    color = 'g' if y == 1 else 'r'
                    if 'move_contact' in dummy_plan[ai].name and ctype in action_type:
                        grasp_xy = x[4:]
                        if np.allclose(grasp_xy, grasp):
                            ax.plot(*x[2:4], color=color, marker='o', markerfacecolor='none', markersize=ms)
                    elif dummy_plan[ai].name in ['pick', 'move_holding']:
                        ax.plot(*x, color=color, marker='o', markerfacecolor='none', markersize=ms)

    norm = plt.Normalize(sample_min, sample_max)
    sm =  ScalarMappable(norm=norm, cmap=sample_cmap)
    sm.set_array([])
    ax = axes[:] if len(plt_indices) == 1 else axes[:, axi]
    cbar = fig.colorbar(sm, ax=ax)

    skeleton_name = get_skeleton_name(samples[0][0], skel_key)
    if grasp is not None:
        grasp_str = 'p1' if grasp == [.1,0] else 'n1'
    else:
        grasp_str = ''
    fname = '%s_%s.svg' % (skeleton_name, grasp_str)
    plt.savefig(os.path.join(fig_path, fname))
    plt.close()

def main():
    mi_min = 0
    _, txs = logger.get_dir_indices('models')
    for txi in sorted(txs)[::model_step]:
        ensembles, mi_max = logger.load_trans_model(i=txi, ret_i=True)
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
