import os
import dill as pickle
import numpy as np
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen
from evaluate.plot_value_fns import get_model_accuracy_fn
from experiments.strategies import sequential_bald, calc_plan_feasibility, EPS, bald
from experiments.gen_ss_skeleton import plot_action
from experiments.skeletons import get_skeleton_name
from experiments.utils import ExperimentLogger
from learning.utils import model_forward

###### PARAMETERS
path = 'logs/experiments/yellow500-20220330-120042'
import pdb; pdb.set_trace()

# plot parameters
sample_cmap = 'viridis'
model_cmap = 'binary'
sample_min = 0
sample_max = 1
ms = 3              # marker size
model_step = 1     # if 1 then plot all models on path
######

fig_dir = os.path.join(path, 'figures', 'sample_seq')
logger = ExperimentLogger(path)

world = ToolsWorld()
contacts_fn = get_contact_gen(world.panda.planning_robot)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
contact_preds = {}
for contact in contacts:
    if contact[0].type not in contact_preds:
        contact_preds[contact[0].type] = contact[0]


def action_feas_fn(pddl_plan, ensembles, world, action_name, obj):
    return calc_plan_feasibility([pddl_plan[-1]], ensembles, world)


def plan_feas_fn(pddl_plan, ensembles, world, action_name, obj):
    return calc_plan_feasibility(pddl_plan, ensembles, world)


def bald1_fn(pddl_plan, ensembles, world, action_name, obj):
    x = world.action_to_vec(pddl_plan[-1])
    predictions = model_forward(ensembles, x, action_name, obj, single_batch=True)

    mp_c1 = np.mean(predictions)
    mp_c0 = np.mean(1 - predictions)

    overall_ent = -(mp_c1 * np.log(mp_c1+EPS) + mp_c0 * np.log(mp_c0+EPS))
    #print(overall_ent)
    return overall_ent


def bald2_fn(pddl_plan, ensembles, world, action_name, obj):
    x = world.action_to_vec(pddl_plan[-1])
    predictions = model_forward(ensembles, x, action_name, obj, single_batch=True)

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = (p_c1 * np.log(p_c1+EPS) + p_c0 * np.log(p_c0+EPS))
    ent = -np.mean(ent_per_model)
    #print(ent)
    return ent


def seq_fn(pddl_plan, ensembles, world, action_name, obj):
    score = 0
    for ai in range(1, len(pddl_plan)+1):
        pddl_action = plan[ai-1]
        subplan = plan[:ai]
        plan_feas = calc_plan_feasibility(subplan, ensembles, world)
        x = world.action_to_vec(pddl_action)
        predictions = model_forward(ensembles, x, action_name, obj, single_batch=True)
        score += plan_feas*bald(predictions)
    return score


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


def plot_dataset(ax, dataset, action_name, obj, grasp, ctype):
    ## plot dataset ##
    for ixs in dataset.ixs:
        x, y, action_type, object = dataset[ixs]
        if action_name == action_type and object == obj:
            color = 'g' if y == 1 else 'r'
            if ctype is not None and ctype in action_name:
                grasp_xy = x[4:]
                if np.allclose(grasp_xy, grasp):
                    ax.plot(*x[2:4], color=color, marker='o', markerfacecolor='none', markersize=ms)
            elif action_name in ['pick', 'move_holding']:
                ax.plot(*x, color=color, marker='o', markerfacecolor='none', markersize=ms)


def plot_samples(ax, ai, samples, grasp_ixs, fn, action_name, obj, world, ensembles):
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
        cs.append(fn(plan[:ai+1], ensembles, world, action_name, obj))
    ax.scatter(xs, ys, s=ms, c=cs, cmap=sample_cmap, vmin=sample_min, vmax=sample_max)
    ax.set_title(action_name)
    ax.set_aspect('equal')


def gen_plot(txi, world, ensembles, skel_key, samples, grasp):
    plt_fns = [action_feas_fn, plan_feas_fn, bald1_fn, bald2_fn, seq_fn]
    plt_titles = ['Action Feas', 'Plan Feas', 'BALD Term 1', 'BALD Term 2', 'Seq']
    dummy_plan = samples[0][0] # all plans in list are the same length
    plt_indices = get_num_ax(dummy_plan)
    fig_path = os.path.join(fig_dir, str(txi))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    mci = 0 # if more than 1 move_contact in a plan have to index through different push names

    # get indices that match this grasp
    grasp_ixs = get_grasp_indices(samples, grasp)

    fig, axes = plt.subplots(4, len(plt_indices), figsize=(20,10))
    dataset = logger.load_trans_dataset('', i=txi)
    for axi, ai in enumerate(plt_indices): # iterate through plotable actions
        ctype = None
        action_name = dummy_plan[ai].name
        contact = None
        obj = skel_key.goal_obj
        if samples[0][0][ai].name == 'move_contact':
            ctype = skel_key.ctypes[mci]
            mci += 1
            contact = contact_preds[ctype]
            action_name += '-'+ctype

        for aii, (fn, yl) in enumerate(zip(plot_fns, plt_titles)):
            if len(plt_indices) == 1:
                ax = axes[aii]
                if axi == 0: axes[aii].set_ylabel(yl)
            else:
                ax = axes[aii, 0]
                if axi == 0: axes[aii, 0].set_ylabel(yl)
            plot_samples(ax, ai, samples, grasp_ixs, fn, action_name, obj, world, ensembles)
            plot_dataset(ax, dataset, action_name, obj, grasp, ctype)
            
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
    _, txs = logger.get_dir_indices('models')
    stxs = sorted(txs)
    for txi in stxs[::model_step]:
        ensembles = logger.load_trans_model(i=txi)
        for skel_num in logger.args.skel_nums:
            filename = 'logs/all_skels/skel%i/ss_skeleton_samples.pkl'%skel_num
            with open(filename, 'rb') as handle:
                samples_dict = pickle.load(handle)
            skel_key = list(samples_dict.keys())[0]
            samples = samples_dict[skel_key]
            if len(skel_key.ctypes) > 0: # means a move_contact action is in the skeleton
                for grasp in [[-.1, 0.], [.1, 0.]]:
                    gen_plot(txi, world, ensembles, skel_key, samples, grasp=grasp)
            else:
                gen_plot(txi, world, ensembles, skel_key, samples, grasp=None)

if __name__ == '__main__':
    main()
