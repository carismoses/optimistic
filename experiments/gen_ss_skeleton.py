import dill as pickle
import matplotlib.pyplot as plt

from pddlstream.language.constants import Certificate

from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen
from experiments.skeletons import get_skeletons, plan_from_skeleton


## Parameters ##
n_plans = 300       # number of plans per contact type
save_path = 'logs/ss_skeleton_samples.pkl'
fig_dir = 'logs/search_space_figs_skel/'
actions = ['push-push_pull', 'push-poke']#, 'pick']
objects = ['yellow_block']#, 'blue_block']
restart = False
#import pdb; pdb.set_trace()
##

# get all contact info (for plotting later)
world = ToolsWorld(False, None, actions, objects)
contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
contact_preds = {}
for contact in contacts:
    if contact[0].type not in contact_preds:
        contact_preds[contact[0].type] = contact[0]
contact_types = world.contact_types
world.disconnect()

# if restarting, load all previously saved plans
counts = {ctype: 0 for ctype in contact_types}
type_plans = {ctype: [] for ctype in contact_types}
if restart:
    with open(save_path, 'rb') as handle:
        all_plans = pickle.load(handle)
    for plan_info in all_plans:
        ctype = plan_info[0][-1].args[5].type
        counts[ctype] += 1
        type_plans[ctype] += [plan_info]
else:
    all_plans = []
print(counts)

for ctype in contact_types:
    fig, ax = plt.subplots()
    # get action string for corresponding ctype
    for action in actions:
        if ctype in action:
            contact_action = action
    pi = counts[ctype]
    while pi < n_plans:
        world = ToolsWorld(False, None, [contact_action], objects)
        goal_pred, add_to_state = world.generate_goal()
        skeleton = get_skeletons(world, goal_pred)
        plan_info = plan_from_skeleton(skeleton, world, 'opt_no_traj')

        if plan_info is not None:
            pddl_plan, problem, init_expanded = plan_info
            pi += 1
            init_expanded = Certificate(init_expanded.all_facts, [])
            problem = problem[:3] + problem[4:] # remove stream map
            plan_info = (pddl_plan, problem, init_expanded)
            all_plans.append(plan_info)
            type_plans[ctype].append(plan_info)
            with open(save_path, 'wb') as handle:
                pickle.dump(all_plans, handle)

            # visualize all samples
            fig, ax = plt.subplots()
            for plan,_,_ in type_plans[ctype]:
                goal_xy = world.action_to_vec(plan[-1])
                ax.plot(*goal_xy, 'k.')
            world.vis_tool_ax(contact_preds[ctype], ax, frame='cont')
            plt.savefig(fig_dir+ctype+'.png')
            plt.close()
        world.disconnect()
