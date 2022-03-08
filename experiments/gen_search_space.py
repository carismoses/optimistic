import dill as pickle
import matplotlib.pyplot as plt

from pddlstream.language.constants import Certificate

from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen
from experiments.strategies import goals
from tamp.utils import get_simple_state


## Parameters ##
n_plans = 300       # number of plans per contact type
save_path = 'logs/search_space_samples.pkl'
fig_dir = 'logs/search_space_figs/'
goal_obj='yellow_block'
goal_type='push'
contact_types=['push_pull', 'poke']
restart = True
#import pdb; pdb.set_trace()
##

world = ToolsWorld(False, None, contact_types=contact_types)
contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
contact_preds = {}
for contact in contacts:
    if contact[0].type not in contact_preds:
        contact_preds[contact[0].type] = contact[0]
world.disconnect()

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
    pi = counts[ctype]
    while pi < n_plans:
        world = ToolsWorld(False, None, contact_types=[ctype])
        goal, add_to_state = world.generate_goal(goal_obj, goal_type)
        pddl_plan, problem, init_expanded = goals(world, 'opt_no_traj', goal, add_to_state)

        if pddl_plan:
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

