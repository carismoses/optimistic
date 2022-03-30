import dill as pickle
import matplotlib.pyplot as plt
import argparse
import os

from pddlstream.language.constants import Certificate

from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen
from experiments.skeletons import get_skeleton_fns, plan_from_skeleton, \
                                    get_all_skeleton_keys, get_skeleton_name


## file paths ##
save_path = 'logs/ss_skeleton_samples.pkl'
fig_dir = 'logs/search_space_figs/'
##

def gen_ss(args):
    # get all contact predicates (for plotting later)
    world = ToolsWorld(False, None, args.objects)
    contacts_fn = get_contact_gen(world.panda.planning_robot)
    contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
    contact_preds = {}
    for contact in contacts:
        if contact[0].type not in contact_preds:
            contact_preds[contact[0].type] = contact[0]

    all_skeleton_keys = get_all_skeleton_keys(args.objects)

    # load pre saved plans if restarting
    all_plans = {}
    if args.restart:
        with open(save_path, 'rb') as handle:
            all_plans = pickle.load(handle)

    # generate new plans/samples
    skeleton_key = all_skeleton_keys[args.skel_num]
    #for skeleton_key in all_skeleton_keys:
    skeleton_fn, block_name, ctypes = skeleton_key
    print('Generating plans for skeleton: ', skeleton_key)
    plans_from_skeleton(args, block_name, skeleton_fn, ctypes, skeleton_key, all_plans, contact_preds)


def plans_from_skeleton(args, block_name, skeleton_fn, ctypes, skeleton_key, all_plans, contact_preds):
    if skeleton_key in all_plans:
        pi = len(all_plans[skeleton_key])
        print('Already have %i plans for %s' % (pi, skeleton_key))
    else:
        pi = 0
        all_plans[skeleton_key] = []

    while pi < args.n_plans:
        world = ToolsWorld(False, None, [block_name])
        goal_pred, add_to_state = world.generate_goal()
        goal_skeleton = skeleton_fn(world, goal_pred, ctypes)
        plan_info = plan_from_skeleton(goal_skeleton, world, 'opt_no_traj', add_to_state)

        if plan_info is not None:
            pi += 1
            pddl_plan, problem, init_expanded = plan_info
            init_expanded = Certificate(init_expanded.all_facts, [])
            problem = problem[:3] + problem[4:] # remove stream map (causes pickling issue)
            plan_info = (pddl_plan, problem, init_expanded)
            all_plans[skeleton_key].append(plan_info)
            with open(save_path, 'wb') as handle:
                pickle.dump(all_plans, handle)

            # plot all sampled plans
            n_actions = len(all_plans[skeleton_key][0][0]) # all plans in list are the same length
            skeleton_name = get_skeleton_name(pddl_plan, skeleton_key)
            fig_path = os.path.join(fig_dir, skeleton_name)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            mci = 0
            for ai in range(n_actions):
                fig, ax = plt.subplots()
                ctype = None
                if all_plans[skeleton_key][0][0][ai].name == 'move_contact':
                    ctype = ctypes[mci]
                    mci += 1
                for plan,_,_ in all_plans[skeleton_key]:
                    plot_action(world, plan[ai], ax, contact_preds, ctype)
                fname = '%s.png' % plan[ai].name
                plt.savefig(os.path.join(fig_path, fname))
                plt.close()
        world.disconnect()


def plot_action(world, action, ax, contact_preds, ctype, color='k'):
    if action.name == 'move_contact':
        goal_xy = world.action_to_vec(action)[2:4]
        ax.plot(*goal_xy, color=color, marker='.')
        #world.vis_tool_ax(contact_preds[ctype], action.args[2].readableName, action, ax, frame='cont')
    elif action.name == 'pick':
        if action.args[0].readableName != 'tool':
            pick_pos_xy = action.args[1].pose[0][:2]
            ax.plot(*pick_pos_xy, color=color, marker='.')
    elif action.name == 'move_holding':
        if action.args[0].readableName != 'tool':
            x = world.action_to_vec(action)
            ax.plot(*x, color=color, marker='.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-plans',
                        type=int,
                        help='number of plans for generate for each action-object pair')
    parser.add_argument('--objects',
                        nargs='+',
                        type=str,
                        choices=['yellow_block', 'blue_block'],
                        default=['yellow_block', 'blue_block'])
    parser.add_argument('--restart',
                        action='store_true',
                        help='use to restart generating samples from a pervious run')
    parser.add_argument('--skel-num',
                        type=int,
                        help='the index into the list of skeletons to generate samples for')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_ss(args)

'''
all possible skeletons, use for --skel-num input
0: push_skeleton at 0x168bab430>, goal_obj='yellow_block', ctypes=('poke',))
1: push_skeleton at 0x168bab430>, goal_obj='yellow_block', ctypes=('push_pull',))
2: push_skeleton at 0x168bab430>, goal_obj='blue_block', ctypes=('poke',))
3: push_skeleton at 0x168bab430>, goal_obj='blue_block', ctypes=('push_pull',))
4: pick_skeleton at 0x168bab4c0>, goal_obj='yellow_block', ctypes=())
5: pick_skeleton at 0x168bab4c0>, goal_obj='blue_block', ctypes=())
6: push_pick_skeleton at 0x168bab550>, goal_obj='yellow_block', ctypes=('poke',))
7: push_pick_skeleton at 0x168bab550>, goal_obj='yellow_block', ctypes=('push_pull',))
8: push_pick_skeleton at 0x168bab550>, goal_obj='blue_block', ctypes=('poke',))
9: push_pick_skeleton at 0x168bab550>, goal_obj='blue_block', ctypes=('push_pull',))
10: pick_push_skeleton at 0x168bab5e0>, goal_obj='yellow_block', ctypes=('poke',))
11: pick_push_skeleton at 0x168bab5e0>, goal_obj='yellow_block', ctypes=('push_pull',))
12: pick_push_skeleton at 0x168bab5e0>, goal_obj='blue_block', ctypes=('poke',))
13: pick_push_skeleton at 0x168bab5e0>, goal_obj='blue_block', ctypes=('push_pull',))
14: push_push_skeleton at 0x168bab670>, goal_obj='yellow_block', ctypes=('poke', 'poke'))
15: push_push_skeleton at 0x168bab670>, goal_obj='yellow_block', ctypes=('poke', 'push_pull'))
16: push_push_skeleton at 0x168bab670>, goal_obj='yellow_block', ctypes=('push_pull', 'poke'))
17: push_push_skeleton at 0x168bab670>, goal_obj='yellow_block', ctypes=('push_pull', 'push_pull'))
18: push_push_skeleton at 0x168bab670>, goal_obj='blue_block', ctypes=('poke', 'poke'))
19: push_push_skeleton at 0x168bab670>, goal_obj='blue_block', ctypes=('poke', 'push_pull'))
20: push_push_skeleton at 0x168bab670>, goal_obj='blue_block', ctypes=('push_pull', 'poke'))
21: push_push_skeleton at 0x168bab670>, goal_obj='blue_block', ctypes=('push_pull', 'push_pull'))
22: pick_pick_skeleton at 0x168bab700>, goal_obj='yellow_block', ctypes=())
23: pick_pick_skeleton at 0x168bab700>, goal_obj='blue_block', ctypes=())
'''
