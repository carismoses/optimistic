import dill as pickle
import matplotlib.pyplot as plt
import argparse
import os

from pddlstream.language.constants import Certificate

from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen
from experiments.skeletons import get_skeleton_fns, plan_from_skeleton


## file paths ##
save_path = 'logs/ss_skeleton_samples.pkl'
fig_dir = 'logs/search_space_figs/'
##


def gen_ss(args):
    # get all contact info (for plotting later)
    world = ToolsWorld(False, None, args.actions, args.objects)
    contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
    contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
    contact_preds = {}
    for contact in contacts:
        if contact[0].type not in contact_preds:
            contact_preds[contact[0].type] = contact[0]
    contact_types = world.contact_types
    dummy_goal = world.generate_dummy_goal()

    # get all skeleton names and functions
    all_skeletons = {}
    for skeleton_fn in get_skeleton_fns():
        skeleton_name = '-'.join([a_name for a_name, a_args in skeleton_fn(world, dummy_goal)])
        all_skeletons[skeleton_name] = skeleton_fn
    world.disconnect()

    # load pre saved plans if restarting
    all_plans = {}
    if args.restart:
        with open(save_path, 'rb') as handle:
            all_plans = pickle.load(handle)

    # generate new plans/samples
    for skeleton_name, skeleton_fn in all_skeletons.items():
        if 'move_contact' in skeleton_name:
            for ctype in contact_types:
                plan_key = '%s-%s'%(skeleton_name, ctype)
                plans_from_skeleton(args, skeleton_fn, ctype, plan_key, all_plans, contact_preds)
        else:
            plans_from_skeleton(args, skeleton_fn, None, skeleton_name, all_plans, contact_preds)


def plans_from_skeleton(args, skeleton_fn, ctype, plan_key, all_plans, contact_preds):
    if plan_key in all_plans:
        pi = len(all_plans[plan_key])
        print('Already have %i plans for %s' % (pi, plan_key))
    else:
        pi = 0
        all_plans[plan_key] = []

    if ctype is None:
        world_actions = args.actions
    else:
        for action in args.actions:
            if ctype in action:
                world_actions = [action]

    while pi < args.n_plans:
        world = ToolsWorld(False, None, world_actions, args.objects)
        goal_pred, add_to_state = world.generate_goal()
        goal_skeleton = skeleton_fn(world, goal_pred)
        plan_info = plan_from_skeleton(goal_skeleton, world, 'opt_no_traj', add_to_state)

        if plan_info is not None:
            pi += 1
            pddl_plan, problem, init_expanded = plan_info
            init_expanded = Certificate(init_expanded.all_facts, [])
            problem = problem[:3] + problem[4:] # remove stream map (causes pickling issue)
            plan_info = (pddl_plan, problem, init_expanded)
            all_plans[plan_key].append(plan_info)
            with open(save_path, 'wb') as handle:
                pickle.dump(all_plans, handle)

            # plot all sampled plans
            n_actions = len(all_plans[plan_key][0][0]) # all plans in list are the same length
            fig_path = os.path.join(fig_dir, plan_key)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            for ai in range(n_actions):
                fig, ax = plt.subplots()
                for plan,_,_ in all_plans[plan_key]:
                    plot_action(world, plan[ai], ax, contact_preds, ctype)
                fname = '%s.png' % plan[ai].name
                plt.savefig(os.path.join(fig_path, fname))
                plt.close()
        world.disconnect()


def plot_action(world, action, ax, contact_preds, ctype):
    if action.name == 'move_contact':
        goal_xy = world.action_to_vec(action)
        ax.plot(*goal_xy, 'k.')
        world.vis_tool_ax(contact_preds[ctype], ax, frame='cont')
    elif action.name == 'pick':
        pass
    elif action.name == 'place':
        place_pos_xy = action.args[1].pose[0][:2]
        ax.plot(*place_pos_xy, 'k.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-plans',
                        type=int,
                        help='number of plans for generate for each action-object pair')
    parser.add_argument('--actions',
                        nargs='+',
                        type=str,
                        choices=['push-push_pull', 'push-poke', 'pick'])
    parser.add_argument('--objects',
                        nargs='+',
                        type=str,
                        choices=['yellow_block', 'blue_block'])
    parser.add_argument('--restart',
                        action='store_true',
                        help='use to restart generating samples from a pervious run')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_ss(args)
