import dill as pickle
import matplotlib.pyplot as plt
import argparse
import os

from pddlstream.language.constants import Certificate

from experiments.strategies import random_plan
from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen


## file paths ##
skel_fname = 'ss_random_samples.pkl'
fig_dir = 'search_space_figs/'
##

def gen_ss(args):
    # load pre saved plans if restarting
    all_plans = []
    save_path = os.path.join(args.save_path_prefix, skel_fname)
    if args.restart:
        with open(save_path, 'rb') as handle:
            all_plans = pickle.load(handle)

    while len(all_plans) < args.n_plans:
        world = ToolsWorld()
        plan_info = random_plan(world, 'opt_no_traj', ret_states=False)
        if plan_info is not None:
            pddl_plan, problem, init_expanded = plan_info
            init_expanded = Certificate(init_expanded.all_facts, [])
            problem = problem[:3] + problem[4:] # remove stream map (causes pickling issue)
            plan_info = (pddl_plan, problem, init_expanded)
            all_plans.append(plan_info)
            save_path = os.path.join(args.save_path_prefix, skel_fname)
            with open(save_path, 'wb') as handle:
                pickle.dump(all_plans, handle)
        world.disconnect()


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
    parser.add_argument('--save-path-prefix',
                        type=str,
                        help='path to samples file and directory for sample figures')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_ss(args)
