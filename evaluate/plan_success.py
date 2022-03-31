import numpy as np
import dill as pickle
import argparse

from pddlstream.language.constants import Certificate

from experiments.utils import ExperimentLogger
from experiments.skeletons import get_all_skeleton_keys, plan_from_skeleton, \
                                get_skeleton_name
from domains.tools.world import ToolsWorld
from experiments.strategies import solve_trajectories
from tamp.utils import execute_plan
from learning.utils import model_forward


# TODO: parallelize (run in own process then when all are done merge pkl files)
n_attempts = 10 # number of times to try to plan to achieve a goal for a specific skeleton

def calc_plan_success(args):
    logger = ExperimentLogger(args.exp_path)

    # load goals
    with open(args.goal_path, 'rb') as f:
        goals = pickle.load(f)
    assert len(goals) >= args.n_goals, \
            'cannot generate %i goals from file with %i goals'%(args.n_goals, len(goals))

    _, txs = logger.get_dir_indices('models')
    for mi in sorted(txs)[:150:10]:
        #for model, mi in logger.get_model_iterator():
        model = logger.load_trans_model(i=mi)
        success_data = []
        for gi in range(args.n_goals):
            # generate a goal
            goal_xy = goals[gi][0][2].pose[0][:2]
            goal_obj = goals[gi][0][1].readableName

            # get all possible skeletons
            all_skeleton_keys = get_all_skeleton_keys()

            # generate a space of plans (opt_no_traj) for skeletons
            all_plans = []
            world = ToolsWorld()
            for si, skel_key in enumerate(all_skeleton_keys):
                skel_fn, block_name, ctypes = skel_key
                if si in args.skel_nums:
                    if goal_obj == block_name:
                        ns = 0
                        na = 0
                        while ns < args.n_samples and na < n_attempts:
                            print('--> Planning sample %i for skel %i with obj %s. Attempt %i'%(ns, si, goal_obj, na))
                            goal_pred, add_to_state = world.generate_goal(goal_xy=goal_xy, goal_obj=goal_obj)
                            goal_skeleton = skel_fn(world, goal_pred, ctypes)
                            plan_info = plan_from_skeleton(goal_skeleton,
                                                            world,
                                                            'opt_no_traj',
                                                            add_to_state)
                            na += 1
                            if plan_info is not None:
                                all_plans.append((skel_key, plan_info))
                                ns += 1

            # calculate feasibility scores
            if len(all_plans) == 0:
                success_data.append((None, None, None, False))
            else:
                scores = []
                for skel_key, plan_info in all_plans:
                    pddl_plan, problem, init_expanded = plan_info
                    plan_feas = calc_plan_feasibility(pddl_plan, model, world)
                    scores.append(plan_feas)

                # select plan and ground trajectories
                traj_pddl_plan = None
                while traj_pddl_plan is None and len(scores)>0:
                    max_i = np.argmax(scores)
                    skel_key, (pddl_plan, problem, init_expanded) = all_plans[max_i]
                    traj_pddl_plan, add_to_init = solve_trajectories(world,
                                                            pddl_plan,
                                                            ret_full_plan=True)
                    if traj_pddl_plan is None:
                        print('Grounding trajectory failed')
                        del scores[max_i]
                        del all_plans[max_i]

                # if none of the trajectories can be grounded, goal failed
                if traj_pddl_plan is None:
                    success_data.append((None, None, None, False))
                else:
                    # execute and store result
                    init_expanded = Certificate(add_to_init+init_expanded.all_facts, [])
                    trajectory = execute_plan(world, problem, traj_pddl_plan, init_expanded)
                    success = [t[-1] for t in trajectory]
                    skel_name = get_skeleton_name(pddl_plan, skel_key)
                    success_data.append((skel_name, trajectory, goal_pred, success))

        # save to file (action step, plan and whether or not goal was achieved and which skeleton was used?)
        logger.save_success_data(success_data, mi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        required=True,
                        help='path to evaluate')
    parser.add_argument('--n-goals',
                        type=int,
                        required=True,
                        help='number of goals to evaluate model on for each time step')
    parser.add_argument('--n-samples',
                        type=int,
                        required=True,
                        help='number of plans to generate per skeleton per goal')
    parser.add_argument('--skel-nums',
                        type=int,
                        nargs='+',
                        default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
                        help='which skeletons to use for plans')
    parser.add_argument('--goal-path',
                        type=str,
                        required=True,
                        help='path to pkl file with list of goal predicates')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    calc_plan_success(args)
