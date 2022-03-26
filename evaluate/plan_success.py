import numpy as np
import dill as pickle

from experiments.utils import ExperimentLogger
from experiments.skeletons import get_all_skeleton_keys, plan_from_skeleton, \
                                get_skeleton_name
from domains.tools.world import ToolsWorld
from experiments.strategies import sequential_bald
from tamp.utils import execute_plan


# TODO: parallelize (run in own process then when all are done merge pkl files)

def calc_plan_success(args):
    logger = ExperimentLogger(args.exp_path)
    world = ToolsWorld()

    # load goals
    with open(args.goal_path, 'rb') as f:
        goals = pickle.load(f)
    assert len(goals) >= args.n_goals, \
            'cannot generate %i goals from file with %i goals'%(args.n_goals, len(goals))

    for model, mi in logger.get_model_iterator():
        success_data = []
        for gi in range(args.n_goals):
            # generate a goal
            goal_pred, add_to_state = goals[gi]

            # get all possible skeletons
            all_skeleton_keys = get_all_skeleton_keys()

            # generate a space of plans (opt_no_traj) for skeletons
            all_plans = []
            for si, (skel_fn, block_name, ctypes) in all_skeleton_keys:
                if si in args.skel_nums:
                    if goal_pred[1].readableName == block_name:
                        si = 0
                        while si < args.n_samples:
                            goal_skeleton = skel_fn(world, goal_pred, ctypes)
                            plan_info = plan_from_skeleton(goal_skeleton,
                                                            world,
                                                            'opt_no_traj',
                                                            add_to_state)
                            if plan_info is not None:
                                all_plans.append((skel_key, plan_info))
                                si += 1

            # score each plan
            scores = [sequential_bald(plan, model, world) for plan,_,_ in all_plans]

            # select plan and ground trajectories
            skel_key, (pddl_plan, problem, init_expanded) = all_plans[np.argmax(scores)]

            # execute and store result
            trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
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
                        required=True,
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
