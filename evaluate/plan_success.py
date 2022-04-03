import numpy as np
import dill as pickle
import argparse
import matplotlib.pyplot as plt

from pddlstream.language.constants import Certificate

from experiments.utils import ExperimentLogger
from experiments.skeletons import get_all_skeleton_keys, plan_from_skeleton, \
                                get_skeleton_name
from domains.tools.world import ToolsWorld
from experiments.strategies import solve_trajectories, calc_plan_feasibility
from tamp.utils import execute_plan
from learning.utils import model_forward

# TODO: parallelize (run in own process then when all are done merge pkl files)

def gen_feas_push_poses(model):
    n_poses = 1000       # per (block, ctype, grasp)
    n_feas_max = 5   # number of most feasible samples to keep
    n_feas_thresh = 0.7

    world = ToolsWorld()
    push_poses = {}
    for block_name in ['yellow_block', 'blue_block']:
        push_poses[block_name] = {}
        init_xy = world.init_objs_pos_xy[block_name]
        for ctype in ['push_pull', 'poke']:
            push_poses[block_name][ctype] = {}
            for grasp_str in ['p1', 'n1']:
                xs = []
                for _ in range(n_poses):
                    grasp = [.1,0] if grasp_str == 'p1' else [-.1,0]

                    limits = world.goal_limits[block_name]
                    pose2_pos_xy = np.array([np.random.uniform(limits['min_x'], limits['max_x']),
                                            np.random.uniform(limits['min_y'], limits['max_y'])])

                    x = np.array([*init_xy, *pose2_pos_xy, *grasp])
                    xs.append(x)
                xs = np.array(xs)
                all_preds = model_forward(model, xs, 'move_contact-'+ctype, block_name)
                preds = all_preds.mean(axis=0)
                best_ixs = np.argsort(preds)
                # only keep ones above thresh
                top_ixs = np.argwhere(preds > n_feas_thresh).squeeze()
                if top_ixs.size == 0:
                    pos_xys = []
                    small_all_preds = []
                    #print('0', pos_xys)
                elif top_ixs.size > n_feas_max:
                    pos_xys = [xs[ix,:][2:4] for ix in best_ixs[-n_feas_max:]]
                    small_all_preds = [all_preds[:,ix] for ix in best_ixs[-n_feas_max:]]
                    #print('1', pos_xys)
                else:
                    pos_xys = [xs[ix,:][2:4] for ix in top_ixs]
                    small_all_preds = [all_preds[:,ix] for ix in top_ixs]
                    #print('2', pos_xys)
                push_poses[block_name][ctype][grasp_str] = pos_xys
                '''
                fig, ax = plt.subplots(2)
                #print([preds[i] for i in np.argsort(preds)[-10:]])
                print(block_name, ctype, grasp_str)
                for (x,y), apred in zip(pos_xys, small_all_preds):
                    mean = apred.mean()
                    std = apred.std()
                    ax[0].plot(x,y,color=str(mean), marker='.', markeredgecolor='k')
                    ax[1].plot(x,y,color=str(std), marker='.', markeredgecolor='k')
                ax[0].set_title('Mean')
                ax[1].set_title('St Dev')

                plt.show()
                plt.close()
                '''

    world.disconnect()
    #print(push_poses)
    return push_poses


def calc_plan_success(args):
    logger = ExperimentLogger(args.exp_path)

    # load goals
    with open(args.goal_path, 'rb') as f:
        goals = pickle.load(f)

    _, txs = logger.get_dir_indices('models')
    for mi in sorted(txs)[::args.action_step]:
        #for model, mi in logger.get_model_iterator():
        model = logger.load_trans_model(i=mi)

        # calculate samples from high feasibility area for contact models
        feas_push_poses = gen_feas_push_poses(model)

        success_data = []
        for goal in goals:
            #print(p.getBasePositionAndOrientation(0, physicsClientId=0))
            # generate a goal
            goal_xy = goal[0][2].pose[0][:2]
            goal_obj = goal[0][1].readableName

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
                        while ns < args.n_samples:
                            print('--> Planning sample %i for skel %i with obj %s'%(ns, si, goal_obj))

                            goal_pred, add_to_state = world.generate_goal(goal_xy=goal_xy, goal_obj=goal_obj)
                            goal_skeleton = skel_fn(world, goal_pred, ctypes)
                            plan_info = plan_from_skeleton(goal_skeleton,
                                                            world,
                                                            'opt_no_traj',
                                                            add_to_state,
                                                            push_poses=feas_push_poses)

                            if plan_info is not None:
                                all_plans.append((skel_key, plan_info))
                                ns += 1


            # calculate feasibility scores
            if len(all_plans) == 0:
                success_data.append((None, None, None, [False]))
            else:
                scores = []
                for skel_key, plan_info in all_plans:
                    pddl_plan, problem, init_expanded = plan_info
                    #print(pddl_plan)
                    plan_feas = calc_plan_feasibility(pddl_plan, model, world)
                    scores.append(plan_feas)
                    #print(pddl_plan, plan_feas)
                    #input('press enter')


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
                    success_data.append((None, None, goal_pred, [False]))
                else:
                    # execute and store result
                    init_expanded = Certificate(add_to_init+init_expanded.all_facts, [])
                    trajectory = execute_plan(world, problem, traj_pddl_plan, init_expanded)
                    success = [t[-1] for t in trajectory]
                    skel_name = get_skeleton_name(pddl_plan, skel_key)
                    success_data.append((skel_name, trajectory, goal_pred, success))

            world.disconnect()
        # save to file (action step, plan and whether or not goal was achieved and which skeleton was used?)
        logger.save_success_data(success_data, mi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        required=True,
                        help='path to evaluate')
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
    parser.add_argument('--action-step',
                        type=int,
                        default=100)
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    calc_plan_success(args)
