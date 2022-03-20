import os
import time
from copy import copy
import numpy as np
import dill as pickle
import argparse
import subprocess
import matplotlib.pyplot as plt

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.downward import fact_from_fd, apply_action
from pddlstream.language.constants import Certificate, Action

from tamp.utils import execute_plan, get_simple_state, task_from_problem, \
                        get_fd_action, postprocess_plan, failed_abstract_plan_to_traj, \
                        random_action
from learning.utils import model_forward, add_trajectory_to_dataset
from domains.tools.primitives import get_traj
from domains.tools.world import ToolsWorld


MAX_PLAN_LEN = 6           # max num of actions in a randomly generated plan
EPS = 1e-5


def collect_trajectory_wrapper(args, pddl_model_type, dataset_logger, model_logger=None, \
                            separate_process=False, save_to_dataset=True, goal=None):
    if separate_process:
        # write solver args to file (remove if one is there)
        tmp_dir = 'temp'
        os.makedirs(tmp_dir, exist_ok=True)
        in_pkl = '%s/solver_args.pkl' % tmp_dir
        out_pkl = '%s/solver_solution.pkl' % tmp_dir
        if os.path.exists(in_pkl):
            os.remove(in_pkl)
        with open(in_pkl, 'wb') as handle:
            pickle.dump([args, pddl_model_type, dataset_logger, model_logger, \
                            save_to_dataset, goal], handle)

        # call planner with pickle file
        print('Collecting trajectory.')
        proc = subprocess.run(["python3", "-m", "experiments.strategies", \
                            "--in-pkl", in_pkl, \
                            "--out-pkl", out_pkl], stdout=subprocess.PIPE)

        # read results from pickle file
        with open(out_pkl, 'rb') as handle:
            trajectory = pickle.load(handle)
    else:
        # call planner
        trajectory = collect_trajectory(args, pddl_model_type, dataset_logger, \
                                        model_logger, save_to_dataset, goal)
    return trajectory


def collect_trajectory(args, pddl_model_type, dataset_logger, model_logger, \
                            save_to_dataset, goal):
    # in sequential and learned methods data collection and training happen simultaneously
    if args.data_collection_mode == 'random-goals-learned' and not model_logger:
        model_logger = dataset_logger
    elif args.data_collection_mode in ['sequential-plans', 'sequential-goals', 'random-goals-learned']:
        model_logger = dataset_logger
    world = ToolsWorld(args.vis,
                        model_logger,
                        args.actions,
                        args.objects)

    if args.data_collection_mode == 'random-actions':
        pddl_plan, problem, init_expanded = random_plan(world, 'optimistic')
    elif args.data_collection_mode == 'random-goals-opt':
        goal, add_to_state = world.generate_goal(goal=goal)
        pddl_plan, problem, init_expanded = goals(world, 'optimistic', goal, add_to_state)
    elif args.data_collection_mode == 'random-goals-learned':
        goal, add_to_state = world.generate_goal(goal=goal)
        pddl_plan, problem, init_expanded = goals(world, 'learned', goal, add_to_state)
    elif args.data_collection_mode == 'sequential-plans':
        pddl_plan, problem, init_expanded = sequential(args, world, 'plans', args.n_seq_plans, args.samples_from_file)
    elif args.data_collection_mode == 'sequential-goals':
        pddl_plan, problem, init_expanded =  sequential(args, world, 'goals', args.n_seq_plans, args.samples_from_file)
    else:
        raise NotImplementedError('Strategy %s is not implemented' % args.data_collection_mode)

    trajectory = []
    # is sequential method, have to solve for trajectories
    if 'sequential' in args.data_collection_mode:
        print('Abstract Plan: ', pddl_plan)
        # if plan is to achieve a given goal then only return a low-level plan if it
        # reaches the goal. otherwise can return the plan found until planning failed
        ret_full_plan = 'goals' in args.data_collection_mode
        traj_pddl_plan, add_to_init = solve_trajectories(world,
                                                    pddl_plan,
                                                    ret_full_plan=ret_full_plan)
        failed_pddl_plan = None
        if not traj_pddl_plan:
            add_to_init = []
            failed_pddl_plan = pddl_plan
            print('Planning trajectories failed.')
            if world.use_panda:
                world.panda.add_text('Planning trajectories failed.',
                                    position=(0, -1, 1),
                                    size=1.5)
        init_expanded = Certificate(add_to_init+init_expanded.all_facts, [])
        pddl_plan = traj_pddl_plan
    else:
        # preimage_facts in init_expanded was causing a pickling error, so just use all_facts
        init_expanded = Certificate(init_expanded.all_facts, [])

    if pddl_plan:
        print('Plan: ', pddl_plan)
        if world.use_panda:
            world.panda.add_text('Executing found plan',
                                position=(0, -1, 1),
                                size=1.5)
        trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
    else:
        print('Planning failed.')
        if not pddl_plan and world.use_panda:
            world.panda.add_text('Planning failed.',
                                position=(0, -1, 1),
                                size=1.5)
            time.sleep(.5)

    # if planning low level trajectories failed, also save to dataset
    if 'sequential' in args.data_collection_mode and failed_pddl_plan:
        trajectory = failed_abstract_plan_to_traj(world, problem, failed_pddl_plan, init_expanded)

    if save_to_dataset:
        # add to dataset and save
        added_datapoints = add_trajectory_to_dataset(args.domain, dataset_logger, trajectory, world)

    if 'sequential' in args.data_collection_mode and failed_pddl_plan:
        dataset_logger.add_to_failed_plans(added_datapoints)

    # disconnect from world
    world.disconnect()
    return trajectory


# finds a random plan where all preconditions are met (in optimistic model)
def random_plan(world, pddl_model_type, ret_states=False):
    print('Planning random actions.')
    if world.use_panda:
        world.panda.add_text('Planning random actions.',
                            position=(0, -1, 1),
                            size=1.5)
    goal = world.generate_dummy_goal() # dummy variable (TODO: can be None??)
    states = []
    pddl_plan = []
    pddl_state = world.get_init_state()
    print('Init: ', pddl_state)
    all_expanded_states = []
    problem = None
    pddl_info = world.get_pddl_info(pddl_model_type)
    _, _, _, streams_map = pddl_info
    while len(pddl_plan) < MAX_PLAN_LEN:
        # get random actions
        pddl_state = get_simple_state(pddl_state)
        action_info = random_action(pddl_state, world, streams_map)
        if action_info is None:
            break
        else:
            pddl_actions, expanded_states = action_info
        all_expanded_states += expanded_states

        # apply logical state transitions
        problem = tuple([*pddl_info, pddl_state+expanded_states, goal])
        task = task_from_problem(problem)
        fd_state = set(task.init)
        print('Random actions:', pddl_actions)
        for pddl_action in pddl_actions:
            if ret_states:
                pddl_plan += [(pddl_state, pddl_action)]
            else:
                pddl_plan += [pddl_action]
            fd_action = get_fd_action(task, pddl_action)
            new_fd_state = copy(fd_state)
            apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
            new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
            fd_state = new_fd_state
            pddl_state = new_pddl_state

    return pddl_plan, problem, Certificate(world.get_init_state()+all_expanded_states, [])


# plans to achieve a random goal under the given (optimistic or learned) model
def goals(world, pddl_model_type, goal, add_to_state, ret_states=False):
    print('Goal: ', *goal[:2], goal[2].pose[0][:2])
    print('Planning with %s model'%pddl_model_type)
    if world.use_panda:
        world.panda.add_text('Planning with %s model'%pddl_model_type,
                            position=(0, -1, 1),
                            size=1.5)

    pddl_info = world.get_pddl_info(pddl_model_type)
    problem = tuple([*pddl_info, world.get_init_state()+add_to_state, goal])
    print('Init: ', world.get_init_state()+add_to_state)
    ic = 2 if world.use_panda else 0
    pddl_plan, cost, init_expanded = solve_focused(problem,
                                        success_cost=INF,
                                        max_skeletons=2,
                                        search_sample_ratio=1.0,
                                        max_time=120,
                                        verbose=False,
                                        unit_costs=True,
                                        initial_complexity=ic,
                                        max_iterations=2)
    if pddl_plan and ret_states:
        task, fd_plan = postprocess_plan(problem, pddl_plan, init_expanded)
        fd_state = set(task.init)
        pddl_plan_with_states = []
        if pddl_plan:
            for fd_action, pddl_action in zip(fd_plan, pddl_plan):
                pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
                pddl_plan_with_states += [(pddl_state, pddl_action)]
                new_fd_state = copy(fd_state)
                apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
                new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
                fd_state = new_fd_state
        return pddl_plan_with_states, problem, init_expanded
    return pddl_plan, problem, init_expanded


def sequential(args, world, mode, n_seq_plans, samples_from_file=False):
    model = world.logger.load_trans_model()
    best_plan_info = None
    best_bald_score = float('-inf')
    n_plans_searched = 0
    if samples_from_file:
        with open('logs/ss_skeleton_samples.pkl', 'rb') as handle:
            samples = pickle.load(handle)
        n_seq_plans = len(samples)
    while n_plans_searched < n_seq_plans:
        if samples_from_file:
            pddl_plan, problem, init_expanded = samples[n_plans_searched]
            pddl_info = world.get_pddl_info('opt_no_traj')
            problem = list(problem[:3])+[pddl_info[3]]+list(problem[3:])  # add back stream map
        else:
            if mode == 'plans':
                pddl_plan, problem, init_expanded = random_plan(world, 'opt_no_traj')
            elif mode == 'goals':
                goal, add_to_state = world.generate_goal()
                pddl_plan, problem, init_expanded = goals(world, 'opt_no_traj', goal, add_to_state)
        if pddl_plan:
            n_plans_searched += 1
            bald_score = sequential_bald(pddl_plan, model, world)
            if bald_score >= best_bald_score:
                best_plan_info = pddl_plan, problem, init_expanded
                best_bald_score = bald_score
    return best_plan_info


def sequential_bald(plan, model, world, ret_states=False):
    score = 0
    x = None
    for pddl_action in plan:
        if pddl_action.name == 'move_contact':
            x = world.action_to_vec(pddl_action)
            contact_type = pddl_action.args[5].type
            action = '%s-%s' % ('push', contact_type)
            obj_name = pddl_action.args[2].readableName
            predictions = model_forward(model, x, action, obj_name, single_batch=True)
            mean_prediction = predictions.mean()
            score += mean_prediction*bald(predictions)
        # TODO add 'pick' option
    if ret_states:
        return score, x
    else:
        return score


def bald(predictions):
    mp_c1 = np.mean(predictions)
    mp_c0 = np.mean(1 - predictions)

    m_ent = -(mp_c1 * np.log(mp_c1+EPS) + mp_c0 * np.log(mp_c0+EPS))

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = p_c1 * np.log(p_c1+EPS) + p_c0 * np.log(p_c0+EPS)
    ent = np.mean(ent_per_model)

    bald = m_ent + ent

    return bald


def solve_trajectories(world, pddl_plan, ret_full_plan=False):
    '''
    ret_full_plan: set to True if only want to return a plan if the ENTIRE abstract
                    plan can be grounded (trajectories solved for), else will just
                    return the plan that it was able to ground (potentially shorter
                    than the abstract plan)
    '''
    pddl_plan_traj = []
    robot = world.panda.planning_robot
    obstacles = world.fixed # TODO: add other important things to obstacles (action-type dependent)
    add_to_init = []
    for pddl_action in pddl_plan:
        if pddl_action.name in ['move_free', 'move_holding', 'move_contact']:
            if len(pddl_action.args[-1]) == 0: # check that there isn't already a trajectory
                command, init = get_traj(robot, obstacles, pddl_action)
                if command is None:
                    print('Could not solve for %s trajectories.' % pddl_action.name)
                    if ret_full_plan: # if want to ground the entire plan, then return None if that's not possible
                        return None, None
                    else:
                        return pddl_plan_traj, add_to_init
                pddl_plan_traj += [Action(name=pddl_action.name,
                                args=tuple([a for a in pddl_action.args[:-1]]+[command]))]
                add_to_init.append(init)
            else:
                pddl_plan_traj.append(pddl_action)
        else:
            pddl_plan_traj.append(pddl_action)
    return pddl_plan_traj, add_to_init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-pkl',
                        type=str,
                        required=True,
                        help='pickle file containing planner args')
    parser.add_argument('--out-pkl',
                        type=str,
                        required=True,
                        help='pickle file to write output to')
    args = parser.parse_args()

    # read input args
    with open(args.in_pkl, 'rb') as handle:
        planner_inputs = pickle.load(handle)

    # collect trajectory
    trajectory = collect_trajectory(*planner_inputs)

    # return results
    if os.path.exists(args.out_pkl):
        os.remove(args.out_pkl)
    with open(args.out_pkl, 'wb') as handle:
        pickle.dump(trajectory, handle)
