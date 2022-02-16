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

from tamp.utils import execute_plan, get_simple_state, task_from_problem,   \
                        get_fd_action, postprocess_plan, failed_abstract_plan_to_traj
from learning.utils import model_forward, add_trajectory_to_dataset
from domains.tools.primitives import get_traj
from domains.utils import init_world


MAX_PLAN_LEN = 6           # max num of actions in a randomly generated plan
EPS = 1e-5


def collect_trajectory_wrapper(args, pddl_model_type, dataset_logger, progress, \
                            separate_process=False, model_logger=None, save_to_dataset=True):
    if separate_process:
        # write solver args to file (remove if one is there)
        tmp_dir = 'temp'
        os.makedirs(tmp_dir, exist_ok=True)
        in_pkl = '%s/solver_args.pkl' % tmp_dir
        out_pkl = '%s/solver_solution.pkl' % tmp_dir
        if os.path.exists(in_pkl):
            os.remove(in_pkl)
        with open(in_pkl, 'wb') as handle:
            pickle.dump([args, pddl_model_type, dataset_logger, progress, model_logger, save_to_dataset], handle)

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
        trajectory = collect_trajectory(args, pddl_model_type, \
                                    dataset_logger, progress, model_logger, save_to_dataset)
    return trajectory


def collect_trajectory(args, pddl_model_type, dataset_logger, progress, model_logger, save_to_dataset):
    # in sequential method data collection and training happen simultaneously
    if 'sequential' in args.data_collection_mode:
        model_logger = dataset_logger
    world = init_world(args.domain,
                        args.domain_args,
                        pddl_model_type,
                        args.vis,
                        model_logger)
    world.change_goal_space(progress)

    if args.data_collection_mode == 'random-actions':
        pddl_plan, problem, init_expanded = random_plan(world, 'optimistic')
    elif args.data_collection_mode == 'random-goals-opt':
        goal, add_to_state = world.generate_goal()
        pddl_plan, problem, init_expanded = goals(world, 'optimistic', goal, add_to_state)
    elif args.data_collection_mode == 'random-goals-learned':
        goal, add_to_state = world.generate_goal()
        pddl_plan, problem, init_expanded = goals(world, 'learned', goal, add_to_state)
    elif args.data_collection_mode == 'sequential-plans':
        pddl_plan, problem, init_expanded = sequential(world, 'plans', args.n_seq_plans)
    elif args.data_collection_mode == 'sequential-goals':
        pddl_plan, problem, init_expanded =  sequential(world, 'goals', args.n_seq_plans)
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
            failed_pddl_plan = pddl_plan
            print('Planning trajectories failed.')
            if world.use_panda:
                world.panda.add_text('Planning trajectories failed.',
                                    position=(0, -1, 1),
                                    size=1.5)
        else:
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
        add_trajectory_to_dataset(args.domain, dataset_logger, trajectory, world)

    if 'sequential' in args.data_collection_mode and failed_pddl_plan:
        last_datapoint = dataset_logger.load_trans_dataset('curr')[-1]
        dataset_logger.add_to_goals(last_datapoint, False)
        '''
        if 'goals' in args.data_collection_mode:
            planability = bool(pddl_plan)
            dataset_logger.add_to_goals(goal, planability)
            # visualize goals so far
            #fig, ax = plt.subplots()
            #all_goals, all_planabilities = dataset_logger.load_goals()
            #world.vis_goals(ax, all_goals, all_planabilities)
            #plt.show()
        '''
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
    goal, _ = world.generate_goal() # dummy variable (TODO: can be None??)
    states = []
    pddl_plan = []
    pddl_state = world.get_init_state()
    print('Init: ', pddl_state)
    all_expanded_states = []
    problem = None
    pddl_info = world.get_pddl_info(pddl_model_type)
    while len(pddl_plan) < MAX_PLAN_LEN:
        # get random actions
        pddl_state = get_simple_state(pddl_state)
        pddl_actions, expanded_states, actions_found = world.random_actions(pddl_state, pddl_model_type)
        if not actions_found:
            break
        all_expanded_states += expanded_states

        # apply logical state transitions
        problem = tuple([*pddl_info, pddl_state+expanded_states, goal])
        task = task_from_problem(problem)
        fd_state = set(task.init)
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
    print('Goal: ', goal)
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


def sequential(world, mode, n_seq_plans):
    model = world.logger.load_trans_model()
    best_plan_info = None
    best_bald_score = float('-inf')
    n_plans_searched = 0
    bald_scores, states = [], []
    i = 0
    while n_plans_searched < n_seq_plans:
        # need to return states to calculate the sequential score
        if mode == 'plans':
            plan_with_states, problem, init_expanded = random_plan(world, 'opt_no_traj', ret_states=True)
        elif mode == 'goals':
            goal, add_to_state = world.generate_goal()
            plan_with_states, problem, init_expanded = goals(world, 'opt_no_traj', goal, add_to_state, ret_states=True)
        if plan_with_states:
            n_plans_searched += 1
            bald_score, state = sequential_bald(plan_with_states, model, world, ret_states=True)
            bald_scores.append(bald_score)
            states.append(state)
            if bald_score >= best_bald_score:
                best_i = i
                best_plan_info = plan_with_states, problem, init_expanded
                best_bald_score = bald_score
            i += 1
    return [pa for ps, pa in best_plan_info[0]], best_plan_info[1], best_plan_info[2]


def sequential_bald(plan, model, world, ret_states=False):
    score = 0
    of, ef, af = None, None, None
    for pddl_state, pddl_action in plan:
        if pddl_action.name == 'move_contact':
            of, ef = world.state_to_vec(pddl_state)
            af = world.action_to_vec(pddl_action)
            predictions = model_forward(model, [of, ef, af], single_batch=True)
            mean_prediction = predictions.mean()
            score += mean_prediction*bald(predictions)
    if ret_states:
        return score, [of, ef, af]
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
