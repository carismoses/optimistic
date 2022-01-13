import time
from copy import copy
import numpy as np

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.downward import fact_from_fd, apply_action
from pddlstream.language.constants import Certificate, Action

from tamp.utils import execute_plan, get_simple_state, task_from_problem,   \
                        get_fd_action, postprocess_plan
from learning.utils import model_forward
from domains.tools.primitives import get_traj


MAX_PLAN_LEN = 6           # max num of actions in a randomly generated plan
EPS = 1e-5


def collect_trajectory(data_collection_mode, world, n_seq_plans, progress):
    if data_collection_mode == 'random-actions':
        pddl_plan, problem, init_expanded = random_plan(world, 'optimistic')
    elif data_collection_mode == 'random-goals-opt':
        pddl_plan, problem, init_expanded = goals(world, 'optimistic', 'random')
    elif data_collection_mode == 'random-goals-learned':
        pddl_plan, problem, init_expanded = goals(world, 'learned', 'random')
    elif data_collection_mode == 'sequential-plans':
        pddl_plan, problem, init_expanded = sequential(world, 'plans', n_seq_plans)
    elif data_collection_mode == 'sequential-goals':
        pddl_plan, problem, init_expanded =  sequential(world, 'goals', n_seq_plans)
    elif data_collection_mode == 'engineered-goals-dist':
        pddl_plan, problem, init_expanded = goals(world, 'optimistic', 'engineered-dist', progress=progress)
    elif data_collection_mode == 'engineered-goals-size':
        pddl_plan, problem, init_expanded = goals(world, 'optimistic', 'engineered-size', progress=progress)
    else:
        raise NotImplementedError('Strategy %s is not implemented' % data_collection_mode)

    if 'sequential' in data_collection_mode:
        print('Abstract Plan: ', pddl_plan)
        ret_full_plan = 'goals' in data_collection_mode
        traj_pddl_plan, add_to_init = solve_trajectories(world,
                                                    pddl_plan,
                                                    ret_full_plan=ret_full_plan)
        pddl_plan = traj_pddl_plan
        if not traj_pddl_plan:
            print('Planning trajectories failed.')
            if world.use_panda:
                world.panda.add_text('Planning trajectories failed.',
                                    position=(0, -1, 1),
                                    size=1.5)
            return []
        init_expanded = Certificate(add_to_init+init_expanded.all_facts, [])
    if pddl_plan:
        print('Plan: ', pddl_plan)
        if world.use_panda:
            world.panda.add_text('Executing found plan',
                                position=(0, -1, 1),
                                size=1.5)
        return execute_plan(world, problem, pddl_plan, init_expanded)
    else:
        print('Planning failed.')
        if not pddl_plan and world.use_panda:
            world.panda.add_text('Planning failed.',
                                position=(0, -1, 1),
                                size=1.5)
            time.sleep(.5)
        return []


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
def goals(world, pddl_model_type, goal_type, ret_states=False, progress=None):
    goal, add_to_state = world.generate_goal(goal_type, goal_progress=progress)
    print('Goal: ', goal)
    print('Planning with %s model'%pddl_model_type)
    if world.use_panda:
        world.panda.add_text('Planning with %s model'%pddl_model_type,
                            position=(0, -1, 1),
                            size=1.5)

    # generate plan (using PDDLStream) to reach random goal
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
    model = world.logger.load_trans_model(world)
    best_plan_info = None
    best_bald_score = 0.0
    n_plans_searched = 0
    while n_plans_searched < n_seq_plans:
        if mode == 'plans':
            plan_with_states, problem, init_expanded = random_plan(world, 'opt_no_traj', ret_states=True)
        elif mode == 'goals':
            plan_with_states, problem, init_expanded = goals(world, 'opt_no_traj', 'random', ret_states=True)
        n_plans_searched += 1
        if plan_with_states:
            bald_score = sequential_bald(plan_with_states, model, world)
            if bald_score >= best_bald_score:
                best_plan_info = plan_with_states, problem, init_expanded
    return [pa for ps, pa in best_plan_info[0]], best_plan_info[1], best_plan_info[2]


def sequential_bald(plan, model, world):
    score = 0
    for pddl_state, pddl_action in plan:
        if pddl_action.name == 'move_contact':
            of, ef = world.state_to_vec(pddl_state)
            af = world.action_to_vec(pddl_action)
            predictions = model_forward(model, [of, ef, af], single_batch=True)
            mean_prediction = predictions.mean()
            score += mean_prediction*bald(predictions)
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
