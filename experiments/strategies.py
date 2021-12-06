import time
from copy import copy
import numpy as np

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.downward import fact_from_fd, apply_action
from pddlstream.language.constants import Certificate

from tamp.utils import execute_plan, get_simple_state, task_from_problem,   \
                        get_fd_action, postprocess_plan
from learning.utils import model_forward

MAX_PLAN_LEN = 10           # max num of actions in a randomly generated plan
SEQUENTIAL_N_PLANS = 50    # number of plans to select from
EPS = 1e-5


def collect_trajectory(data_collection_mode, world, logger):
    if data_collection_mode == 'random-actions':
        pddl_plan, problem, init_expanded = random_plan(world)
    elif data_collection_mode == 'random-goals-opt':
        pddl_plan, problem, init_expanded = random_goals(world, 'optimistic')
    elif data_collection_mode == 'random-goals-learned':
        pddl_plan, problem, init_expanded = random_goals(world, 'learned')
    elif data_collection_mode == 'sequential-plans':
        pddl_plan, problem, init_expanded = sequential(world, logger, 'plans')
    elif data_collection_mode == 'sequential-goals':
        pddl_plan, problem, init_expanded =  sequential(world, logger, 'goals')
    else:
        raise NotImplementedError('Strategy %s is not implemented' % data_collection_mode)

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
def random_plan(world, ret_states=False):
    print('Planning random actions.')
    if world.use_panda:
        world.panda.add_text('Planning random actions.',
                            position=(0, -1, 1),
                            size=1.5)
    goal = world.generate_random_goal() # dummy variable (TODO: can be None??)
    plan = []
    pddl_state = world.init_state
    all_expanded_states = pddl_state
    while len(plan) < MAX_PLAN_LEN:
        # get random actions
        pddl_state = get_simple_state(pddl_state)
        pddl_actions, expanded_states, actions_found = world.random_actions(pddl_state)
        if not actions_found:
            break
        all_expanded_states += expanded_states

        # apply logical state transitions
        problem = tuple([*world.pddl_info, pddl_state+expanded_states, goal])
        task = task_from_problem(problem)
        fd_state = set(task.init)
        for pddl_action in pddl_actions:
            if ret_states:
                plan += [(pddl_state, pddl_action)]
            else:
                plan += [pddl_action]
            fd_action = get_fd_action(task, pddl_action)
            new_fd_state = copy(fd_state)
            apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
            new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
            fd_state = new_fd_state
            pddl_state = new_pddl_state

    return plan, problem, Certificate(all_expanded_states, [])


# plans to achieve a random goal under the given (optimistic or learned) model
def random_goals(world, plan_model, ret_states=True):
    goal = world.generate_random_goal()
    print('Goal: ', goal)
    print('Planning with %s model'%plan_model)
    if world.use_panda:
        world.panda.add_text('Planning with %s model'%plan_model,
                            position=(0, -1, 1),
                            size=1.5)

    # generate plan (using PDDLStream) to reach random goal
    problem = tuple([*world.pddl_info, world.init_state, goal])
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
    if ret_states:
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


def sequential(world, logger, mode):
    model = logger.load_trans_model(world)
    all_plans_info = []
    while len(all_plans_info) < SEQUENTIAL_N_PLANS:
        if mode == 'plans':
            plan_with_states, problem, init_expanded = random_plan(world, ret_states=True)
        elif mode == 'goals':
            plan_with_states, problem, init_expanded = random_goals(world, 'optimistic', ret_states=True)
        if plan_with_states:
            all_plans_info.append((plan_with_states, problem, init_expanded))
    bald_scores = [sequential_bald(plan, model, world) for plan, _, _ in all_plans_info]
    best_plan_index = np.argmax(bald_scores)
    pddl_plan_with_states, problem, init_expanded = all_plans_info[best_plan_index]
    return [pa for ps, pa in pddl_plan_with_states], problem, init_expanded


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
