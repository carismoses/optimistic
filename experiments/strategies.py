import time
from copy import copy

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.downward import fact_from_fd, apply_action
from pddlstream.language.constants import Certificate
from tamp.utils import execute_plan, get_simple_state, task_from_problem,   \
                        get_fd_action

MAX_PLAN_LEN = 10


def collect_trajectory(data_collection_mode, world):
    if data_collection_mode == 'random-actions':
        pddl_plan, problem, init_expanded = random_plan(world)
    elif data_collection_mode == 'random-goals-opt':
        pddl_plan, problem, init_expanded = random_goals(world, 'optimistic')
    elif data_collection_mode == 'random-goals-learned':
        pddl_plan, problem, init_expanded = random_goals(world, 'learned')
    elif data_collection_mode == 'sequential':
        pddl_plan, problem, init_expanded = sequential(world)
    #elif data_collection_mode == 'sequential_goals':
    #    pddl_plan, problem, init_expanded =  sequential_goals(world)
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
def random_plan(world):
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
        plan += pddl_actions
        all_expanded_states += expanded_states

        # apply logical state transitions
        problem = tuple([*world.pddl_info, pddl_state+expanded_states, goal])
        task = task_from_problem(problem)
        fd_state = set(task.init)
        for pddl_action in pddl_actions:
            fd_action = get_fd_action(task, pddl_action)
            new_fd_state = copy(fd_state)
            apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
            new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
            fd_state = new_fd_state
            pddl_state = new_pddl_state

    return plan, problem, Certificate(all_expanded_states, [])


# plans to achieve a random goal under the given (optimistic or learned) model
def random_goals(world, plan_model):
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
    return pddl_plan, problem, init_expanded

'''
def sequential(world):
    n_plans = 100
    max_plan_len = 10
    all_plans_info = [random_plan(world, max_plan_len) for _ in range(n_plans)]
    bald_scores = [bald(plan) for plan, _, _ in all_plans_info]
    best_plan, problem, init_expanded = all_plans_info[np.argmax(bald_scores)]
    return execute_plan_wrapper(world, problem, best_plan, init_expanded)

def sequential_goals():
    pass


def bald(plan):
    for state, action in plan:
        if action.name == 'move_contact':
            predictions =
    mp_c1 = torch.mean(predictions, dim=1)
    mp_c0 = torch.mean(1 - predictions, dim=1)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=1)

    bald = m_ent + ent

    return bald



'''
