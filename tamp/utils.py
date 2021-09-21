from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.downward import task_from_domain_problem, get_action_instances, \
                                            get_problem, parse_action
from pddlstream.language.conversion import Object, transform_plan_args
from pddlstream.algorithms.downward import fact_from_fd, is_applicable

def postprocess_plan(problem, plan, init_facts_expanded):
    # replace init in problem with init_expanded
    full_init_state = list(set(init_facts_expanded.all_facts+init_facts_expanded.preimage_facts))
    pddl_info = problem[:4]
    goal = problem[5]
    problem = tuple([*pddl_info, full_init_state, goal])
    evaluations, goal_exp, domain, externals = parse_problem(problem, unit_costs=True)
    problem = get_problem(evaluations, goal_exp, domain, unit_costs=True)
    task = task_from_domain_problem(domain, problem)
    plan_args = transform_plan_args(plan, Object.from_value)
    action_instances = get_action_instances(task, plan_args)
    return task, action_instances

def get_fd_action(task, pddl_action):
    return get_action_instances(task, transform_plan_args([pddl_action], Object.from_value))[0]

def task_from_problem(problem):
    evaluations, goal_exp, domain, externals = parse_problem(problem, unit_costs=True)
    problem = get_problem(evaluations, goal_exp, domain, unit_costs=True)
    return task_from_domain_problem(domain, problem)


# predicate is strings and ints, but state potentially has pddlstream.language.object.Object
def get_simple_state(state):
    simple_state = []
    for pddl_predicate in state:
        simple_pddl_predicate = [pddl_predicate[0]]
        for arg in pddl_predicate[1:]:
            if isinstance(arg, int):
                simple_pddl_predicate.append(arg)
            elif isinstance(arg, Object):
                simple_pddl_predicate.append(arg.value)
        simple_state.append(tuple(simple_pddl_predicate))
    return simple_state


def execute_plan(world, problem, pddl_plan, init_expanded):
    task, fd_plan = postprocess_plan(problem, pddl_plan, init_expanded)
    fd_state = set(task.init)
    trajectory = []
    ai = 0
    valid_transition = True
    while valid_transition:
        fd_action, pddl_action = fd_plan[ai], pddl_plan[ai]
        assert is_applicable(fd_state, fd_action), 'Something wrong with planner. An invalid action is in the plan.'
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_pddl_state, new_fd_state, valid_transition = world.transition(pddl_state,
                                                                            fd_state,
                                                                            pddl_action,
                                                                            fd_action)
        trajectory.append((pddl_state, pddl_action, new_pddl_state, valid_transition))
        fd_state = new_fd_state
        valid_transition = valid_transition and ai < len(fd_plan)-1 # stop when fail action or at end of trajectory
        ai += 1
    return trajectory


def execute_random(world, opt_pddl_info):
    trajectory = []
    valid_transition = True
    goal = world.generate_random_goal() # placeholder/dummy variable
    pddl_plan, expanded_states = world.random_optimistic_plan()
    opt_problem = tuple([*opt_pddl_info, world.init_state+expanded_states, goal]) # used in execute_random()
    task = task_from_problem(opt_problem)
    fd_state = set(task.init)
    pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
    i = 0
    while valid_transition:
        if i >= len(pddl_plan):
            break
        else:
            pddl_action = pddl_plan[i]
            print('Random action: ', pddl_action)
            fd_action = get_fd_action(task, pddl_action)
            new_pddl_state, new_fd_state, valid_transition = world.transition(pddl_state,
                                                                                fd_state,
                                                                                pddl_action,
                                                                                fd_action)
            trajectory.append((pddl_state, pddl_action, new_pddl_state, valid_transition))
            fd_state = new_fd_state
            pddl_state = new_pddl_state
        i += 1
    return trajectory
