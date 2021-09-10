from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.downward import task_from_domain_problem, get_action_instances, \
                                            get_problem, parse_action
from pddlstream.language.conversion import Object, transform_plan_args

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
def predicate_in_state(predicate, state):
    for pddl_predicate in state:
        simple_pddl_predicate = [pddl_predicate[0]]
        for arg in pddl_predicate[1:]:
            if isinstance(arg, int):
                simple_pddl_predicate.append(arg)
            elif isinstance(arg, Object):
                simple_pddl_predicate.append(arg.value)
        if tuple(simple_pddl_predicate) == predicate:
            return True
    return False
