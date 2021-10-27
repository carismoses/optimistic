from shutil import copyfile
import os

from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.downward import task_from_domain_problem, get_action_instances, \
                                            get_problem, parse_action
from pddlstream.language.conversion import Object, transform_plan_args
from pddlstream.algorithms.downward import fact_from_fd, is_applicable
from pddlstream.utils import read

from learning.datasets import model_forward


def postprocess_plan(problem, plan, init_facts_expanded):
    # replace init in problem with init_expanded
    full_init_state = init_facts_expanded.all_facts+init_facts_expanded.preimage_facts
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
        pre = pddl_predicate[0]
        # NOTE: the obj_from_value_expression() treats the 2 arguments
        # to = as other expressions, but they are just pb_robot.bodies so it
        # crashes. Remove as they aren't important predicates anyway
        if pre == '=':
            continue
        simple_pddl_predicate = [pre]
        for arg in pddl_predicate[1:]:
            if isinstance(arg, Object):
                simple_pddl_predicate.append(arg.value)
            else:
                simple_pddl_predicate.append(arg)
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
    goal = world.generate_random_goal() # placeholder/dummy variable
    valid_transition = True
    pddl_state = world.init_state
    trajectory = []
    while valid_transition:
        pddl_state = get_simple_state(pddl_state)
        pddl_plan, expanded_states = world.random_actions(pddl_state)
        if not pddl_plan:
            break
        opt_problem = tuple([*opt_pddl_info, pddl_state+expanded_states, goal]) # used in execute_random()
        task = task_from_problem(opt_problem)
        fd_state = set(task.init)
        ai = 0
        while valid_transition and ai < len(pddl_plan):
            pddl_action = pddl_plan[ai]
            print('Random action: ', pddl_action)
            fd_action = get_fd_action(task, pddl_action)
            pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
            new_pddl_state, new_fd_state, valid_transition = world.transition(pddl_state,
                                                                                fd_state,
                                                                                pddl_action,
                                                                                fd_action)
            trajectory.append((pddl_state, pddl_action, new_pddl_state, valid_transition))
            fd_state = new_fd_state
            pddl_state = new_pddl_state
            pddl_state = get_simple_state(pddl_state)
            ai += 1
    return trajectory

# The only requirements for these files are that action: is before pre: for each pair
# and there is not space between the 2 lines (there can be spaces between the pairs)
# and that there is a single space after the :
def get_add_to_domain_pddl_info(add_to_pddl_path):
    actions =[]
    pres = []
    with open(add_to_pddl_path, 'r') as add_to_pddl_file:
        lines = add_to_pddl_file.readlines()
        for li, line in enumerate(lines):
            if 'action:' in line:
                action = line.replace('action: ', '').replace('\n', '')
                pre = lines[li+1].replace('pre: ', '').replace('\n', '')
                actions.append(action)
                pres.append(pre)
    return actions, pres


# NOTE: This assumes that all add to stream predicates are 2 lines
# and start with :predicate
def get_add_to_streams_pddl_info(add_to_streams_path):
    new_streams = []
    with open(add_to_streams_path, 'r') as add_to_streams_file:
        lines = add_to_streams_file.readlines()
        for li, line in enumerate(lines):
            if ':predicate' in line:
                new_streams.append(line)
                new_streams.append(lines[li+1])
    return new_streams


# NOTE!! This assumes that there are at least 2 preconditions in the action.
# If that is not true then the parenthesis won't work
def add_to_domain(domain_pddl_path, add_to_pddl_path, domain_pddl_path_dir):
    actions, pres = get_add_to_domain_pddl_info(add_to_pddl_path)
    learned_domain_pddl_path = os.path.join(domain_pddl_path_dir, 'tmp', 'learned_domain.pddl')
    os.makedirs(os.path.dirname(learned_domain_pddl_path), exist_ok=True)
    copyfile(domain_pddl_path, learned_domain_pddl_path)
    new_learned_domain_pddl = []
    with open(learned_domain_pddl_path, 'r') as learned_domain_pddl_file:
        lines = learned_domain_pddl_file.readlines()
        found_action = False
        ai = None
        for line in lines:
            new_learned_domain_pddl.append(line)
            if ':action' in line:
                for i, action in enumerate(actions):
                    if action in line:
                        found_action = True
                        ai = i
            if ':precondition' in line and found_action:
                new_learned_domain_pddl.append(pres[i]+'\n')
                found_action = False
            if ':predicates' in line:
                new_learned_domain_pddl.append('\n'.join(pres)+'\n')

    with open(learned_domain_pddl_path, 'w') as learned_domain_pddl_file:
        learned_domain_pddl_file.writelines(new_learned_domain_pddl)

    return learned_domain_pddl_path


# NOTE: This assumes the streams file being added to just has a single close
# parenthesis on the last line
def add_to_streams(streams_pddl_path, add_to_streams_path, domain_pddl_path_dir):
    new_streams = get_add_to_streams_pddl_info(add_to_streams_path)
    learned_streams_pddl_path = os.path.join(domain_pddl_path_dir, 'tmp', 'learned_streams.pddl')
    os.makedirs(os.path.dirname(learned_streams_pddl_path), exist_ok=True)
    new_learned_streams_pddl = []
    if streams_pddl_path:
        raise NotImplementedError('Need to handle case where streams are added to')
        copyfile(streams_pddl_path, learned_streams_pddl_path)
    else:
        new_learned_streams_pddl += ['(define (stream tmp)\n']
        new_learned_streams_pddl += new_streams
        new_learned_streams_pddl += [')\n']

    with open(learned_streams_pddl_path, 'w') as learned_streams_pddl_file:
        learned_streams_pddl_file.writelines(new_learned_streams_pddl)

    return learned_streams_pddl_path

def get_learned_pddl(opt_domain_pddl_path, opt_streams_pddl_path, \
                    add_to_domain_path, add_to_streams_path):
    domain_pddl_path_dir = os.path.dirname(opt_domain_pddl_path)
    learned_domain_pddl_path = add_to_domain(opt_domain_pddl_path, add_to_domain_path, domain_pddl_path_dir)
    learned_streams_pddl_path = add_to_streams(opt_streams_pddl_path, add_to_streams_path, domain_pddl_path_dir)
    return read(learned_domain_pddl_path), read(learned_streams_pddl_path)
