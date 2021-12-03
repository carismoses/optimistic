import time

from pddlstream.utils import INF
from pddlstream.algorithms.focused import solve_focused

from tamp.utils import execute_random, execute_plan


def collect_trajectory(data_collection_mode, world):
    if data_collection_mode == 'random-actions':
        return random_actions(world)
    elif data_collection_mode == 'random-goals-opt':
        return random_goals_opt(world)
    elif data_collection_mode == 'random-goals-learned':
        return random_goals_learned(world)
    #elif data_collection_mode == 'sequential':
    #    return sequential(world)
    #elif data_collection_mode == 'sequential_goals':
    #    return sequential_goals(world)
    else:
        raise NotImplementedError('Strategy %s is not implemented' % data_collection_mode)

def random_actions(world):
    # execute random actions
    print('Planning random actions.')
    if world.use_panda:
        world.panda.add_text('Planning random actions',
                            position=(0, -1, 1),
                            size=1.5)
    return execute_random(world)

def random_goals_opt(world):
    goal = world.generate_random_goal()
    pddl_plan, problem, init_expanded = plan_wrapper(goal, world, 'optimistic')
    if pddl_plan:
        return execute_plan_wrapper(world, problem, pddl_plan, init_expanded)
    return []

def random_goals_learned(world):
    goal = world.generate_random_goal()
    pddl_plan, problem, init_expanded = plan_wrapper(goal, world, 'learned')
    if pddl_plan:
        execute_plan_wrapper(world, problem, pddl_plan, init_expanded)
    return []

def sequential():
    pass

def sequential_goals():
    pass


def plan_wrapper(goal, world, pddl_model_type):
    print('Goal: ', goal)
    print('Planning with %s model'%pddl_model_type)
    if world.use_panda:
        world.panda.add_text('Planning with %s model'%pddl_model_type,
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
    print('Plan: ', pddl_plan)

    if not pddl_plan and world.use_panda:
        world.panda.add_text('Planning failed.',
                            position=(0, -1, 1),
                            size=1.5)
        time.sleep(.5)

    return pddl_plan, problem, init_expanded


def execute_plan_wrapper(world, problem, pddl_plan, init_expanded):
    if world.use_panda:
        world.panda.add_text('Executing found plan',
                            position=(0, -1, 1),
                            size=1.5)
    trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
    return trajectory
