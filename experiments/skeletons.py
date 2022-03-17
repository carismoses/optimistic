from copy import copy

from pddlstream.language.constants import Certificate
from pddlstream.algorithms.downward import fact_from_fd, apply_action

from tamp.utils import get_simple_state, task_from_problem, get_fd_action, execute_plan


n_attempts = 30  # number of attempts to ground each action in skeleton

# return a list of skeletons which are functions that take in a goal pose
# and return a skeleton for reaching that goal
def get_skeleton_fns():
    # move_free is generated before picks and move_holdings are generated before places and move_contacts
    # and move_contacts
    push_skeleton = lambda world, goal_pred : \
        [
        # push block
        ('pick', (world.objects['tool'],
                    world.obj_init_poses['tool'],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),
        ('move_contact', (world.objects['tool'],
                    '#g1',
                    goal_pred[1],
                    None,
                    goal_pred[2],
                    None,
                    None,
                    None,
                    None,
                    None))
    ]

    # pick and place block
    pick_skeleton = lambda world, goal_pred : \
        [
        # pick block
        ('pick', (goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),
        ('place', (goal_pred[1],
                    goal_pred[2],
                    world.panda.table,
                    world.obj_init_poses['table'],
                    '#g1',
                    None,
                    None,
                    None))
    ]


    # push then pick and place block
    #move_free, pick_tool, move_holding, move_contact, move_holding, place_tool, move_free, pick_block, move_holding, place_block

    # push block twice
    #move_free, pick_tool, move_holding, move_contact, move_holding, move_contact
    return [push_skeleton, pick_skeleton]


def plan_from_skeleton(skeleton, world, pddl_model_type, add_to_state):
    pddl_state = world.get_init_state()+add_to_state
    pddl_info = world.get_pddl_info(pddl_model_type)
    streams_map = pddl_info[3]
    dummy_goal = world.generate_dummy_goal()
    all_expanded_states = []
    pddl_plan = []
    hash_args = {}
    for action_name, skeleton_args in skeleton:
        # ground reused args and add new ungrounded args
        for ai, skeleton_arg in enumerate(skeleton_args):
            if isinstance(skeleton_arg, str) and skeleton_arg[0] == '#':
                if skeleton_arg in hash_args and hash_args[skeleton_arg] is not None:
                    skeleton_arg = hash_args[skeleton_arg]
                elif skeleton_arg in hash_args and hash_args[skeleton_arg] is None:
                    assert False, 'Action param %s was not grounded by previous action'%skeleton_arg
                else:
                    hash_args[skeleton_arg] = None

        # ground action
        pddl_state = get_simple_state(pddl_state)
        action_fn_kwargs = {'state': pddl_state, 'streams_map': streams_map}
        action_fn = world.action_fns[action_name]
        n_action_args = len(skeleton_args)
        for action_param, skeleton_arg in zip(action_fn.__code__.co_varnames[3:3+n_action_args],
                                                skeleton_args):
            if skeleton_arg in hash_args:
                action_fn_kwargs[action_param] = hash_args[skeleton_arg]
            else:
                action_fn_kwargs[action_param] = skeleton_arg
        for _ in range(n_attempts):
            action_info = action_fn(**action_fn_kwargs)
            if action_info is not None:
                break
        if action_info is None:
            return None

        pddl_actions, expanded_states = action_info

        # get next state
        all_expanded_states += expanded_states
        problem = tuple([*pddl_info, pddl_state+expanded_states, dummy_goal])
        task = task_from_problem(problem)
        fd_state = set(task.init)
        for pddl_action in pddl_actions:
            fd_action = get_fd_action(task, pddl_action)
            new_fd_state = copy(fd_state)
            apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
            new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
            fd_state = new_fd_state
            pddl_state = new_pddl_state

        # save newly grounded args
        for skeleton_arg, grounded_action_arg in zip(skeleton_args, pddl_actions[-1].args):
            if skeleton_arg in hash_args:
                if hash_args[skeleton_arg] is None:
                    hash_args[skeleton_arg] = grounded_action_arg

        pddl_plan += pddl_actions

    init_expanded = Certificate(world.get_init_state()+all_expanded_states, [])
    return pddl_plan, problem, init_expanded


if __name__ == '__main__':
    from domains.tools.world import ToolsWorld

    #import pdb; pdb.set_trace()
    actions = ['push-push_pull', 'push-poke', 'pick']
    objects = ['yellow_block', 'blue_block']
    world = ToolsWorld(False, None, actions, objects)

    push_skeleton = get_skeletons(world)
    pddl_plan, problem, init_expanded = plan_from_skeleton(push_skeleton, world, 'optimistic')
    execute_plan(world, problem, pddl_plan, init_expanded)
