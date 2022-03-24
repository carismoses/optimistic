from copy import copy
import dill as pickle

from pddlstream.language.constants import Certificate
from pddlstream.algorithms.downward import fact_from_fd, apply_action

from domains.tools.world import ToolsWorld
from tamp.utils import get_simple_state, task_from_problem, get_fd_action, execute_plan


n_attempts = 30  # number of attempts to ground each action in skeleton

# return a list of skeletons which are functions that take in a goal pose
# and return a skeleton for reaching that goal
def get_skeleton_fns():
    # move_free is generated before picks and move_holdings are generated before places and move_contacts
    # and move_contacts
    def push_skeleton(world, goal_pred, ctypes=[None]):
        conts = ground_contacts(world, goal_pred[1], ctypes)
        skeleton = [
        # pick tool
        ('pick', (world.objects['tool'],
                    world.obj_init_poses['tool'],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),

        # push block
        ('move_contact', (world.objects['tool'],
                    '#g1',
                    goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    goal_pred[2],
                    conts[0],
                    None,
                    None,
                    None,
                    None))
        ]
        return skeleton


    # pick and place block
    def pick_skeleton(world, goal_pred, ctypes=[]):
        skeleton = [
        # pick block
        ('pick', (goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),

        # place block
        ('place', (goal_pred[1],
                    goal_pred[2],
                    world.panda.table,
                    world.panda.table_pose,
                    '#g1',
                    None,
                    None,
                    None))
        ]
        return skeleton


    # push then pick and place block
    def push_pick_skeleton(world, goal_pred, ctypes=[None]):
        conts = ground_contacts(world, goal_pred[1], ctypes)
        '''
        from copy import copy
        import pb_robot
        init_pose = world.obj_init_poses['blue_block']
        pose = ((init_pose.pose[0][0]+.15, *init_pose.pose[0][1:]), init_pose.pose[1])
        p1 = pb_robot.vobj.BodyPose(world.objects['blue_block'], pose)
        '''
        skeleton = [
        # pick tool
        ('pick', (world.objects['tool'],
                    world.obj_init_poses['tool'],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),

        # push block
        ('move_contact', (world.objects['tool'],
                    '#g1',
                    goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    '#p1',
                    conts[0],
                    None,
                    None,
                    None,
                    None)),

        # place tool
        ('place', (world.objects['tool'],
                    None,
                    world.panda.table,
                    world.panda.table_pose,
                    '#g1',
                    None,
                    None,
                    None)),

        # pick block
        ('pick', (goal_pred[1],
                    '#p1',
                    world.panda.table,
                    '#g2',
                    None,
                    None,
                    None)),

        # place block
        ('place', (goal_pred[1],
                    goal_pred[2],
                    world.panda.table,
                    world.panda.table_pose,
                    '#g2',
                    None,
                    None,
                    None))
        ]
        return skeleton


    # push then pick and place block
    def pick_push_skeleton(world, goal_pred, ctypes=[None]):
        conts = ground_contacts(world, goal_pred[1], ctypes)
        skeleton = [
        # pick block
        ('pick', (goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),

        # place block
        ('place', (goal_pred[1],
                    '#p1',
                    world.panda.table,
                    world.panda.table_pose,
                    '#g1',
                    None,
                    None,
                    None)),

        # pick tool
        ('pick', (world.objects['tool'],
                    world.obj_init_poses['tool'],
                    world.panda.table,
                    '#g2',
                    None,
                    None,
                    None)),

        # push block
        ('move_contact', (world.objects['tool'],
                    '#g2',
                    goal_pred[1],
                    '#p1',
                    None,
                    conts[0],
                    None,
                    None,
                    None,
                    None)),
        ]
        return skeleton


    # push block twice
    def push_push_skeleton(world, goal_pred, ctypes=[None, None]):
        conts = ground_contacts(world, goal_pred[1], ctypes)
        skeleton = [
        # pick tool
        ('pick', (world.objects['tool'],
                    world.obj_init_poses['tool'],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),

        # push block
        ('move_contact', (world.objects['tool'],
                    '#g1',
                    goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    '#p1',
                    conts[0],
                    None,
                    None,
                    None,
                    None)),

        # push block
        ('move_contact', (world.objects['tool'],
                    '#g1',
                    goal_pred[1],
                    '#p1',
                    goal_pred[2],
                    conts[1],
                    None,
                    None,
                    None,
                    None))
        ]
        return skeleton

    # pick and place block twice
    def pick_pick_skeleton(world, goal_pred, ctypes=[]):
        skeleton = [
        # pick block
        ('pick', (goal_pred[1],
                    world.obj_init_poses[goal_pred[1].readableName],
                    world.panda.table,
                    '#g1',
                    None,
                    None,
                    None)),

        # place block
        ('place', (goal_pred[1],
                    '#p1',
                    world.panda.table,
                    world.panda.table_pose,
                    '#g1',
                    None,
                    None,
                    None)),

        # pick block
        ('pick', (goal_pred[1],
                    '#p1',
                    world.panda.table,
                    '#g2',
                    None,
                    None,
                    None)),

        # place block
        ('place', (goal_pred[1],
                    goal_pred[2],
                    world.panda.table,
                    world.panda.table_pose,
                    '#g2',
                    None,
                    None,
                    None))
        ]
        return skeleton
    return [push_skeleton, pick_skeleton, push_pick_skeleton, pick_push_skeleton, \
            push_push_skeleton, pick_pick_skeleton]


def ground_contacts(world, block, ctypes):
    pddl_info = world.get_pddl_info('opt_no_traj')
    streams_map = pddl_info[3]
    conts = []
    for ctype in ctypes:
        if ctype is None:
            cont = None
        else:
            cont = streams_map['sample-contact'](world.objects['tool'],
                                                    block,
                                                    contact_types=[ctype]).next()[0][0]
        conts.append(cont)
    return conts


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


#skel_nums = [0, 1, 2, 3, 6, 7, 8, 9]
def merge_skeletons(skel_nums):
    all_skels_path = 'logs/all_skels/skel'
    final_path = 'logs/ss_skeleton_samples.pkl'

    all_plans = []
    for skel_num in skel_nums:
        skel_path = '%s%i/ss_skeleton_samples.pkl' % (all_skels_path, skel_num)
        with open(skel_path, 'rb') as handle:
            skel_plans = pickle.load(handle)
        key = list(skel_plans.keys())[0]
        print('Adding skel %i: %s' % (skel_num, key))
        all_plans += skel_plans[key]
    print('Merged %s plans' % len(all_plans))
    with open(final_path, 'wb') as handle:
        pickle.dump(all_plans, handle)


def get_all_skeleton_keys(objects=['yellow_block', 'blue_block']):
    # generate a list of all possible skeleton keys
    all_skeleton_keys = []
    for skeleton_fn in get_skeleton_fns():
        for block_name in objects:
            # make a world for this block_name
            dummy_world = ToolsWorld(False, None, [block_name])
            dummy_goal = dummy_world.generate_dummy_goal()
            dummy_skeleton = skeleton_fn(dummy_world, dummy_goal)
            all_ctypes = []
            for a_name, _ in dummy_skeleton:
                if a_name == 'move_contact':
                    if len(all_ctypes) == 0:
                        all_ctypes = [['poke'], ['push_pull']]
                    else:
                        new_all_ctypes = []
                        for ctype in all_ctypes:
                            for new_ctype in ['poke', 'push_pull']:
                                new_all_ctypes.append(ctype+[new_ctype])
                        all_ctypes = new_all_ctypes
            dummy_world.disconnect()
            if len(all_ctypes) > 0:
                for ctype_list in all_ctypes:
                    all_skeleton_keys.append(SkeletonKey(skeleton_fn, block_name, tuple(ctype_list)))
            else:
                all_skeleton_keys.append(SkeletonKey(skeleton_fn, block_name, tuple()))
    #for sk in all_skeleton_keys:
    #    print(sk)
    #print('There are %s potential skeletons' % len(all_skeleton_keys))
    return all_skeleton_keys


def get_skeleton_name(pddl_plan, skeleton_key):
    ctype_str = '_'.join(skeleton_key.ctypes)
    actions_str = '_'.join([name for name, args in pddl_plan])
    if len(ctype_str) > 0:
        return '%s-%s-%s' % (skeleton_key.goal_obj, actions_str, ctype_str)
    else:
        return '%s-%s' % (skeleton_key.goal_obj, actions_str)


if __name__ == '__main__':
    from domains.tools.world import ToolsWorld
    import pdb; pdb.set_trace()

    skel_num = 2

    for _ in range(100):
        objects = ['blue_block']#, 'blue_block']
        world = ToolsWorld(True, None, objects)

        goal_pred, add_to_state = world.generate_goal()

        skeleton_fns = get_skeleton_fns()
        skeleton = skeleton_fns[skel_num](world, goal_pred, ctypes=['poke'])

        plan_info = plan_from_skeleton(skeleton,
                                        world,
                                        'optimistic',
                                        add_to_state)
        if plan_info:
            pddl_plan, problem, init_expanded = plan_info
            trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
            print([t[-1] for t in trajectory])
        world.disconnect()
