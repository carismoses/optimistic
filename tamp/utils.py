from shutil import copyfile
import os
from copy import copy
import numpy as np
import time

import odio_urdf
import pybullet as p
import pb_robot

from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.downward import task_from_domain_problem, get_action_instances, \
                                            get_problem, parse_action, \
                                            fact_from_fd, apply_action, is_applicable
from pddlstream.language.conversion import Object, transform_plan_args
from pddlstream.utils import read


class Contact(object):
    # rel_pose is the pose from body2 (block) to body1 (tool)
    # type in domains.tools.world.CONTACT_TYPES
    # tool_in_cont_tform is the pose of the tool in the contact frame
    def __init__(self, body1, body2, rel_pose, type, tool_in_cont_tform):
        self.body1 = body1
        self.body2 = body2
        self.rel_pose = rel_pose
        self.type = type
        self.tool_in_cont_tform = tool_in_cont_tform


    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)


def pause(client=0):
    print('pausing (make sure you are visualizing the correct robot)')
    try:
        while True:
            p.stepSimulation(physicsClientId=client)
    except KeyboardInterrupt:
        pass


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
        if pre in ['=', 'identical']:
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
    while valid_transition and ai < len(fd_plan):
        fd_action, pddl_action = fd_plan[ai], pddl_plan[ai]
        assert is_applicable(fd_state, fd_action), 'Something wrong with planner. An invalid action is in the plan.'
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_pddl_state, new_fd_state, valid_transition = transition(world,
                                                                    pddl_state,
                                                                    fd_state,
                                                                    pddl_action,
                                                                    fd_action)
        trajectory.append((pddl_state, pddl_action, new_pddl_state, valid_transition))
        fd_state = new_fd_state
        ai += 1
    return trajectory


def failed_abstract_plan_to_traj(world, problem, pddl_plan, init_expanded):
    task, fd_plan = postprocess_plan(problem, pddl_plan, init_expanded)
    fd_state = set(task.init)
    trajectory = []
    for fd_action, pddl_action in zip(fd_plan, pddl_plan):
        assert is_applicable(fd_state, fd_action), 'Something wrong with planner. An invalid action is in the plan.'
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_fd_state = copy(fd_state)
        apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
        new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
        trajectory.append((pddl_state, pddl_action, new_pddl_state, False))
        fd_state = new_fd_state
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
    new_streams += lines
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
        copyfile(streams_pddl_path, learned_streams_pddl_path)
        with open(learned_streams_pddl_path, 'r') as learned_streams_pddl_file:
            lines = learned_streams_pddl_file.readlines()
            new_learned_streams_pddl = lines[:-1]+new_streams+[')\n']
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


def transition(world, pddl_state, fd_state, pddl_action, fd_action):
    if world.use_panda:
        print('Executing action: ', pddl_action)
        world.panda.execute_action(pddl_action, world.fixed, world_obstacles=world.fixed)

    new_fd_state = copy(fd_state)
    apply_action(new_fd_state, fd_action) # apply action (optimistically) in PDDL action model
    new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
    valid_transition = world.valid_transition(new_pddl_state, pddl_action) # check that real state matches opt pddl state
    if valid_transition:
        print('Valid transition.')
    else:
        print('INVALID transitions.')
    return new_pddl_state, new_fd_state, valid_transition # TODO: should really just return valid transition
    # and remove others since aren't used when model type is a classifier


def block_to_urdf(obj_name, urdf_path, color):
    I = 0.001
    side = 0.05
    mass = 0.1
    link_urdf = odio_urdf.Link(obj_name,
                  odio_urdf.Inertial(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Mass(mass),
                      odio_urdf.Inertia(ixx=I,
                                        ixy=0,
                                        ixz=0,
                                        iyy=I,
                                        iyz=0,
                                        izz=I)
                  ),
                  odio_urdf.Collision(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Box(size=(side,
                                            side,
                                            side))
                      )
                  ),
                  odio_urdf.Visual(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Box(size=(side,
                                                side,
                                                side))
                      ),
                      odio_urdf.Material('color',
                                    odio_urdf.Color(rgba=color)
                                    )
                  ))

    block_urdf = odio_urdf.Robot(link_urdf, name=obj_name)
    with open(urdf_path, 'w') as handle:
        handle.write(str(block_urdf))

def goal_to_urdf(name, urdf_path, color, radius):
    length = 0.0001
    link_urdf = odio_urdf.Link(name,
                  odio_urdf.Inertial(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Mass(0),
                      odio_urdf.Inertia(ixx=0,
                                        ixy=0,
                                        ixz=0,
                                        iyy=0,
                                        iyz=0,
                                        izz=0)
                  ),
                  odio_urdf.Collision(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Cylinder(radius=0,
                                            length=0)
                      )
                  ),
                  odio_urdf.Visual(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Cylinder(radius=radius,
                                            length=length)
                      ),
                      odio_urdf.Material('color',
                                    odio_urdf.Color(rgba=color)
                                    )
                  ))

    block_urdf = odio_urdf.Robot(link_urdf, name=name)
    with open(urdf_path, 'w') as handle:
        handle.write(str(block_urdf))

def vis_frame(pb_pose, client, length=0.2, lifeTime=0.):
    pos, quat = pb_pose
    obj_tform = pb_robot.geometry.tform_from_pose(pb_pose)

    for dim in [0,1,2]:
        dim_tform = np.eye(4)
        dim_tform[dim,3] = 1.
        rgb = np.zeros(3)
        rgb[dim] = 1
        p.addUserDebugLine(pos,
                            (obj_tform@dim_tform)[:3,3],
                            lineColorRGB=rgb,
                            lifeTime=lifeTime,
                            physicsClientId=client)
