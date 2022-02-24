from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import time

import pb_robot
from pddlstream.language.constants import Certificate

from domains.tools.primitives import get_contact_gen
from experiments.strategies import goals, solve_trajectories
from domains.utils import init_world
from tamp.utils import execute_plan, pause, goal_to_urdf, failed_abstract_plan_to_traj


## Parameters ##
# initial block xy in the world fram is (.4, -.3)
goal_min_x, goal_max_x = .4, .85
goal_min_y, goal_max_y = -.25, -.35
n_trials = 100
mode = 'abstract' #'abstract' or 'lowlevel'
vis = True
#import pdb; pdb.set_trace()
################

world = init_world('tools', None, 'optimistic', vis, None)
contacts_fn = get_contact_gen(world.panda.planning_robot)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
cont = contacts[0][0]
world.disconnect()

goal_data = {'feasible': [], 'infeasible': [], 'plan_failure': []}
for _ in range(n_trials):
    world = init_world('tools',
                        None,
                        'optimistic',
                        vis,
                        None)
    init_state = world.get_init_state()
    object = world.objects['yellow_block']
    init_pose = world.get_obj_pose_from_state(object, init_state)
    #goal_xy = np.array([np.random.uniform(goal_min_x, goal_max_x),
    #                    np.random.uniform(goal_min_y, goal_max_y)])
    goal_xy = (0.5, -0.4) #(0.4, -0.35)
    goal_pose = ((goal_xy[0], goal_xy[1], init_pose[0][2]), init_pose[1])
    final_pose = pb_robot.vobj.BodyPose(object, goal_pose)
    table_pose = pb_robot.vobj.BodyPose(world.panda.table, world.panda.table.get_base_link_pose())
    add_to_state = [('pose', object, final_pose),
                        ('supported', object, final_pose, world.panda.table, table_pose)]
    goal = ('atpose', object, final_pose)

    if mode == 'abstract':
        # first plan in abstract then low-level
        pddl_plan, problem, init_expanded = goals(world, 'opt_no_traj', goal, add_to_state)
        if not pddl_plan:
            input('could not find abstract plan')
        print('Solving for trajectories')
        traj_pddl_plan, add_to_init = solve_trajectories(world,
                                                    pddl_plan,
                                                    ret_full_plan=True)

        # execute plan
        if traj_pddl_plan:
            init_expanded = Certificate(add_to_init+init_expanded.all_facts, [])
            trajectory = execute_plan(world, problem, traj_pddl_plan, init_expanded)
            opt_accuracy = trajectory[-1][-1]
            state, final_action, _, _ = trajectory[-1]
            x = world.state_and_action_to_vec(state, final_action)
            print('goal in contact state', x)
            key = 'feasible' if opt_accuracy else 'infeasible'
            goal_data[key].append(x)
        else:
            init_expanded = Certificate(init_expanded.all_facts, [])
            trajectory = failed_abstract_plan_to_traj(world, problem, pddl_plan, init_expanded)
            opt_accuracy = trajectory[-1][-1]
            state, final_action, _, _ = trajectory[-1]
            x = world.state_and_action_to_vec(state, final_action)
            goal_data['plan_failure'].append(x)
            print('Failed to solve for trajectories')
            # show goal and pause
            name = 'goal_patch'
            color = (0.0, 1.0, 0.0, 1.0)
            urdf_path = 'tamp/urdf_models/%s.urdf' % name
            goal_to_urdf(name, urdf_path, color, world.goal_radius)
            world.panda.execute()
            world.place_object(name, urdf_path, goal_xy)
            world.panda.plan()
            world.place_object(name, urdf_path, goal_xy)
            time.sleep(1)

    elif mode == 'lowlevel':
        # plan low-level
        pddl_plan, problem, init_expanded = goals(world, 'optimistic', goal, add_to_state)

        # execute plan
        trajectory = execute_plan(world, problem, pddl_plan, init_expanded)
        opt_accuracy = trajectory[-1][-1]
        state, final_action, _, _ = trajectory[-1]
        x = world.state_and_action_to_vec(state, final_action)
        goal_data['plan_failure'].append(x)

    # print final pose of yellow block
    world.panda.execute()
    print(world.objects['yellow_block'].get_base_link_pose())

    # visualize goal
    fig, ax = plt.subplots()
    world.vis_tool_ax(cont, ax, frame='cont')
    for goal_data_xy in goal_data['feasible']:
        world.plot_block(ax, goal_data_xy, 'g')
    for goal_data_xy in goal_data['infeasible']:
        world.plot_block(ax, goal_data_xy, 'r')
    for goal_data_xy in goal_data['plan_failure']:
        world.plot_block(ax, goal_data_xy, 'b')

    #plt.show()

    world.disconnect()
