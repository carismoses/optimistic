from copy import copy
import os
import numpy as np
import random
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pybullet as p

import pb_robot

from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.language.generator import from_list_fn, from_fn

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import get_simple_state, get_learned_pddl, block_to_urdf
from domains.tools.primitives import get_free_motion_gen, \
    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_tool_grasp_gen, \
    get_block_grasp_gen, get_contact_motion_gen, get_contact_gen
from domains.tools.add_to_primitives import get_trust_model
from learning.datasets import model_forward


# TODO: make parent world template class
class ToolsWorld:
    @staticmethod
    def init(domain_args, pddl_model_type, vis, logger=None):
        world = ToolsWorld(vis)
        opt_pddl_info, pddl_info = world.get_pddl_info(pddl_model_type, logger)
        return world, opt_pddl_info, pddl_info


    def __init__(self, vis):
        self.use_panda = True
        self.panda = PandaAgent(vis)
        self.panda.plan()
        self.objects, self.orig_poses, self.obj_init_state = self.place_objects()
        self.panda.execute()
        self.place_objects()
        self.panda.plan()
        self.fixed = [self.panda.table]#, self.objects['tunnel']]
        self.obstacles = list(self.objects.values())
        self.init_state = self.get_init_state()

        # TODO: test without gravity?? maybe will stop robot from jumping around so much
        p.setGravity(0, 0, -9.81, physicsClientId=self.panda._execution_client_id)

        # GNN model params
        self.n_of_in = 1
        self.n_ef_in = 3
        self.n_af_in = 7

        self.first_plot = True

    def get_init_state(self):
        pddl_state = self.obj_init_state
        pddl_state += [('table', self.panda.table)]
        pddl_state += self.panda.get_init_state()
        random.shuffle(pddl_state)
        return pddl_state


    # NOTE: this reset looks like it works but then planning fails
    def reset(self):
        def reset_objects():
            for pb_object, object_pose in self.orig_poses.items():
                pb_object.set_base_link_pose(object_pose)
        self.panda.plan()
        reset_blocks()
        self.panda.execute()
        reset_blocks()
        self.panda.reset()


    def disconnect(self):
        self.panda.plan()
        pb_robot.utils.disconnect()
        self.panda.execute()
        pb_robot.utils.disconnect()


    def place_object(self, obj_name, path, pos_xy):
        fname = os.path.basename(path)
        copyfile(path, os.path.join('pb_robot/models', fname))
        obj = pb_robot.body.createBody(os.path.join('models', fname))
        z = pb_robot.placements.stable_z(obj, self.panda.table)
        obj_pose = ((*pos_xy, z), (0., 0., 0., 1.))
        obj.set_base_link_pose(obj_pose)
        pb_pose = pb_robot.vobj.BodyPose(obj, obj.get_base_link_pose())
        return obj, pb_pose


    # world frame aligns with the robot base
    def place_objects(self):
        pb_objects, orig_poses = {}, {}
        init_state = []
        self.obj_init_poses = {}

        # tool
        tool_name = 'tool'
        tool, pose = self.place_object(tool_name, 'tamp/urdf_models/%s.urdf' % tool_name, (0.3, -0.4))
        pb_objects[tool_name] = tool
        orig_poses[tool_name] = pose
        self.obj_init_poses[tool_name] = pose
        init_state += [('tool', tool),
                        ('on', tool, self.panda.table),
                        ('clear', tool), \
                        ('atpose', tool, pose),
                        ('pose', tool, pose),
                        ('freeobj', tool),
                        ('notheavy', tool)]

        # blocks
        blocks = [#('red_block', (1.0, 0.0, 0.0, 1.0), (0.9, 0.0)),
                    #('blue_block', (0.0, 0.0, 1.0, 1.0), (0.3, 0.4)),
                    ('yellow_block', (1.0, 1.0, 0.0, 1.0), (0.4, -0.3))]

        for name, color, pos_xy in blocks:
            urdf_path = 'tamp/urdf_models/%s.urdf' % name
            block_to_urdf(name, urdf_path, color)
            block, pose = self.place_object(name, urdf_path, pos_xy)
            pb_objects[name] = block
            orig_poses[name] = pose
            self.obj_init_poses[name] = pose
            init_state += [('block', block), ('on', block, self.panda.table), ('clear', block), \
                            ('atpose', block, pose), ('pose', block, pose), ('freeobj', block)]

        # tunnel
        '''
        tunnel_name = 'tunnel'
        tunnel, pose = self.place_object(tunnel_name, 'tamp/urdf_models/%s.urdf' % tunnel_name, (0.3, 0.4))
        pb_objects[tunnel_name] = tunnel
        orig_poses[tunnel_name] = pose
        init_state += [('tunnel', tunnel)]#, ('on', tunnel, self.panda.table), ('clear', block), \
                        #('atpose', tunnel, pose), ('pose', tunnel, pose)]
        '''

        return pb_objects, orig_poses, init_state


    def get_pddl_info(self, pddl_model_type, logger):
        add_to_domain_path = 'domains/tools/add_to_domain.pddl'
        add_to_streams_path = 'domains/tools/add_to_streams.pddl'
        robot = self.panda.planning_robot
        opt_domain_pddl_path = 'domains/tools/domain.pddl'
        opt_streams_pddl_path = 'domains/tools/streams.pddl'
        opt_streams_map = {
            'plan-free-motion': from_fn(get_free_motion_gen(robot,
                                                            self.fixed)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot,
                                                                    self.fixed)),
            'pick-inverse-kinematics': from_fn(get_ik_fn(robot,
                                                        self.fixed,
                                                        approach_frame='gripper',
                                                        backoff_frame='global')),
            'place-inverse-kinematics': from_fn(get_ik_fn(robot,
                                                            self.fixed,
                                                            approach_frame='global',
                                                            backoff_frame='gripper')),
            'sample-pose-block': from_fn(get_pose_gen_block(self.fixed)),
            'sample-block-grasp': from_list_fn(get_block_grasp_gen(robot)),
            'sample-tool-grasp': from_list_fn(get_tool_grasp_gen(robot)),
            'plan-contact-motion': from_fn(get_contact_motion_gen(robot,
                                                                    self.fixed)),
            'sample-contact': from_list_fn(get_contact_gen(robot))
            }

        opt_domain_pddl = read(opt_domain_pddl_path)
        opt_streams_pddl = read(opt_streams_pddl_path) if opt_streams_pddl_path else None
        if pddl_model_type == 'optimistic':
            domain_pddl = opt_domain_pddl
            streams_pddl = opt_streams_pddl
            streams_map = opt_streams_map
        elif pddl_model_type == 'learned':
            domain_pddl, streams_pddl = get_learned_pddl(opt_domain_pddl_path,
                                                        opt_streams_pddl_path,
                                                        add_to_domain_path,
                                                        add_to_streams_path)
            streams_map = copy(opt_streams_map)
            streams_map['TrustModel'] = get_trust_model(self, logger)

        constant_map = {}
        opt_pddl_info = [opt_domain_pddl, constant_map, opt_streams_pddl, opt_streams_map]
        pddl_info = [domain_pddl, constant_map, streams_pddl, streams_map]
        return opt_pddl_info, pddl_info


    def generate_random_goal(self, feasible=False, ret_goal_feas=False):
        # select push distance and angle (rad from x axis)
        push_distance = np.random.uniform(0.5, 0.15) # meters
        push_angle = np.random.uniform(0, np.pi/2) # rad
        push_xy = push_distance*np.array([np.cos(push_angle), np.sin(push_angle)])

        # select a random block
        random_object = random.choice(list(self.objects.values()))
        while not ('block', random_object) in self.init_state:
            random_object = random.choice(list(self.objects.values()))
        init_pose = self.get_obj_pose_from_state(random_object, self.init_state)

        # add desired pose to state
        goal_xy = init_pose[0][:2] + push_xy
        goal_pose = ((goal_xy[0], goal_xy[1], init_pose[0][2]), init_pose[1])
        final_pose = pb_robot.vobj.BodyPose(random_object, goal_pose)
        self.init_state += [('pose', random_object, final_pose)]

        # visualize goal patch in pyBullet
        name = 'goal_patch'
        self.panda.execute()
        self.place_object(name, 'tamp/urdf_models/%s.urdf' % name, goal_xy)
        self.panda.plan()
        self.place_object(name, 'tamp/urdf_models/%s.urdf' % name, goal_xy)

        # return goal
        self.goal = ('atpose', random_object, final_pose)
        return self.goal


    # TODO: select a random discrete action then ground it
    def random_actions(self, state):
        # TODO
        return [], []


    def get_obj_pose_from_state(self, object, state):
        for pred in state:
            if pred[0] == 'atpose' and pred[1] == object:
                return pred[2].pose
            if pred[0] == 'atgrasp' and pred[1] == object:
                grasp_objF = pred[2].grasp_objF
                for pred in state:
                    if pred[0] == 'atconf':
                        ee_pose = self.panda.planning_robot.arm.ComputeFK(pred[1].configuration)
                        return pb_robot.geometry.pose_from_tform(ee_pose@np.linalg.inv(grasp_objF))


    def state_to_vec(self, state):
        state = get_simple_state(state)

        num_objects = len(self.objects)
        object_features = np.zeros((num_objects, self.n_of_in))
        for oi, object in enumerate(self.objects.values()):
            object_features[oi] = object.id

        edge_features = np.zeros((num_objects, num_objects, self.n_ef_in))
        for oi, object_i in enumerate(self.objects.values()):
            for oj, object_j in enumerate(self.objects.values()):
                if oi == oj:
                    edge_features[oi,oj,:] = np.zeros(self.n_ef_in)
                else:
                    obj_i_pos, obj_i_orn = self.get_obj_pose_from_state(object_i, state)
                    obj_j_pos, obj_j_orn = self.get_obj_pose_from_state(object_j, state)
                    rel_pos = np.array(obj_i_pos) - np.array(obj_j_pos)
                    rel_angle = pb_robot.geometry.quat_angle_between(obj_i_orn, obj_j_orn)
                    edge_features[oi,oj,:2] = rel_pos[:2]
                    edge_features[oi,oj,2] = rel_angle

        return object_features, edge_features


    def action_to_vec(self, action):
        action_vec = np.zeros(self.n_af_in)
        final_push_pos, final_push_quat = action.args[4].pose
        action_vec[:3] = final_push_pos
        action_vec[3:] = final_push_quat
        return action_vec


    def pred_args_to_action_vec(self, obj1, obj2, pose1, pose2, cont):
        action = Action(name='move_contact', args=(obj1,
                                                    None,
                                                    obj2,
                                                    pose1,
                                                    pose2,
                                                    cont,
                                                    None,
                                                    None,
                                                    None))
        return self.action_to_vec(action)


    def valid_transition(self, new_pddl_state, pddl_action):
        self.panda.execute()
        valid_transition = True
        if pddl_action.name == 'move_contact':
            tol = 0.05
            # check that block ended up where it was supposed to (with some tolerance)
            goal_pos2 = pddl_action.args[4].pose[0]
            true_pos2 = pddl_action.args[2].get_base_link_pose()[0]
            if np.linalg.norm(np.subtract(goal_pos2, true_pos2)) > tol:
                valid_transition = False
        self.panda.plan()
        return valid_transition


    def add_goal_text(self, goal):
        self.panda.add_text('Planning for Goal: (%s, %s, (%f, %f, %f))' % \
                                            (goal[0], goal[1], *goal[2].pose[0]),
                        position=(0, -1, 1),
                        size=1.5)


    def plot_model_accuracy(self, i, model):
        if self.first_plot:
            # get all possible contact points
            contacts_fn = get_contact_gen(self.panda.planning_robot)
            contacts = contacts_fn(self.objects['tool'], self.objects['yellow_block'])

            # set up generators to move robot to different contact points
            grasp_fn = get_tool_grasp_gen(self.panda.planning_robot)
            grasps = grasp_fn(self.objects['tool'])
            contact_motion_fn = get_contact_motion_gen(self.panda.planning_robot, self.fixed)
            goal_pose = self.goal[2]
            # (for debugging acan probably  remove later)
            #pick_fn = get_ik_fn(self.panda.planning_robot,
            #                    self.fixed,
            #                    approach_frame='gripper',
            #                    backoff_frame='global')
            #move_holding_fn = get_holding_motion_gen(self.panda.planning_robot, self.fixed)
            ##

            self.cont_states = []
            for cont in contacts:
                attempts = 0
                while attempts < 50:
                    try:
                        # select a grasp, and move_holding motion
                        grasp_i = np.random.randint(len(grasps))
                        grasp = grasps[grasp_i]

                        # (for debugging acan probably  remove later)
                        # get pick path
                        #pick_init_conf, pick_final_conf, pick_traj = pick_fn(self.objects['tool'],
                        #                                                self.obj_init_poses['tool'],
                        #                                                grasp[0])
                        ##
                        conf_approach, conf_pose2, _ = contact_motion_fn(self.objects['tool'],
                                                            grasp[0],
                                                            self.objects['yellow_block'],
                                                            self.obj_init_poses['yellow_block'],
                                                            goal_pose,
                                                            cont[0])
                        ## also for debugging
                        '''
                        move_holding_traj = move_holding_fn(pick_final_conf, conf_approach, self.objects['tool'], grasp[0])[0]
                        # make and execute actions (pick, move holding)
                        self.panda.execute()
                        self.panda.execution_robot.arm.SetJointValues(pick_init_conf.configuration)
                        pick_action = Action(name='pick', args=(self.objects['tool'],
                                                                self.obj_init_poses['tool'],
                                                                self.panda.table,
                                                                grasp[0],
                                                                pick_init_conf,
                                                                pick_final_conf,
                                                                pick_traj))
                        move_holding_action = Action(name='move_holding', args=(pick_final_conf,
                                                                                conf_approach,
                                                                                self.objects['tool'],
                                                                                grasp[0],
                                                                                move_holding_traj))
                        self.panda.execute_action(pick_action, self.fixed, world_obstacles=self.obstacles)
                        self.panda.execute_action(move_holding_action, self.fixed, world_obstacles=self.obstacles)
                        try:
                            while True:
                                p.stepSimulation(physicsClientId=1)
                                self.panda.execution_robot.arm.SetJointValues(conf_approach.configuration)
                        except KeyboardInterrupt:
                            import pdb; pdb.set_trace()
                            pass
                        '''
                        ##
                        state = [('atpose', self.objects['yellow_block'], self.obj_init_poses['yellow_block']),
                                ('atgrasp', self.objects['tool'], grasp[0]),
                                ('atconf', conf_approach)]
                        self.cont_states.append(state)
                        #tool_base_pos, tool_base_orn = self.get_obj_pose_from_state(self.objects['tool'], state)
                        #print('used later',tool_base_pos)
                        break
                    except:
                        attempts += 1
                        pass

        preds = []
        for state in self.cont_states:
            # run through network
            vof, vef = self.state_to_vec(state)
            va = self.pred_args_to_action_vec(None, None, None, goal_pose, None)
            preds.append([state, model_forward(model, [vof, vef, va]).squeeze()])

        # visualize
        #plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(-.5,1.)
        ax.set_ylim(-1,.5)
        ax.set_aspect('equal')

        # show initial pose and final pose
        block_dims = np.array(self.objects['yellow_block'].get_dimensions()[:2])
        block_initial_pos = np.array(self.obj_init_poses['yellow_block'].pose[0][:2])
        block_final_pos = np.array(goal_pose.pose[0][:2])
        ax.add_patch(Rectangle(block_initial_pos - block_dims/2,
                                *block_dims,
                                color='y'))
        #print(block_initial_pos)
        ax.add_patch(Rectangle(block_final_pos - block_dims/2,
                                *block_dims,
                                color='y'))

        tool_data = p.getVisualShapeData(self.objects['tool'].id, -1)
        tool_base_dims = np.array(tool_data[0][3][:2])
        tool_link0_dims = np.array(tool_data[1][3][:2])
        link0_relpos = np.array([-0.185, 0.115]) # from URDF

        # show different contacts at final pose colored by prediction
        for state, pred in preds:
            #import pdb; pdb.set_trace()
            tool_base_pos, tool_base_orn = self.get_obj_pose_from_state(self.objects['tool'], state)
            angle = pb_robot.geometry.quat_angle_between(tool_base_orn, [0., 0., 0., 1.])
            ax.add_patch(Rectangle(tool_base_pos[:2] - tool_base_dims/2,
                                    *tool_base_dims,
                                    angle = angle,
                                    color = 'k',
                                    alpha = float(pred)))
            ax.add_patch(Rectangle(tool_base_pos[:2] - tool_base_dims/2,
                                    *tool_base_dims,
                                    angle = angle,
                                    color = 'k',
                                    fill = False))
            #print(tool_base_pos[:2], cont[0].rel_pose)
            ax.add_patch(Rectangle(tool_base_pos[:2] + link0_relpos - tool_link0_dims/2,
                                    *tool_link0_dims,
                                    angle = angle,
                                    color = 'k',
                                    alpha = float(pred)))
            ax.add_patch(Rectangle(tool_base_pos[:2] + link0_relpos - tool_link0_dims/2,
                                    *tool_link0_dims,
                                    angle = angle,
                                    color = 'k',
                                    fill = False))
        ax.set_title('Iteration %i' % i)
        #plt.show()
        #plt.close()
        #return fig


# just for testing
if __name__ == '__main__':
    import pybullet as p
    import time
    from pddlstream.utils import INF
    from pddlstream.algorithms.focused import solve_focused
    from tamp.utils import execute_plan, vis_frame

    import pdb; pdb.set_trace()
    vis = True  # set to visualize pyBullet GUI
    world, opt_pddl_info, pddl_info = ToolsWorld.init(None, 'optimistic', vis, logger=None)

    # get initial state and add yellow block goal pose to fluents
    push_distance = 0.15
    init = world.init_state
    initial_point, initial_orn = world.objects['yellow_block'].get_base_link_pose()
    final_pose = (np.add(initial_point, (push_distance, 0., 0.)), initial_orn)
    final_yb_pose = pb_robot.vobj.BodyPose(world.objects['yellow_block'], final_pose)
    goal = ('atpose', world.objects['yellow_block'], final_yb_pose)
    init += [('pose', world.objects['yellow_block'], goal[2])]

    problem = tuple([*pddl_info, init, goal])

    # call planner
    print('Init: ', init)
    print('Goal: ', goal)
    pddl_plan, cost, init_expanded = solve_focused(problem,
                                        success_cost=INF,
                                        max_skeletons=2,
                                        search_sample_ratio=1.0,
                                        max_time=INF,
                                        verbose=False,
                                        unit_costs=True)
    print('Plan: ', pddl_plan)

    # execute plan
    if pddl_plan:
        trajectory, _ = execute_plan(world, problem, pddl_plan, init_expanded)
    else:
        print('No plan found.')
