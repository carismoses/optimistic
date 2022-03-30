from copy import copy
import os
import numpy as np
import random
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

import pybullet as p

import pb_robot
from pb_robot.transformations import rotation_from_matrix
from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.language.generator import from_list_fn, from_fn, from_test

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import get_learned_pddl, block_to_urdf, goal_to_urdf, get_simple_state
from domains.tools.primitives import get_free_motion_gen, \
    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_pose_gen_tool, \
    get_tool_grasp_gen, get_block_grasp_gen, get_contact_motion_gen, get_contact_gen, \
    contact_approach_fn, ee_ik


DEBUG = False
MODEL_INPUT_DIMS = {'move_contact-poke': 6, 'move_contact-push_pull': 6, 'pick': 2, 'move_holding': 2}

# TODO: make parent world template class
class ToolsWorld:
    def __init__(self, vis=False, logger=None, objects=['yellow_block', 'blue_block']):
        self.init_objs_pos_xy = {'yellow_block': (0.4, -0.3),
                                'blue_block': (0.3, 0.3),
                                'tool': (0.3, -0.45),
                                'tunnel': (0.3, 0.3)}
        self.blocks = objects

        # goal sampling properties
        self.goal_limits = {'yellow_block': {'min_x': 0.05,
                                            'max_x': 0.85,
                                            'min_y': 0.0,
                                            'max_y':-0.5},
                            'blue_block': {'min_x': 0.05,
                                            'max_x': 0.75,
                                            'min_y': 0.5,
                                            'max_y':0.0},
                            'tool': {'min_x': 0.05,
                                            'max_x': 0.85,
                                            'min_y': 0.5,
                                            'max_y':-0.5}}

        self.use_panda = True
        self.panda = PandaAgent(vis)
        self.panda.plan()
        self.objects, self.obj_init_poses, self.obj_init_state = self.place_objects(place_tunnel=False)
        self.panda_init_state = self.panda.get_init_state()
        self.panda.execute()
        self.place_objects(place_tunnel=True)
        self.panda.plan()
        self.fixed = [self.panda.table]
        self.obj_init_poses['table'] = self.panda.table_pose

        # TODO: test without gravity?? maybe will stop robot from jumping around so much
        p.setGravity(0, 0, -9.81, physicsClientId=self.panda._execution_client_id)

        # get pddl domain description
        self.logger = logger

        # parameters that will be learned
        self.push_goal_radius = 0.05
        self.valid_pick_yellow_radius = 0.51
        self.approx_valid_push_angle = np.pi/32

        # action functions
        self.action_fns = {'pick': self.get_pick_action,
                            'place': self.get_place_action,
                            'move_free': self.get_move_free_action,
                            'move_contact': self.get_move_contact_action,
                            'move_holding': self.get_move_holding_action}


    def get_init_state(self):
        pddl_state = copy(self.obj_init_state)
        pddl_state += copy(self.panda_init_state)
        random.shuffle(pddl_state)
        return pddl_state


    def disconnect(self):
        self.panda.plan()
        pb_robot.utils.disconnect()
        self.panda.execute()
        pb_robot.utils.disconnect()


    def place_object(self, obj_name, path, pos_xy, orn):
        fname = os.path.basename(path)
        copyfile(path, os.path.join('pb_robot/models', fname))
        obj = pb_robot.body.createBody(os.path.join('models', fname))
        z = pb_robot.placements.stable_z(obj, self.panda.table)
        pose = ((*pos_xy, z), orn)
        obj.set_base_link_pose(pose)
        pb_pose = pb_robot.vobj.BodyPose(obj, obj.get_base_link_pose())
        return obj, pb_pose


    # world frame aligns with the robot base
    def place_objects(self, place_tunnel=True):
        pb_objects, orig_poses = {}, {}
        init_state = []

        # tool
        tool_name = 'tool'
        orn = (0,0,0,1)
        tool, pb_pose = self.place_object(tool_name,
                                            'tamp/urdf_models/%s.urdf' % tool_name,
                                            self.init_objs_pos_xy['tool'],
                                            orn)
        pb_objects[tool_name] = tool
        orig_poses[tool_name] = pb_pose
        init_state += [('tool', tool),
                        ('on', tool, self.panda.table),
                        ('atpose', tool, pb_pose),
                        ('pose', tool, pb_pose),
                        ('freeobj', tool)]

        name = 'blue_block'
        color = (0.0, 0.0, 1.0, 1.0)
        urdf_path = 'tamp/urdf_models/%s.urdf' % name
        block_to_urdf(name, urdf_path, color)
        orn = (0,0,0,1)
        block, pb_pose = self.place_object(name,
                                        urdf_path,
                                        self.init_objs_pos_xy[name],
                                        orn)
        pb_objects[name] = block
        orig_poses[name] = pb_pose
        init_state += [('block', block),
                        ('on', block, self.panda.table),
                        ('atpose', block, pb_pose),
                        ('pose', block, pb_pose),
                        ('freeobj', block)]

        # yellow block (heavy --> can only be picked with self.valid_pick_yellow_radius)
        name = 'yellow_block'
        color = (1.0, 1.0, 0.0, 1.0)
        urdf_path = 'tamp/urdf_models/%s.urdf' % name
        block_to_urdf(name, urdf_path, color)
        orn = (0,0,0,1)
        block, pb_pose = self.place_object(name,
                                    urdf_path,
                                    self.init_objs_pos_xy[name],
                                    orn)
        pb_objects[name] = block
        orig_poses[name] = pb_pose
        init_state += [('block', block),
                        ('on', block, self.panda.table),
                        ('atpose', block, pb_pose),
                        ('pose', block, pb_pose),
                        ('freeobj', block)]

        if place_tunnel:
            # tunnel
            tunnel_name = 'tunnel'
            orn = (0,0,0,1)
            tunnel, pb_pose = self.place_object(tunnel_name,
                                'tamp/urdf_models/%s.urdf' % tunnel_name,
                                self.init_objs_pos_xy['tunnel'],
                                orn)
            self.tunnel = tunnel

        return pb_objects, orig_poses, init_state


    # pddl_model_type in {optimistic, learned, opt_no_traj}
    def get_pddl_info(self, pddl_model_type):
        if pddl_model_type == 'learned':
            assert self.logger, 'Must pass in logger to world to plan with learned domain'
        ret_traj = True
        if pddl_model_type == 'opt_no_traj':
            ret_traj = False
            pddl_model_type = 'optimistic'
        learned = True if pddl_model_type == 'learned' else False
        robot = self.panda.planning_robot
        domain_pddl_path = 'domains/tools/domain.pddl'
        streams_pddl_path = 'domains/tools/streams.pddl'
        streams_map = {
            'plan-free-motion': from_fn(get_free_motion_gen(robot,
                                                            self.fixed,
                                                            ret_traj=ret_traj)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(self,
                                                                    robot,
                                                                    self.fixed,
                                                                    ret_traj=ret_traj,
                                                                    learned=learned)),
            'pick-inverse-kinematics': from_fn(get_ik_fn(self,
                                                        robot,
                                                        self.fixed,
                                                        approach_frame='gripper',
                                                        backoff_frame='global',
                                                        learned=learned)),
            'place-inverse-kinematics': from_fn(get_ik_fn(self,
                                                            robot,
                                                            self.fixed,
                                                            approach_frame='global',
                                                            backoff_frame='gripper')),
            'sample-pose-block': from_fn(get_pose_gen_block(self, self.fixed)),
            'sample-pose-tool': from_fn(get_pose_gen_tool(self, self.fixed)),
            'sample-block-grasp': from_list_fn(get_block_grasp_gen(robot)),
            'sample-tool-grasp': from_list_fn(get_tool_grasp_gen(robot)),
            'plan-contact-motion': from_fn(get_contact_motion_gen(self,
                                                                    robot,
                                                                    self.fixed,
                                                                    ret_traj=True,
                                                                    learned=learned)),
            'sample-contact': from_list_fn(get_contact_gen(robot))
            }

        streams_pddl = read(streams_pddl_path)
        domain_pddl = read(domain_pddl_path)
        constant_map = {}
        pddl_info = [domain_pddl, constant_map, streams_pddl, streams_map]
        return pddl_info


    def generate_dummy_goal(self):
        return ('atpose', self.objects['yellow_block'], \
                    pb_robot.vobj.BodyPose(self.objects['yellow_block'], ((0,0,0),(0,0,0,1))))


    # TODO: reimplement way to pass in a goal: object, action, and xy_pos
    def generate_goal(self, goal_xy=None, goal_obj=None):
        if goal_obj is None:
            random_block_i = np.random.randint(len(self.blocks))
            goal_obj = self.blocks[random_block_i]

        object = self.objects[goal_obj]
        init_state = self.get_init_state()
        init_pose = self.get_obj_pose_from_state(object, init_state)

        # select random point on table
        limits = self.goal_limits[goal_obj]

        if goal_xy is None:
            goal_xy = np.array([np.random.uniform(limits['min_x'], limits['max_x']),
                                np.random.uniform(limits['min_y'], limits['max_y'])])

        # add desired pose to state
        goal_pose = ((goal_xy[0], goal_xy[1], init_pose[0][2]), init_pose[1])
        final_pose = pb_robot.vobj.BodyPose(object, goal_pose)
        add_to_state = [('pose', object, final_pose),
                        ('supported', object, final_pose, self.panda.table, self.panda.table_pose),
                        ('atpose', self.panda.table, self.panda.table_pose)]

        # visualize goal patch in pyBullet
        # WARNING: SHOWING THE GOAL MESSES UP THE ROBOT INTERACTIONS AND CAUSES COLLISIONS!
        # Do not use if trying to collect accurate data !!
        if False:
            name = 'goal_patch'
            color = (0.0, 1.0, 0.0, 1.0)
            urdf_path = 'tamp/urdf_models/%s.urdf' % name
            goal_to_urdf(name, urdf_path, color, self.push_goal_radius)
            self.panda.execute()
            self.place_object(name, urdf_path, goal_xy, (0,0,0,1))
            self.panda.plan()
            self.place_object(name, urdf_path, goal_xy, (0,0,0,1))

        goal = ('atpose', object, final_pose)
        return goal, add_to_state


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


    def point_from_world_to_cont(self, point_world, cont_w_tform):
        return np.dot(np.linalg.inv(cont_w_tform), point_world)


    def action_to_vec(self, pddl_action):
        if pddl_action.name == 'move_contact':
            cont = pddl_action.args[5]
            x = np.zeros(MODEL_INPUT_DIMS['move_contact-'+cont.type])

            init_pos_xy = pddl_action.args[3].pose[0][:2]
            goal_pos_xy = pddl_action.args[4].pose[0][:2]
            # the position of the gripper in the tool frame
            gripper_tool_pos_xy = pddl_action.args[1].grasp_objF[:2,3]
            x[:2] = init_pos_xy
            x[2:4] = goal_pos_xy
            x[4:] = gripper_tool_pos_xy
            return x
        elif pddl_action.name == 'pick':
            x = np.zeros(MODEL_INPUT_DIMS[pddl_action.name])
            pick_pose = pddl_action.args[1]
            pick_xy = pick_pose.pose[0][:2]
            x[:] = pick_xy
            return x
        elif pddl_action.name == 'move_holding':
            x = np.zeros(MODEL_INPUT_DIMS[pddl_action.name])
            ee_pose = self.panda.planning_robot.arm.ComputeFK(pddl_action.args[3].configuration)
            x[:] = ee_pose[:2,3]
            return x
        else:
            raise NotImplementedError('No vectorization method for action %s' % pddl_action.name)


    def pred_args_to_vec(self, action_name, args):
        if 'move_contact' in action_name:
            action = Action(name='move_contact', args=(*args,
                                                        None,
                                                        None,
                                                        None))
        elif 'pick' in action_name:
            action = Action(name='pick', args=(args[0],
                                                args[1],
                                                None,
                                                args[2],
                                                None,
                                                None,
                                                None))
        elif 'move_holding' in action_name:
            action = Action(name='move_holding', args=(*args,
                                                        None))
        return self.action_to_vec(action)


    def valid_transition(self, new_pddl_state, pddl_action):
        self.panda.execute()
        valid_transition = True
        if pddl_action.name == 'move_contact':
            # check that block ended up where it was supposed to (with some tolerance)
            goal_pos2 = pddl_action.args[4].pose[0]
            true_pos2 = pddl_action.args[2].get_base_link_pose()[0]
            dist_to_goal = np.linalg.norm(np.subtract(goal_pos2, true_pos2))
            if dist_to_goal > self.push_goal_radius:
                valid_transition = False
        elif pddl_action.name == 'pick':
            # check that if yellow block, was close to base of robot
            if pddl_action.args[0].readableName == 'yellow_block':
                init_pos = pddl_action.args[1].pose[0]
                dist_to_base = np.linalg.norm(init_pos)
                if dist_to_base > self.valid_pick_yellow_radius:
                    valid_transition = False
        elif pddl_action.name == 'move_holding':
            if pddl_action.args[0].readableName == 'yellow_block':
                ee_pose = self.panda.planning_robot.arm.ComputeFK(pddl_action.args[3].configuration)
                xy_pos = ee_pose[:2,3]
                dist_to_base = np.linalg.norm(xy_pos)
                if dist_to_base > self.valid_pick_yellow_radius:
                    valid_transition = False
        self.panda.plan()
        return valid_transition


    def add_goal_text(self, goal):
        self.panda.add_text('Planning for Goal: (%s, %s, (%f, %f, %f))' % \
                                            (goal[0], goal[1], *goal[2].pose[0]),
                        position=(0, -1, 1),
                        size=1.5)


    def block_in_tunnel(self, pos_xy):
        tunnel_pos_xy = self.init_objs_pos_xy['tunnel']
        tunnel_dims = (0.2, 0.09) # from urdf
        min_x = tunnel_pos_xy[0] - tunnel_dims[0]/2
        max_x = tunnel_pos_xy[0] + tunnel_dims[0]/2
        min_y = tunnel_pos_xy[1] - tunnel_dims[1]/2
        max_y = tunnel_pos_xy[1] + tunnel_dims[1]/2

        block_dims = np.array(self.objects['yellow_block'].get_dimensions()[:2])
        half_block = block_dims[0]/2
        # check if any corners are inside of tunnel
        for corner_xy in [np.add(pos_xy, (-half_block, -half_block)),
                            np.add(pos_xy, (-half_block, half_block)),
                            np.add(pos_xy, (half_block, -half_block)),
                            np.add(pos_xy, (half_block, half_block))]:
            x, y = corner_xy
            if x > min_x and x < max_x and y > min_y and y < max_y:
                return True
        return False

    '''
    RANDOM ACTION FUNCTIONS
    '''
    # they return an action and the predicates to add to the state if a valid
    # action is found, else returns None
    def get_pick_action(self, state, streams_map, top_obj=None, top_pose=None, bot_obj=None, \
                        grasp=None, conf1=None, conf2=None, traj=None):
        # check that hand is empty
        if ('handempty',) not in state:
            return None

        # check that obj to be picked is a freeObj and is on another obj and
        # TODO: we should check if bot_obj is None
        if top_obj is None:
            shuffled_objects = self.blocks + self.objects['tool']
            random.shuffle(shuffled_objects)
            for obj in shuffled_objects:
                if ('freeobj', obj) in state:
                    for pred in state:
                        if pred[0] == 'on' and pred[1] == obj:
                            bot_obj = pred[2]
                            top_obj = obj
                            break
        if top_obj is None or bot_obj is None:
            if DEBUG: print('failed: pick, top_obj & bot_obj')
            return None

        # ground top pose
        if top_pose is None:
            for pred in state:
                if pred[0] == 'atpose' and pred[1] == top_obj:
                    top_pose = pred[2]
        if top_pose is None:
            if DEBUG: print('failed: pick, top_pose')
            return None

        # select a random grasp
        if grasp is None:
            if ('block', top_obj) in state:
                grasp = streams_map['sample-block-grasp'](top_obj).next()[0][0]
            elif ('tool', top_obj) in state:
                grasp = streams_map['sample-tool-grasp'](top_obj).next()[0][0]
        if grasp is None:
            if DEBUG: print('failed: pick, grasp')
            return None

        # solve for pick configurations and trajectory
        if conf1 is None or conf2 is None or traj is None:
            stream_result = streams_map['pick-inverse-kinematics'](top_obj, top_pose, grasp).next()
            if len(stream_result) == 0:
                if DEBUG: print('failed: pick, traj')
                return None
        conf1, conf2, traj = stream_result[0]

        #conf1, conf2, traj = pick_params
        pick_action = Action(name='pick',
                        args=(top_obj, top_pose, bot_obj, grasp, conf1, conf2, traj))
        pick_expanded_states = [('pickkin', top_obj, top_pose, grasp, conf1, conf2, traj)]

        # first have to move to initial pick conf
        move_free_action, move_free_expanded_states = self.get_move_free_action(state, streams_map, conf2=conf1)

        return [move_free_action[0], pick_action], pick_expanded_states+move_free_expanded_states


    def get_place_action(self, state, streams_map, top_obj=None, top_pose=None, bot_obj=None, \
                        bot_pose=None, grasp=None, conf1=None, conf2=None, traj=None):
        # check that object is being held
        if top_obj is None:
            for pred in state:
                if pred[0] == 'atgrasp':
                    top_obj = pred[1]
                    grasp = pred[2]
        if top_obj is None or grasp is None:
            if DEBUG: print('failed: place, top_obj & grasp')
            return None

        # ground bottom object (if passed in then use that else use the table)
        if bot_obj is None:
            bot_obj = self.objects['table']
        if bot_obj is None:
            if DEBUG: print('failed: place, bot_obj')
            return None

        if bot_pose is None:
            for pred in state:
                if pred[0] == 'atpose' and pred[1] == bot_obj:
                    bot_pose = pred[2]
        if bot_pose is None:
            if DEBUG: print('failed: place, bot_pose')
            return None

        # ground placement pose
        if top_pose is None:
            if ('block', top_obj) in state:
                top_pose = streams_map['sample-pose-block'](top_obj, bot_obj, bot_pose).next()[0][0]
            elif ('tool', top_obj) in state:
                top_pose = streams_map['sample-pose-tool'](top_obj, bot_obj, bot_pose).next()[0][0]
        if top_pose is None:
            if DEBUG: print('failed: place, top_pose')
            return None

        # ground traj parameters
        if conf1 is None or conf2 is None or traj is None:
            stream_result = streams_map['place-inverse-kinematics'](top_obj, top_pose, grasp).next()
            if len(stream_result) == 0:
                if DEBUG: print('failed: place, traj')
                return None
        conf1, conf2, traj = stream_result[0]

        place_action = Action(name='place',
                        args=(top_obj, top_pose, bot_obj, bot_pose, grasp,
                                conf1, conf2, traj))
        place_expanded_states = [('placekin', top_obj, top_pose, grasp, conf1, conf2, traj),
                                    ('supported', top_obj, top_pose, bot_obj, bot_pose)]

        # first have to move to initial place conf
        move_holding_action, move_holding_expanded_states = self.get_move_holding_action(state, streams_map, conf2=conf1)
        return [move_holding_action[0], place_action], place_expanded_states+move_holding_expanded_states


    def get_move_free_action(self, state, streams_map, conf1=None, conf2=None, traj=None):
        # check that hand is empty
        if ('handempty',) not in state:
            if DEBUG: print('failed: move_free hand not empty')
            return None

        # get current robot config
        if conf1 is None:
            for pred in state:
                if pred[0] == 'atconf':
                    conf1 = pred[1]
        if conf1 is None:
            if DEBUG: print('failed: move_free, conf1')
            return None

        # get a end config
        if conf2 is None:
            init_ee_orn = self.panda.planning_robot.arm.ComputeFK(conf1.configuration)[1]
            conf2 = self.get_random_conf(init_ee_orn)
        if conf2 is None:
            if DEBUG: print('failed: move_free, conf2')
            return None

        # solve for traj
        if traj is None:
            stream_result = streams_map['plan-free-motion'](conf1, conf2).next()
            if len(stream_result) == 0:
                if DEBUG: print('failed: move_free, traj')
                return None
        traj = stream_result[0][0]

        action = Action(name='move_free', args=(conf1, conf2, traj))
        expanded_states = [('freemotion', conf1, conf2, traj)]
        return [action], expanded_states


    def get_move_holding_action(self, state, streams_map, obj=None, grasp=None, conf1=None, \
                                conf2=None, traj=None):
        # check if holding something
        if ('handempty',) in state:
            return None

        # get object being held
        if obj is None:
            for pred in state:
                if pred[0] == 'atgrasp':
                    obj = pred[1]
                    grasp = pred[2]
                if pred[0] == 'atconf':
                    conf1 = pred[1]
        if obj is None or grasp is None or conf1 is None:
            if DEBUG: print('failed: move_holding, obj, grasp, conf1')
            return None

        # get final config
        if conf2 is None:
            init_orn = self.panda.planning_robot.arm.ComputeFK(conf1.configuration)[1]
            conf2 = self.get_random_conf(init_orn)
        if conf2 is None:
            if DEBUG: print('failed: move_holding, conf2')
            return None

        if traj is None:
            stream_result = streams_map['plan-holding-motion'](obj, grasp, conf1, conf2).next()
            if len(stream_result) == 0:
                if DEBUG: print('failed: move_holding, traj')
                return None
        traj = stream_result[0]

        action = Action(name='move_holding', args=(obj, grasp, conf1, conf2, *traj))
        expanded_states = [('holdingmotion', obj, grasp, conf1, conf2, *traj)]
        return [action], expanded_states


    def get_move_contact_action(self, state, streams_map, tool=None, grasp=None, pushed_obj=None, \
                            pose1=None, pose2=None, cont=None, conf1=None, conf2=None, \
                            conf3=None, traj=None):
        # must be holding something
        if ('handempty',) in state:
            return None

        # get held object/tool
        if tool is None:
            for pred in state:
                if pred[0] == 'held':
                    tool = pred[1]
                if pred[0] == 'atgrasp':
                    grasp = pred[2]
        if tool is None or grasp is None:
            if DEBUG: print('failed: move_contact, tool, grasp')
            return None

        # can only push with the tool
        if ('tool', tool) not in state:
            return None

        # get pushed object
        if pushed_obj is None:
            shuffled_objects = self.blocks
            random.shuffle(shuffled_objects)
            for obj in shuffled_objects:
                if ('freeobj', obj) in state and \
                    ('block', obj) in state: # can only push blocks
                    pushed_obj = obj
        if pushed_obj is None:
            if DEBUG: print('failed: move_contact, pushed_obj')
            return None

        # solve for contact
        if cont is None:
            cont = streams_map['sample-contact'](tool, pushed_obj).next()[0][0]
        if cont is None:
            if DEBUG: print('failed: move_contact, cont')
            return None

        # get initial block pose
        if pose1 is None:
            for pred in state:
                if pred[0] == 'atpose' and pred[1] == pushed_obj:
                    pose1 = pred[2]
        if pose1 is None:
            if DEBUG: print('failed: move_contact, pose1')
            return None

        # get final block pose
        if pose2 is None:
            block_name = pushed_obj.readableName
            limits = self.goal_limits[block_name]
            pose2_pos_xy = np.array([np.random.uniform(limits['min_x'], limits['max_x']),
                                    np.random.uniform(limits['min_y'], limits['max_y'])])
            pose2 = pb_robot.vobj.BodyPose(pushed_obj,
                                ((*pose2_pos_xy, pose1.pose[0][2]),
                                pose1.pose[1]))
        if pose2 is None:
            if DEBUG: print('failed: move_contact, pose2')
            return None

        if conf1 is None or conf2 is None or conf3 is None or traj is None:
            stream_result = streams_map['plan-contact-motion'](tool, grasp, pushed_obj, pose1, pose2, cont).next()
            if len(stream_result) == 0:
                if DEBUG: print('failed: move_contact, traj')
                return None
        conf1, conf2, conf3, traj = stream_result[0]

        # first have to move to initial pick conf
        move_contact_action = Action(name='move_contact',
                    args=(tool, grasp, pushed_obj, pose1, pose2, cont, conf1, \
                            conf2, conf3, traj))
        move_contact_expanded_states = [('contactmotion', tool, grasp, pushed_obj, pose1, pose2, \
            cont, conf1, conf2, conf3, traj)]

        # first have to move to initial place conf
        move_holding_action, move_holding_expanded_states = self.get_move_holding_action(state, streams_map, conf2=conf1)
        return [move_holding_action[0], move_contact_action], move_holding_expanded_states+move_contact_expanded_states


    def get_random_conf(self, ee_orn, n_attempts=50):
        ai = 0
        random_conf = None
        while random_conf is None and ai < n_attempts:
            # TODO: make sure this spans the full space
            random_pos = np.array([np.random.uniform(0.05,0.85),
                                np.random.uniform(0.2,-0.5),
                                np.random.uniform(0.01, 0.8)])
            random_conf = self.panda.planning_robot.arm.ComputeIK(pb_robot.geometry.tform_from_pose((random_pos, ee_orn)))
            if random_conf is None:
                continue
            if self.panda.planning_robot.arm.IsCollisionFree(random_conf, obstacles=self.fixed):
                break
            else:
                random_conf = None
        if not random_conf:
            return None
        return pb_robot.vobj.BodyConf(self.panda.planning_robot, random_conf)


    '''
    PLOTTING FUNCTIONS
    '''
    def make_array(self, minv, maxv, step):
        if minv > maxv:
            ar = np.flip(np.arange(maxv, minv+step, step))
            extent = minv+step/2, maxv-step/2
            return ar, extent
        else:
            ar = np.arange(minv, maxv+step, step)
            extent = minv-step/2, maxv+step/2
            return ar, extent


    def get_world_limits(self, obj, action, contact=None):
        minx_w = self.goal_limits[obj]['min_x']
        maxx_w = self.goal_limits[obj]['max_x']
        miny_w = self.goal_limits[obj]['min_y']
        maxy_w = self.goal_limits[obj]['max_y']
        return [minx_w, maxx_w], [miny_w, maxy_w]


    # can visualize tool in world or contact frame
    def vis_tool_ax(self, cont, obj, action, ax, frame='world', color='k'):
        if frame == 'world':
            init_block_pos = self.init_objs_pos_xy[obj]
            # TODO: this assumes that the block is always aligned with the world frame
            block_world = pb_robot.geometry.tform_from_pose(self.obj_init_poses[obj].pose)
            tool_tform = block_world@cont.rel_pose
        elif frame == 'cont':
            init_block_pos = (0., 0.)
            tool_tform = cont.tool_in_cont_tform

        self.plot_block(ax, init_block_pos, color=color)
        self.plot_tool(ax, tool_tform, color)
        ax.set_aspect('equal')
        if frame == 'world':
            limits = self.goal_limits[obj]
            ax.set_xlim([limits['min_x'], limits['max_x']])
            ax.set_ylim([limits['min_y'], limits['max_y']])
        elif frame == 'cont':
            xlimits, ylimits = self.get_world_limits(obj, action, cont)
            ax.set_xlim(xlimits)
            ax.set_ylim(ylimits)


    def vis_dense_plot(self, action, obj, ax, x_range, y_range, vmin, vmax, value_fn=None, cell_width=0.05, grasp=None, cmap='binary'):
        # make 2d arrays of mean and std ensemble predictions
        xs, x_extent = self.make_array(*x_range, cell_width)
        ys, y_extent = self.make_array(*y_range, cell_width)
        values = np.zeros((len(ys), len(xs)))

        for xi, xv in enumerate(xs):
            for yi, yv in enumerate(ys):
                values[yi][xi] = value_fn(self, action, obj, xv, yv, grasp)

        # plot predictions w/ colorbars
        extent = (*x_extent, *y_extent)

        im0 = ax.imshow(values, origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, aspect='equal')
        divider0 = make_axes_locatable(ax)
        cax0 = divider0.append_axes("right", size="10%", pad=0.5)
        cbar0 = plt.colorbar(im0, cax=cax0, format="%.2f")


    def vis_goals(self, ax, goals, planabilities):
        for goal, planable in zip(goals, planabilities):
            goal_pos = goal[2].pose[0][:2]
            color = 'g' if planable else 'r'
            self.plot_block(ax, goal_pos, color=color)

        ax.set_aspect('equal')
        ax.set_xlim([self.min_x, self.max_x])
        ax.set_ylim([self.min_y, self.max_y])


    def plot_block(self, ax, pos, color, linestyle='-'):
        block_dims = np.array(self.objects['yellow_block'].get_dimensions()[:2])
        ax.add_patch(Rectangle(pos-block_dims/2,
                                *block_dims,
                                color=color,
                                fill = False,
                                linestyle=linestyle))


    # 00 is either the world or contact frame depending on what tool_tform
    # is passed in
    def plot_tool(self, ax, tool_tform, color):
        tool_data = p.getVisualShapeData(self.objects['tool'].id, -1)
        tool_base_dims = np.array(tool_data[0][3][:2])
        tool_link0_dims = np.array(tool_data[1][3][:2])

        anchor_base_tool = np.array([*-tool_base_dims/2, 0, 1])
        anchor_base_00 = np.dot(tool_tform, anchor_base_tool)
        tool_pose = pb_robot.geometry.pose_from_tform(tool_tform)
        angle_00, _, _ = rotation_from_matrix(tool_tform)
        deg_angle_00 = angle_00*(180/np.pi)

        # relative pose from center of base to link0
        link0_base_relpos = np.array([-0.185, 0.115]) # from URDF
        anchor_link0_tool = np.array([*(link0_base_relpos-tool_link0_dims/2), 0, 1])
        anchor_link0_00 = np.dot(tool_tform, anchor_link0_tool)

        ax.add_patch(Rectangle(anchor_base_00[:2],
                                *tool_base_dims,
                                angle = deg_angle_00,
                                color = color,
                                fill = False))
        ax.add_patch(Rectangle(anchor_link0_00[:2],
                                *tool_link0_dims,
                                angle = deg_angle_00,
                                color = color,
                                fill = False))


    def vis_dataset(self, ax, action, obj, dataset, grasp=None):
        for x, y in dataset:
            color = 'r' if y == 0 else 'g'
            if action in ['move_holding', 'pick']:
                ax.plot(*x, color+'.')
            elif 'move_contact' in action:
                if np.allclose(x[4:], grasp):
                    # shift to plot all pushes in init block frame
                    init_pos_xy = self.init_objs_pos_xy[obj]
                    rel_pos_xy = np.subtract(x[2:4], x[:2])
                    goal_pos_xy = np.add(init_pos_xy, rel_pos_xy)
                    ax.plot(*goal_pos_xy, color+'.')


    def get_model_inputs(self, tool_approach_pose, goal_pose):
        # for now assume all other blocks are at their initial poses
        num_objects = len(self.objects)
        object_features = np.zeros((num_objects, self.n_of_in))
        for oi, object in enumerate(self.objects.values()):
            object_features[oi] = object.id
        edge_features = np.zeros((num_objects, num_objects, self.n_ef_in))
        for oi, (oi_name, object_i) in enumerate(self.objects.items()):
            for oj, (oj_name, object_j) in enumerate(self.objects.items()):
                if oi == oj:
                    edge_features[oi,oj,:] = np.zeros(self.n_ef_in)
                else:
                    if object_i == self.objects['tool']:
                        obj_i_pos, obj_i_orn = tool_approach_pose
                    else:
                        obj_i_pos, obj_i_orn = self.obj_init_poses[oi_name].pose
                    if object_j == self.objects['tool']:
                        obj_j_pos, obj_j_orn = tool_approach_pose
                    else:
                        obj_j_pos, obj_j_orn = self.obj_init_poses[oj_name].pose
                    rel_pos = np.array(obj_i_pos) - np.array(obj_j_pos)
                    rel_angle = pb_robot.geometry.quat_angle_between(obj_i_orn, obj_j_orn)
                    edge_features[oi,oj,:2] = rel_pos[:2]
                    edge_features[oi,oj,2] = rel_angle
        action_vec = np.zeros(self.n_af_in)
        action_vec[:3] = goal_pose.pose[0]
        action_vec[3:] = goal_pose.pose[1]
        return object_features, edge_features, action_vec

    # the goals file also has planability data currently
    # will have to parse differently when just has x model inputs
    def vis_failed_trajes(self, cont, ax, logger, linestyle='-'):
        init_state = self.get_init_state()
        init_pose = self.get_obj_pose_from_state(self.objects['yellow_block'], init_state)

        datapoints = logger.load_failed_plans()
        #datapoints = datapoints[0]
        for x, y in datapoints:
            of, ef, af = x
            goal_pos_xy = af[:2]
            # see if this contact was used when executing the sample
            pose_j = ((*goal_pos_xy, init_pose[0][2]), init_pose[1])
            goal_pose_j = pb_robot.vobj.BodyPose(self.objects['yellow_block'], pose_j)
            tool_approach_j = contact_approach_fn(self.objects['tool'],
                                                    self.objects['yellow_block'],
                                                    self.obj_init_poses['yellow_block'],
                                                    goal_pose_j,
                                                    cont)
            vof_j, vef_j, va_j = self.get_model_inputs(tool_approach_j, goal_pose_j)
            dist = np.linalg.norm(np.subtract(vef_j, ef))
            if dist < 0.01:
                self.plot_block(ax, goal_pos_xy, 'b', linestyle=linestyle)
