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
    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_tool_grasp_gen, \
    get_block_grasp_gen, get_contact_motion_gen, get_contact_gen, contact_approach_fn, ee_ik


N_MC_IN = 2 # input dimensionality for move contact action
CONTACT_TYPES = ['poke', 'push_pull']#, 'hook']


# TODO: make parent world template class
class ToolsWorld:
    @staticmethod
    def init(domain_args, vis, logger=None):
        world = ToolsWorld(vis, logger)
        return world

    @staticmethod
    def get_model_params():
        return N_MC_IN

    def __init__(self, vis, logger, init_objs_pos_xy={}):
        if len(init_objs_pos_xy) == 0:
            init_objs_pos_xy = {'yellow_block': (0.4, -0.25),
                                'blue_block': (0.3, 0.3),
                                'tool': (0.3, -0.45),
                                'tunnel': (0.3, 0.3)}
        self.init_objs_pos_xy = init_objs_pos_xy
        self.use_panda = True
        self.panda = PandaAgent(vis)
        self.panda.plan()
        self.objects, self.orig_poses, self.obj_init_state = self.place_objects(place_tunnel=False)
        self.panda_init_state = self.panda.get_init_state()
        self.panda.execute()
        self.place_objects(place_tunnel=True)
        self.panda.plan()
        self.fixed = [self.panda.table]

        # TODO: test without gravity?? maybe will stop robot from jumping around so much
        p.setGravity(0, 0, -9.81, physicsClientId=self.panda._execution_client_id)

        # get pddl domain description
        self.logger = logger

        # goal sampling properties
        self.min_x, self.max_x = 0.05, 0.85
        self.min_y, self.max_y = 0.2, -0.5
        self.min_goal_radius = 0.05
        self.min_push_dist = 0.05

        self.push_goal_radius = 0.05
        self.valid_pick_yellow_radius = 0.5
        self.approx_valid_push_angle = np.pi/32


    def get_init_state(self):
        pddl_state = copy(self.obj_init_state)
        pddl_state += copy(self.panda_init_state)
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
        self.obj_init_poses = {}

        # tool
        tool_name = 'tool'
        orn = (0,0,0,1)
        #orn_90 = (0,0,1,0)
        #r = np.random.rand()
        #orn = orn if r < .5 else orn_90  # place it randomly at an orientation
        tool, pb_pose = self.place_object(tool_name,
                                            'tamp/urdf_models/%s.urdf' % tool_name,
                                            self.init_objs_pos_xy['tool'],
                                            orn)
        pb_objects[tool_name] = tool
        orig_poses[tool_name] = pb_pose
        self.obj_init_poses[tool_name] = pb_pose
        init_state += [('tool', tool),
                        ('on', tool, self.panda.table),
                        ('clear', tool), \
                        ('atpose', tool, pb_pose),
                        ('pose', tool, pb_pose),
                        ('freeobj', tool)]

        # blue_block (initially constrained by tunnel)

        name = 'blue_block'
        color = (0.0, 0.0, 1.0, 1.0)
        pos_xy = self.init_objs_pos_xy[name]
        urdf_path = 'tamp/urdf_models/%s.urdf' % name
        block_to_urdf(name, urdf_path, color)
        orn = (0,0,0,1)
        block, pb_pose = self.place_object(name, urdf_path, pos_xy, orn)
        pb_objects[name] = block
        orig_poses[name] = pb_pose
        self.obj_init_poses[name] = pb_pose
        init_state += [('block', block),
                        ('on', block, self.panda.table),
                        ('clear', block), \
                        ('atpose', block, pb_pose),
                        ('pose', block, pb_pose),
                        ('freeobj', block)]

        # yellow block (heavy --> must be pushed when outside specified region)
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
        self.obj_init_poses[name] = pb_pose
        init_state += [('block', block),
                        ('on', block, self.panda.table),
                        ('clear', block), \
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
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot,
                                                                    self.fixed,
                                                                    ret_traj=ret_traj)),
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
            'sample-pose-block': from_fn(get_pose_gen_block(self.fixed)),
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


    def generate_goal(self, show_goal=False, goal_xy=None):
        init_state = self.get_init_state()

        # select a random block
        random_object = random.choice(list(self.objects.values()))
        while not ('block', random_object) in init_state:
            random_object = random.choice(list(self.objects.values()))
        init_pose = self.get_obj_pose_from_state(random_object, init_state)
        init_x, init_y = init_pose[0][:2]

        # select random point on table (not near tunnel)
        if not goal_xy:
            goal_xy = np.array([np.random.uniform(self.min_x, self.max_x),
                                np.random.uniform(self.min_y, self.max_y)])

        # add desired pose to state
        goal_pose = ((goal_xy[0], goal_xy[1], init_pose[0][2]), init_pose[1])
        final_pose = pb_robot.vobj.BodyPose(random_object, goal_pose)
        table_pose = pb_robot.vobj.BodyPose(self.panda.table, self.panda.table.get_base_link_pose())
        add_to_state = [('pose', random_object, final_pose),
                            ('supported', random_object, final_pose, self.panda.table, table_pose)]
                            #('atpose', self.panda.table, table_pose),
                            #('clear', self.panda.table)] # TODO: make a place action that
                                                        # doesn't have the effect of making
                                                        # things clear so can place blocks on
                                                        # table. for not hack by saying your
                                                        # can only place something on the table once
                                                        # (then it becomes not clear)

        # visualize goal patch in pyBullet
        # WARNING: SHOWING THE GOAL MESSES UP THE ROBOT INTERACTIONS AND CAUSES COLLISIONS!
        # Do not use if trying to collect accurate data !!
        if show_goal:
            name = 'goal_patch'
            color = (0.0, 1.0, 0.0, 1.0)
            urdf_path = 'tamp/urdf_models/%s.urdf' % name
            goal_to_urdf(name, urdf_path, color, self.push_goal_radius)
            self.panda.execute()
            self.place_object(name, urdf_path, goal_xy, (0,0,0,1))
            self.panda.plan()
            self.place_object(name, urdf_path, goal_xy, (0,0,0,1))

        # return goal
        self.goal = ('atpose', random_object, final_pose)
        return self.goal, add_to_state


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


    def goal_to_vec(self, goal):
        goal_orn = pb_robot.geometry.quat_angle_between(goal[2].pose[1], [0., 0., 0., 1.])
        return np.array([*goal[2].pose[0][:2], goal_orn])


    def action_to_vec(self, pddl_action):
        x = np.zeros(N_MC_IN)
        if pddl_action.name == 'move_contact':
            # calculate the pose of the push goal in the contact frame
            cont = pddl_action.args[5]
            pose1 = pddl_action.args[3]

            # tool pose at contact
            block_world = pb_robot.geometry.tform_from_pose(pose1.pose)
            tool_w_tform = block_world@cont.rel_pose

            # contact frame in the world
            cont_w_tform = np.dot(tool_w_tform, np.linalg.inv(cont.tool_in_cont_tform))
            goal_world_point = (*pddl_action.args[4].pose[0], 1)
            goal_cont = np.dot(np.linalg.inv(cont_w_tform), goal_world_point)
            x[:] = goal_cont[:2]
            return x
        else:
            raise NotImplementedError('No vectorization method for action %s' % pddl_action.name)


    def pred_args_to_vec(self, obj2, pose1, pose2, cont):
        action = Action(name='move_contact', args=(None,
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
            # check that block ended up where it was supposed to (with some tolerance)
            goal_pos2 = pddl_action.args[4].pose[0]
            true_pos2 = pddl_action.args[2].get_base_link_pose()[0]
            dist_to_goal = np.linalg.norm(np.subtract(goal_pos2, true_pos2))
            if dist_to_goal > self.push_goal_radius:
                valid_transition = False
        elif pddl_action.name == 'pick':
            # check that if yellow block, was close to base of robot
            if pddl_action.args[0] == self.objects['yellow_block']:
                init_pos = pddl_action.args[1].pose[0]
                dist_to_base = np.linalg.norm(init_pos)
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
    # selects random optimistic actions
    # pddl_model_type in optimistic or opt_no_traj
    def random_actions(self, state, pddl_model_type):
        ret_traj = True
        if pddl_model_type == 'opt_no_traj':
            ret_traj = False
            pddl_model_type = 'optimistic'
        # get all stream functions
        tool_grasp_fn = get_tool_grasp_gen(self.panda.planning_robot)
        block_grasp_fn = get_block_grasp_gen(self.panda.planning_robot)
        block_place_pose_fn = get_pose_gen_block(self.fixed)
        contacts_fn = get_contact_gen(self.panda.planning_robot)
        pick_fn = get_ik_fn(self,
                            self.panda.planning_robot,
                            self.fixed,
                            approach_frame='gripper',
                            backoff_frame='global')
        place_fn = get_ik_fn(self,
                                self.panda.planning_robot,
                                self.fixed,
                                approach_frame='global',
                                backoff_frame='gripper')
        move_contact_fn = get_contact_motion_gen(self.panda.planning_robot, self.fixed, ret_traj=True)
        move_free_fn = get_free_motion_gen(self.panda.planning_robot, self.fixed, ret_traj=ret_traj)
        move_holding_fn = get_holding_motion_gen(self.panda.planning_robot, self.fixed, ret_traj=ret_traj)

        shuffled_objects = list(self.objects.values())
        random.shuffle(shuffled_objects)
        # generate random actions
        def get_pick_action():
            obj_picked = False
            for obj in shuffled_objects:
                if ('freeobj', obj) in state:
                    for pred in state:
                        if pred[0] == 'on' and pred[1] == obj:
                            bot_obj = pred[2]
                            top_obj = obj
                            obj_picked = True
            if not obj_picked:
                return None, [], False
            for pred in state:
                if pred[0] == 'atpose' and pred[1] == top_obj:
                    pose = pred[2]
            if ('block', top_obj) in state:
                grasps = block_grasp_fn(top_obj)
            elif ('tool', top_obj) in state:
                grasps = tool_grasp_fn(top_obj)
            grasp_i = np.random.randint(len(grasps))
            grasp = grasps[grasp_i][0]
            pick_params = pick_fn(top_obj, pose, grasp)
            if pick_params:
                approach_conf, backoff_conf, traj = pick_params
                # first have to move to initial pick conf
                actions, expanded_states, actions_found = get_move_free_action(final_conf=approach_conf)
                if actions_found:
                    return actions+[Action(name='pick',
                                    args=(top_obj, pose, bot_obj, grasp, *pick_params))], \
                            [('pickkin', top_obj, pose, grasp, *pick_params)]+expanded_states, \
                            True
            else:
                return None, [], False


        def get_place_action():
            obj_held = False
            for pred in state:
                if pred[0] == 'atgrasp':
                    holding_obj = pred[1]
                    holding_grasp = pred[2]
                    obj_held = True
            if not obj_held:
                return None, [], False
            if ('block', holding_obj) not in state: # for now can only place blocks on blocks
                return None, [], False
            bot_obj_set = False
            for obj in shuffled_objects:
                if obj != obj_held and \
                    ('clear', obj) in state and \
                    ('block', obj) in state: # for now can only place blocks on blocks
                    bot_obj = obj
                    for pred in state:
                        if pred[0] == 'atpose' and pred[1] == bot_obj:
                            bot_obj_pose = pred[2]
                    bot_obj_set = True
            if not bot_obj_set:
                return None, [], False
            pose = get_pose_gen_block(holding_obj, bot_obj, bot_obj_pose)
            place_params = place_fn(top_obj, pose, grasp)
            if place_params:
                approach_conf, backoff_conf, traj = place_params
                # first have to move to initial pick conf
                actions, expanded_states, actions_found = get_move_holding_action(final_conf=approach_conf)
                if actions_found:
                    return actions+[Action(name='place',
                                args=(top_obj, pose, bot_obj, bot_pose, \
                                    holding_grasp, *place_params))], \
                            expanded_states+[('placekin', top_obj, pose, holding_grasp, *place_params)], \
                            True
            else:
                return None, [], False


        def get_move_free_action(final_conf=None):
            if ('handempty',) not in state:
                return None, [], False
            for pred in state:
                if pred[0] == 'atconf':
                    init_conf = pred[1]
            init_pose = self.panda.planning_robot.arm.ComputeFK(init_conf.configuration)
            init_orn = init_pose[1]
            if final_conf is None:
                n_attempts = 50
                a = 0
                random_conf = None
                while random_conf is None and a < n_attempts:
                    random_pos = np.array([np.random.uniform(0.05,0.85),
                                        np.random.uniform(0.2,-0.5),
                                        np.random.uniform(0.01, 0.8)])
                    random_conf = self.panda.planning_robot.arm.ComputeIK(pb_robot.geometry.tform_from_pose((random_pos, init_orn)))
                    if random_conf is None:
                        continue
                    if self.panda.planning_robot.arm.IsCollisionFree(random_conf, obstacles=self.fixed):
                        break
                    else:
                        random_conf = None
                if not random_conf:
                    return None, [], False
                final_conf = pb_robot.vobj.BodyConf(self.panda.planning_robot, random_conf)
            traj = move_free_fn(init_conf, final_conf)
            if traj:
                return [Action(name='move_free',
                                args=(init_conf, final_conf, *traj))], \
                                [('freemotion', init_conf, final_conf, *traj)], \
                                True
            else:
                return None, [], False

        def get_move_holding_action(final_conf=None):
            if ('handempty',) in state:
                return None, [], False
            else:
                for pred in state:
                    if pred[0] == 'atgrasp':
                        held_obj = pred[1]
                        grasp = pred[2]
                    if pred[0] == 'atconf':
                        init_conf = pred[1]
            init_pose = self.panda.planning_robot.arm.ComputeFK(init_conf.configuration)
            init_orn = init_pose[1]
            if final_conf is None:
                orig_pose = held_obj.get_base_link_pose()
                self.panda.planning_robot.arm.Grab(held_obj, grasp.grasp_objF)
                n_attempts = 50
                a = 0
                random_conf = None
                while random_conf is None and a < n_attempts:
                    random_pos = np.array([np.random.uniform(0.05,0.85),
                                        np.random.uniform(0.2,-0.5),
                                        np.random.uniform(0.01, 0.8)])
                    random_conf = self.panda.planning_robot.arm.ComputeIK(pb_robot.geometry.tform_from_pose((random_pos, init_orn)))
                    if random_conf is None:
                        continue
                    if self.panda.planning_robot.arm.IsCollisionFree(random_conf, obstacles=self.fixed):
                        break
                    else:
                        random_conf = None
                self.panda.planning_robot.arm.Release(held_obj)
                held_obj.set_base_link_pose(orig_pose)
                if not random_conf:
                    return None, [], False
                final_conf = pb_robot.vobj.BodyConf(self.panda.planning_robot, random_conf)
            traj = move_holding_fn(held_obj, grasp, init_conf, final_conf)
            if traj:
                return [Action(name='move_holding', args=(held_obj, grasp, init_conf, final_conf, *traj))], \
                        [('holdingmotion', held_obj, grasp, init_conf, final_conf, *traj)], \
                        True
            else:
                return None, [], False


        def get_move_contact_action():
            held_obj = None
            for pred in state:
                if pred[0] == 'held':
                    held_obj = pred[1]
                if pred[0] == 'atgrasp':
                    grasp = pred[2]
            if held_obj is None:
                return None, [], False
            pushed_obj = None
            for obj in shuffled_objects:
                if ('freeobj', obj) in state and \
                    ('block', obj) in state and \
                    ('tool', held_obj) in state: # can only push blocks with tool right now
                    pushed_obj = obj
            if pushed_obj is None:
                return None, [], False
            contacts = contacts_fn(held_obj, pushed_obj)
            contact_i = np.random.randint(len(contacts))
            cont = contacts[contact_i][0]
            for pred in state:
                if pred[0] == 'atpose' and pred[1] == pushed_obj:
                    pushed_obj_pose = pred[2]
            final_pos_xy = np.array([np.random.uniform(0.05,0.85),
                                np.random.uniform(0.2,-0.5)])
            final_pose = pb_robot.vobj.BodyPose(pushed_obj,
                                ((*final_pos_xy, pushed_obj_pose.pose[0][2]),
                                pushed_obj_pose.pose[1]))
            move_params = move_contact_fn(held_obj, grasp, pushed_obj, pushed_obj_pose, \
                                            final_pose, cont)
            if move_params:
                init_conf, pose1_conf, final_conf, traj = move_params
                # first have to move to initial pick conf
                actions, expanded_states, actions_found = get_move_holding_action(final_conf=init_conf)
                if actions_found:
                    return actions+[Action(name='move_contact',
                                args=(held_obj, grasp, pushed_obj, pushed_obj_pose, final_pose, \
                                    cont, *move_params))], \
                        expanded_states+[('contactmotion', held_obj, grasp, pushed_obj, pushed_obj_pose, final_pose, \
                            cont, *move_params)], \
                        True
            return None, [], False

        action_fns = [get_pick_action, get_place_action, get_move_free_action, \
                        get_move_contact_action, get_move_holding_action]
        precondition_met = False
        attempts = 50
        attempt = 0
        while not precondition_met and attempt < attempts:
            random_action_fn = np.random.choice(action_fns)
            actions, expanded_states, precondition_met = random_action_fn()
            if precondition_met:
                return actions, expanded_states, precondition_met
            attempt += 1
        return None, None, False


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


    # can visualize tool in world or contact frame
    def vis_tool_ax(self, cont, ax, frame='world'):
        if frame == 'world':
            init_block_pos = self.init_objs_pos_xy['yellow_block']
            # TODO: this assumes that the block is always aligned with the world frame
            tool_tform = cont.rel_pose
        elif frame == 'cont':
            init_block_pos = (0., 0.)
            tool_tform = cont.tool_in_cont_tform

        self.plot_block(ax, init_block_pos, color='m')
        self.plot_tool(ax, tool_tform, 'k')
        ax.set_aspect('equal')
        if frame == 'world':
            ax.set_xlim([self.min_x, self.max_x])
            ax.set_ylim([self.min_y, self.max_y])
        elif frame == 'cont':
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])


    def vis_dense_plot(self, type, ax, x_range, y_range, vmin, vmax, value_fn=None, cell_width=0.05):
        # make 2d arrays of mean and std ensemble predictions
        xs, x_extent = self.make_array(*x_range, cell_width)
        ys, y_extent = self.make_array(*y_range, cell_width)
        values = np.zeros((len(ys), len(xs)))

        for xi, xv in enumerate(xs):
            for yi, yv in enumerate(ys):
                values[yi][xi] = value_fn(self, type, xv, yv)

        # plot predictions w/ colorbars
        extent = (*x_extent, *y_extent)

        im0 = ax.imshow(values, origin='lower', cmap='binary', extent=extent, vmin=vmin, vmax=vmax, aspect='equal')
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


    # for now can only run this after vis_model_accuracy since it sets up the axes
    def vis_bald(self, bald_scores, states, best_i, axes=None):
        contacts_fn = get_contact_gen(self.panda.planning_robot)
        contacts = contacts_fn(self.objects['tool'], self.objects['yellow_block'], shuffle=False)
        #cont = contacts[0][0]   # NOTE: this was when I was debugging and there was only 1 possible contact

        if not axes:
            axes = {}

        for ci, contact in enumerate(contacts):
            if ci in axes:
                ax = axes[ci]
            else:
                fig, ax = plt.subplots(3, figsize=(8,15))
                axes[ci] = ax

            # plot initial and goal (from BALD) poses as well as BALD's sampled poses
            best_state = states[best_i]
            vof, vef, va = best_state

            max_score = max(bald_scores)
            normalized_scores = [score/max_score for score in bald_scores]
            for ax_k in ax[:2]:# show a block at initial pos
                # visualize goal that was selected by BALD
                self.plot_block(ax_k, va[:2], color='m', linestyle='--')

                # visualize all BALD scores
                for n_score, state in zip(normalized_scores, states):
                    vof, vef, va = state
                    plot_vec = va[:2]

                    ax_k.plot(*plot_vec[:2], 'cx')#, color=str(n_score))
                    print(plot_vec[:2], n_score)

        return axes


    # for now can only run this after vis_model_accuracy since it sets up the axes
    # each axis in axes is a 3 part subplot for a single contact
    def vis_dataset(self, ax, dataset, linestyle='-'):
        # plot initial position
        self.plot_block(ax, (0,0), 'm')
        for x, y in dataset:
            color = 'r' if y == 0 else 'g'
            self.plot_block(ax, x, color, linestyle=linestyle)


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
