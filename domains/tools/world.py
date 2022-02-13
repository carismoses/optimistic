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

from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.language.generator import from_list_fn, from_fn, from_test

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import get_simple_state, get_learned_pddl, block_to_urdf, goal_to_urdf
from domains.tools.primitives import get_free_motion_gen, \
    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_tool_grasp_gen, \
    get_block_grasp_gen, get_contact_motion_gen, get_contact_gen, contact_approach_fn, ee_ik
from domains.tools.add_to_primitives import get_trust_model
from learning.utils import model_forward

N_OF_IN = 1
N_EF_IN = 3
N_AF_IN = 7

# TODO: make parent world template class
class ToolsWorld:
    @staticmethod
    def init(domain_args, pddl_model_type, vis, logger=None, planning_model_i=None):
        world = ToolsWorld(vis, pddl_model_type, logger, planning_model_i)
        return world

    @staticmethod
    def get_model_params():
        return N_OF_IN, N_EF_IN, N_AF_IN

    def __init__(self, vis, pddl_model_type, logger, planning_model_i):
        # initial block poses
        self.init_pos_yellow = (0.4, -0.3)

        self.use_panda = True
        self.panda = PandaAgent(vis)
        self.panda.plan()
        self.objects, self.orig_poses, self.obj_init_state = self.place_objects()
        self.panda_init_state = self.panda.get_init_state()
        self.panda.execute()
        self.place_objects()
        self.panda.plan()
        self.fixed = [self.panda.table, self.tunnel]

        # TODO: test without gravity?? maybe will stop robot from jumping around so much
        p.setGravity(0, 0, -9.81, physicsClientId=self.panda._execution_client_id)

        # GNN model params
        self.n_of_in = N_OF_IN
        self.n_ef_in = N_EF_IN
        self.n_af_in = N_AF_IN

        # get pddl domain description
        self.logger = logger
        self.planning_model_i = planning_model_i

        # goal sampling properties
        self.min_x, self.max_x = 0.05, 0.85
        self.min_y, self.max_y = 0.2, -0.5
        self.min_goal_radius = 0.05
        self.min_push_dist = 0.05

        init_x, init_y = self.init_pos_yellow
        self.max_dist = max([abs(init_x-self.min_x),
                        abs(init_x-self.max_x),
                        abs(init_y-self.min_y),
                        abs(init_y-self.max_y)])
        self.goal_radius = 0.05

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
        '''
        # blue_block (under tunnel)
        name = 'blue_block'
        color = (0.0, 0.0, 1.0, 1.0)
        pos_xy = (0.3, 0.4)
        urdf_path = 'tamp/urdf_models/%s.urdf' % name
        block_to_urdf(name, urdf_path, color)
        block, pose = self.place_object(name, urdf_path, pos_xy)
        pb_objects[name] = block
        orig_poses[name] = pose
        self.obj_init_poses[name] = pose
        init_state += [('block', block), ('on', block, self.panda.table), ('clear', block), \
                        ('atpose', block, pose), ('pose', block, pose), ('freeobj', block)]#, \
                        #('notheavy', block)]
        '''
        # yellow block (heavy --> must be pushed)
        name = 'yellow_block'
        color = (1.0, 1.0, 0.0, 1.0)
        pos_xy = self.init_pos_yellow
        urdf_path = 'tamp/urdf_models/%s.urdf' % name
        block_to_urdf(name, urdf_path, color)
        block, pose = self.place_object(name, urdf_path, pos_xy)
        pb_objects[name] = block
        orig_poses[name] = pose
        self.obj_init_poses[name] = pose
        init_state += [('block', block), ('on', block, self.panda.table), ('clear', block), \
                        ('atpose', block, pose), ('pose', block, pose), ('freeobj', block)]
        '''
        # red block (notheavy --> can be picked)
        name = 'red_block'
        color = (1.0, 0.0, 0.0, 1.0)
        pos_xy = (0.6, 0.0)
        urdf_path = 'tamp/urdf_models/%s.urdf' % name
        block_to_urdf(name, urdf_path, color)
        block, pose = self.place_object(name, urdf_path, pos_xy)
        pb_objects[name] = block
        orig_poses[name] = pose
        self.obj_init_poses[name] = pose
        init_state += [('block', block), ('on', block, self.panda.table), ('clear', block), \
                        ('atpose', block, pose), ('pose', block, pose), ('freeobj', block), \
                        ('notheavy', block)]
        '''
        # tunnel
        tunnel_name = 'tunnel'
        tunnel, pose = self.place_object(tunnel_name, 'tamp/urdf_models/%s.urdf' % tunnel_name, (0.3, 0.4))
        self.tunnel = tunnel

        return pb_objects, orig_poses, init_state


    # pddl_model_type in {optimistic, learned, opt_no_traj}
    def get_pddl_info(self, pddl_model_type):
        ret_traj = True
        if pddl_model_type == 'opt_no_traj':
            ret_traj = False
            pddl_model_type = 'optimistic'
        robot = self.panda.planning_robot
        opt_domain_pddl_path = 'domains/tools/domain.pddl'
        opt_streams_pddl_path = 'domains/tools/streams.pddl'
        opt_streams_map = {
            'plan-free-motion': from_fn(get_free_motion_gen(robot,
                                                            self.fixed,
                                                            ret_traj=ret_traj)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot,
                                                                    self.fixed,
                                                                    ret_traj=ret_traj)),
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
                                                                    self.fixed,
                                                                    ret_traj=ret_traj)),
            'sample-contact': from_list_fn(get_contact_gen(robot))
            }

        opt_streams_pddl = read(opt_streams_pddl_path) if opt_streams_pddl_path else None
        if pddl_model_type == 'optimistic':
            domain_pddl = read(opt_domain_pddl_path)
            streams_pddl = opt_streams_pddl
            streams_map = opt_streams_map
        elif pddl_model_type == 'learned':
            add_to_domain_path = 'domains/tools/add_to_domain.pddl'
            add_to_streams_path = 'domains/tools/add_to_streams.pddl'
            domain_pddl, streams_pddl = get_learned_pddl(opt_domain_pddl_path,
                                                        opt_streams_pddl_path,
                                                        add_to_domain_path,
                                                        add_to_streams_path)
            streams_map = copy(opt_streams_map)
            streams_map['trust-model'] = from_test(get_trust_model(self, self.logger, planning_model_i=self.planning_model_i))
        constant_map = {}
        pddl_info = [domain_pddl, constant_map, streams_pddl, streams_map]
        return pddl_info


    def change_goal_space(self, progress):
        if progress is not None:
            new_goal_radius = self.max_dist*(1-progress)
            if new_goal_radius > self.min_goal_radius:
                self.goal_radius = new_goal_radius


    def generate_goal(self, feasible=False, ret_goal_feas=False, show_goal=False):
        init_state = self.get_init_state()

        # select a random block
        random_object = random.choice(list(self.objects.values()))
        while not ('block', random_object) in init_state:
            random_object = random.choice(list(self.objects.values()))
        init_pose = self.get_obj_pose_from_state(random_object, init_state)
        init_x, init_y = init_pose[0][:2]

        # select random point on table (not near tunnel)
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
            goal_to_urdf(name, urdf_path, color, self.goal_radius)
            self.panda.execute()
            self.place_object(name, urdf_path, goal_xy)
            self.panda.plan()
            self.place_object(name, urdf_path, goal_xy)

        # return goal
        self.goal = ('atpose', random_object, final_pose)
        return self.goal, add_to_state


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
        pick_fn = get_ik_fn(self.panda.planning_robot,
                            self.fixed,
                            approach_frame='gripper',
                            backoff_frame='global')
        place_fn = get_ik_fn(self.panda.planning_robot,
                                self.fixed,
                                approach_frame='global',
                                backoff_frame='gripper')
        move_contact_fn = get_contact_motion_gen(self.panda.planning_robot, self.fixed, ret_traj=ret_traj)
        move_free_fn = get_free_motion_gen(self.panda.planning_robot, self.fixed, ret_traj=ret_traj)
        move_holding_fn = get_holding_motion_gen(self.panda.planning_robot, self.fixed, ret_traj=ret_traj)

        shuffled_objects = list(self.objects.values())
        random.shuffle(shuffled_objects)
        # generate random actions
        def get_pick_action():
            obj_picked = False
            for obj in shuffled_objects:
                if ('notheavy', obj) in state and ('freeobj', obj) in state:
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
            # check that block ended up where it was supposed to (with some tolerance)
            goal_pos2 = pddl_action.args[4].pose[0]
            true_pos2 = pddl_action.args[2].get_base_link_pose()[0]
            dist_to_goal = np.linalg.norm(np.subtract(goal_pos2, true_pos2))
            if dist_to_goal > self.goal_radius:
                valid_transition = False
        self.panda.plan()
        return valid_transition


    def add_goal_text(self, goal):
        self.panda.add_text('Planning for Goal: (%s, %s, (%f, %f, %f))' % \
                                            (goal[0], goal[1], *goal[2].pose[0]),
                        position=(0, -1, 1),
                        size=1.5)


    '''
    PLOTTING FUNCTIONS
    '''
    def vis_model_accuracy(self, model, goal_from_state=False, axes=None):
        def make_array(minv, maxv, step):
            if minv > maxv:
                ar = np.flip(np.arange(maxv, minv+step, step))
                extent = minv+step/2, maxv-step/2
                return ar, extent
            else:
                ar = np.arange(minv, maxv+step, step)
                extent = minv-step/2, maxv+step/2
                return ar, extent

        # make 2d arrays of mean and std ensemble predictions
        cell_width = 0.05
        xs, x_extent = make_array(self.min_x, self.max_x, cell_width)
        ys, y_extent = make_array(self.min_y, self.max_y, cell_width)
        mean_preds = np.zeros((len(ys), len(xs)))
        std_preds = np.zeros((len(ys), len(xs)))

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

            cont = contact[0]
            if goal_from_state:
                init_state = self.get_init_state()
                init_pose = self.get_obj_pose_from_state(self.objects['yellow_block'], init_state)

                for xi, xv in enumerate(xs):
                    for yi, yv in enumerate(ys):
                        pose = ((xv, yv, init_pose[0][2]), init_pose[1])
                        goal_pose = pb_robot.vobj.BodyPose(self.objects['yellow_block'], pose)

                        # NOTE this will generate approach configurations that might
                        # not actually be able to follow a push path (due to kinematic constraints)
                        tool_approach = contact_approach_fn(self.objects['tool'],
                                                                self.objects['yellow_block'],
                                                                self.obj_init_poses['yellow_block'],
                                                                goal_pose,
                                                                cont)
                        vof, vef, va = self.get_model_inputs(tool_approach, goal_pose)

                        # calc mean pred
                        mean_pred = model_forward(model, [vof, vef, va], single_batch=True).mean().squeeze()

                        # calc std pred
                        std_pred =  model_forward(model, [vof, vef, va], single_batch=True).std().squeeze()

                        mean_preds[yi][xi] = mean_pred
                        std_preds[yi][xi] = std_pred
            else:
                for xi, xv in enumerate(xs):
                    for yi, yv in enumerate(ys):
                        predictions = model_forward(model, [np.array([xv, yv, 0.0])], single_batch=True)
                        mean_preds[yi][xi] = predictions.mean().squeeze()
                        std_preds[yi][xi] = predictions.std().squeeze()

            # plot predictions w/ colorbars
            extent = (*x_extent, *y_extent)

            im0 = ax[0].imshow(mean_preds, origin='lower', cmap='binary', extent=extent, vmin=0, vmax=1, aspect='equal')
            divider0 = make_axes_locatable(ax[0])
            cax0 = divider0.append_axes("right", size="10%", pad=0.5)
            cbar0 = plt.colorbar(im0, cax=cax0, format="%.2f")
            #print(std_preds)
            im1 = ax[1].imshow(std_preds, origin='lower', cmap='binary', extent=extent, aspect='equal')#), vmin=0, vmax=1)
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes("right", size="10%", pad=0.5)
            cbar1 = plt.colorbar(im1, cax=cax1, format="%.4f")

            ax[0].set_title('Mean Ensemble Predictions')
            ax[1].set_title('Std Ensemble Predictions')

            # show a block at initial pos
            for ax_k in ax[:2]:
                self.plot_block(ax_k, self.init_pos_yellow, color='m', linestyle='-')

            # show tool pose relative to block in final axis
            tool_base_pos, tool_base_orn = cont.rel_pose
            angle = pb_robot.geometry.quat_angle_between(tool_base_orn, [0., 0., 0., 1.])
            tool_2d_pose = (*np.add(self.init_pos_yellow, tool_base_pos[:2]), angle)
            self.plot_block(ax[2], self.init_pos_yellow, color='k')
            self.plot_tool(ax[2], tool_2d_pose, 'k')
            ax[2].set_aspect('equal')
            ax[2].set_xlim([self.min_x, self.max_x])
            ax[2].set_ylim([self.min_y, self.max_y])

        return axes


    # for now can only run this after vis_model_accuracy since it sets up the axes
    def vis_bald(self, bald_scores, states, best_i, axes=None, goal_from_state=True):
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
            if goal_from_state:
                vof, vef, va = best_state
            else:
                va = best_state

            max_score = max(bald_scores)
            normalized_scores = [score/max_score for score in bald_scores]
            for ax_k in ax[:2]:# show a block at initial pos
                # visualize goal that was selected by BALD
                self.plot_block(ax_k, va[:2], color='m', linestyle='--')

                # visualize all BALD scores
                for n_score, state in zip(normalized_scores, states):
                    if goal_from_state:
                        vof, vef, va = state
                        plot_vec = va[:2]
                    else:
                        plot_vec = state[:2]
                    ax_k.plot(*plot_vec[:2], 'cx')#, color=str(n_score))
                    print(plot_vec[:2], n_score)

        return axes


    # for now can only run this after vis_model_accuracy since it sets up the axes
    def vis_dataset(self, logger, axes=None, goal_from_state=False, dataset_i=None):
        contacts_fn = get_contact_gen(self.panda.planning_robot)
        contacts = contacts_fn(self.objects['tool'], self.objects['yellow_block'], shuffle=False)
        #cont = contacts[0][0]   # NOTE: this was when I was debugging and there was only 1 possible contact

        init_state = self.get_init_state()
        init_pose = self.get_obj_pose_from_state(self.objects['yellow_block'], init_state)

        if not axes:
            axes = {}

        for ci, contact in enumerate(contacts):
            if ci in axes:
                ax = axes[ci]
            else:
                fig, ax = plt.subplots(3, figsize=(8,15))
                axes[ci] = ax
            cont = contact[0]

            # plot all previously executed goal poses colored by action success
            if goal_from_state:
                dataset = logger.load_trans_dataset(i=dataset_i)
            else:
                dataset = logger.load_dataset('goal')
            for x, y in dataset:
                if goal_from_state:
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
                    dist = np.linalg.norm(np.subtract(vef_j,ef))
                    if dist < 0.01:
                        #print(dist)
                        color = 'r' if y == 0 else 'g'
                        self.plot_block(ax[0], goal_pos_xy, color)
                        self.plot_block(ax[1], goal_pos_xy, color)
                else:
                    goal_pos_xy = x[:2]
                    color = 'r' if y == 0 else 'g'
                    self.plot_block(ax[0], goal_pos_xy, color)
                    self.plot_block(ax[1], goal_pos_xy, color)

        return axes


    def plot_block(self, ax, pos, color, linestyle='-'):
        block_dims = np.array(self.objects['yellow_block'].get_dimensions()[:2])
        ax.add_patch(Rectangle(pos-block_dims/2,
                                *block_dims,
                                color=color,
                                fill = False,
                                linestyle=linestyle))


    def plot_tool(self, ax, rel_2d_pose, color):
        tool_data = p.getVisualShapeData(self.objects['tool'].id, -1)
        tool_base_dims = np.array(tool_data[0][3][:2])
        tool_link0_dims = np.array(tool_data[1][3][:2])
        link0_relpos = np.array([-0.185, 0.115]) # from URDF

        ax.add_patch(Rectangle(rel_2d_pose[:2] - tool_base_dims/2,
                                *tool_base_dims,
                                angle = rel_2d_pose[2],
                                color = color,
                                fill = False))
        ax.add_patch(Rectangle(rel_2d_pose[:2] + link0_relpos - tool_link0_dims/2,
                                *tool_link0_dims,
                                angle = rel_2d_pose[2],
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


    def plot_datapoint(self, i, logger, color=None, dir='goals', show=False):
        '''
        i: index into the dataset to plot
        color: a color string for matplotlib to use. if None then color is red
                for 0 (infeasible) label and green for feasible label
        dir: which subfolder within exp_path/figures/ to save the plot
        show: boolean indicating if want to show the resulting image (blocking)
        '''
        fig, ax = plt.subplots()

        # plot block initial pose
        block_pos_xy = np.array(self.obj_init_poses['yellow_block'].pose[0][:2])
        self.plot_block(ax, block_pos_xy, 'k')

        # plot all goal poses colored by success and show relative contact
        dataset = logger.load_trans_dataset()
        x, y = dataset[i]
        of, ef, af = x
        goal_pos_xy = af[:2]
        if color is None:
            color = 'r' if y == 0 else 'g'
        self.plot_block(ax, goal_pos_xy, color)

        # rel_pose from block to tool
        tool_feature_index = np.where(of == self.objects['tool'].id)[0]
        block_feature_index = np.where(of == self.objects['yellow_block'].id)[0]

        pose_2d = np.array([*block_pos_xy, 0.0]) + ef[tool_feature_index, block_feature_index][0].numpy()
        self.plot_tool(ax, pose_2d, color)

        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(0.3, -0.6)
        ax.set_aspect('equal')
        fig.suptitle('Iteration %i' % i)
        logger.save_figure('successes_%i.png'%i, dir=dir)

        if show:
            plt.show()
        else:
            plt.close()


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
