import os
import numpy as np
from copy import copy
import random
from shutil import copyfile

import pb_robot

from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.algorithms.downward import fact_from_fd, apply_action
from pddlstream.language.generator import from_list_fn, from_fn

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import get_simple_state, get_learned_pddl, block_to_urdf
#from domains.tools.primitives import get_free_motion_gen, \
#    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_grasp_gen
#from domains.tools.add_to_primitives import get_trust_model

class ToolsWorld:
    @staticmethod
    def init(domain_args, pddl_model_type, vis, logger=None):
        world = ToolsWorld(vis)
        opt_pddl_info, pddl_info = None, None#world.get_pddl_info(pddl_model_type, logger)
        return world, opt_pddl_info, pddl_info


    def __init__(self, vis):
        self.use_panda = True
        self.panda = PandaAgent(vis)
        self.panda.plan()
        self.objects, self.orig_poses = self.place_objects()
        self.panda.execute()
        self.place_objects()
        self.panda.plan()
        self.fixed = [self.panda.table]#, self.tunnel]
        self.init_state = self.get_init_state()


    def get_init_state(self):
        pddl_state = []
        shuffled_objects = list(self.objects.values())
        random.shuffle(shuffled_objects)
        for o in shuffled_objects:
            pose = pb_robot.vobj.BodyPose(o, o.get_base_link_pose())
            pddl_state += [#o.get_type_fluent(),
                            ('clear', o),
                            ('on', o, self.panda.table),
                            ('pose', o, pose),
                            ('atpose', o, pose)]
        pddl_state += [('table', self.panda.table)]
        pddl_state += self.panda.get_init_state()
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


    # world frame aligns with the robot base
    def place_objects(self):
        # tool
        tool_name = 'tool'
        tool_info = (tool_name, 'tamp/urdf_models/%s.urdf' % tool_name, (0.3, -0.4))

        # blocks
        blocks = [('red_block', (1.0, 0.0, 0.0, 1.0), (0.9, 0.0)),
                    ('blue_block', (0.0, 0.0, 1.0, 1.0), (0.3, 0.4)),
                    ('yellow_block', (1.0, 1.0, 0.0, 1.0), (0.7, -0.3))]
        blocks_info = []
        for name, color, pos_xy in blocks:
            urdf_path = 'tamp/urdf_models/%s.urdf' % name
            block_to_urdf(name, urdf_path, color)
            blocks_info.append((name, urdf_path, pos_xy))

        # tunnel
        tunnel_name = 'tunnel'
        tunnel_info = (tunnel_name, 'tamp/urdf_models/%s.urdf' % tunnel_name, (0.3, 0.4))

        # patches
        patches_info = []
        for name, pos_xy in [('green_patch', (0.4, -0.3)), ('violet_patch', (0.7, 0.4))]:
            patches_info.append((name,
                            'tamp/urdf_models/%s.urdf' % name,
                            pos_xy))

        pb_objects = {}
        orig_poses = {}
        object_info = [tool_info]+blocks_info+[tunnel_info]+patches_info
        for obj_name, path, pos_xy in object_info:
            fname = os.path.basename(path)
            copyfile(path, os.path.join('pb_robot/models', fname))
            obj = pb_robot.body.createBody(os.path.join('models', fname))
            z = pb_robot.placements.stable_z(obj, self.panda.table)
            obj_pose = ((*pos_xy, z), (0., 0., 0., 1.))
            obj.set_base_link_pose(obj_pose)
            pb_objects[obj_name] = obj
            orig_poses[obj_name] = obj_pose
        return pb_objects, orig_poses


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
            'sample-grasp': from_list_fn(get_grasp_gen(robot)),
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
            streams_map = opt_streams_map
            streams_map['TrustModel'] = get_trust_model(self, logger)

        constant_map = {}
        opt_pddl_info = [opt_domain_pddl, constant_map, opt_streams_pddl, opt_streams_map]
        pddl_info = [domain_pddl, constant_map, streams_pddl, streams_map]
        return opt_pddl_info, pddl_info


    def generate_random_goal(self, feasible=False, ret_goal_feas=False):
        random_block = random.choice(list(self.pb_blocks))
        random_patch = random.choice(list(self.patches))
        goal = ('on', random_block, random_patch)
        if ret_goal_feas:
            return goal, True # all goals in this domain are feasible
        return goal


    # TODO: select a random discrete action then ground it
    def random_actions(self, state):
        # TODO
        return [], []


    def state_to_vec(self, state):
        n_state_features = 6    # obj_id, obj_type, clear, free_obj, grasp, pose
        n_global_features = 2   # hand_empty, at_conf
        n_edge_features = 1     # on
        state = get_simple_state(state)
        object_features = np.zeros((len(self.objects), n_state_features))
        global_features = np.zeros(n_global_features)
        edge_features = np.zeros((n_edge_features, n_edge_features, 1))
        for oi, object in enumerate(self.objects):
            object_features[oi] = self.object_to_vec(object)

        for ai, object_a in self.objects:
            for bi, object_b in self.objects:
                if ('on', object_a, object_b) in state:
                    edge_features[ai, bi, 0] = 1

        global_features = None# TODO
        return object_features, edge_features, global_features


    def action_to_vec(self, action):
        # TODO
        pass


    def action_args_to_action(self, top_block, bottom_block):
        #return Action(name='pickplace', args=(top_block, self.table, bottom_block))
        #TODO
        pass


    # NOTE: in physical domains, to evaluate if a transition matched the optimistic model,
    # we will have to see if the resulting physical state matches the resulting PDDL
    # state. This will require a method of going from the physical world to a PDDL
    # respresentation of the state.
    # NOTE this is a bit hacky. should get indices from param names ?bt and ?bb
    def valid_transition(self, pddl_action):
        # TODO
        return None

# just for testing
if __name__ == '__main__':
    import pybullet as p
    import time

    world, _, _ = ToolsWorld.init(None, 'optimistic', True, logger=None)
    while True:
        p.stepSimulation(physicsClientId=world.panda._execution_client_id)
        time.sleep(.1)
