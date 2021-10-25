import os
import odio_urdf
import numpy as np
from copy import copy
import random

import pb_robot

from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.algorithms.downward import fact_from_fd, apply_action
from pddlstream.language.generator import from_list_fn, from_fn

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import get_simple_state
from domains.ordered_blocks.discrete_domain.learned.primitives import get_trust_model
from domains.ordered_blocks.panda_domain.primitives import get_free_motion_gen, \
    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_grasp_gen

class OrderedBlocksWorld:
    @staticmethod
    def init(domain_args, pddl_model_type, vis, logger=None):
        num_blocks = int(domain_args[0])
        use_panda = domain_args[1] == 'True'
        world = OrderedBlocksWorld(num_blocks, use_panda, vis)
        opt_pddl_info, pddl_info = world.get_pddl_info(pddl_model_type, logger)
        return world, opt_pddl_info, pddl_info


    def __init__(self, num_blocks, use_panda, vis):
        self.num_blocks = num_blocks  # NOTE: 1 < num_blocks <= 8
        self.use_panda = use_panda

        if self.use_panda:
            self.panda = PandaAgent(vis)
            self.panda.plan()
            self.pb_blocks, self.orig_poses = self.place_blocks()
            self.panda.execute()
            self.place_blocks()
            self.panda.plan()
            self.fixed = [self.panda.table]
        self.init_state = self.get_init_state()


    def get_init_state(self):
        pddl_state = []
        for pb in self.pb_blocks:
            pose = pb_robot.vobj.BodyPose(pb, pb.get_base_link_pose())
            pddl_state += [('pose', pb, pose),
                            ('atpose', pb, pose),
                            ('clear', pb),
                            ('on', pb, self.panda.table),
                            ('block', pb)]
        if self.use_panda:
            pddl_state += self.panda.get_init_state()
        return pddl_state


    # NOTE: this reset looks like it works but then planning fails
    def reset(self):
        def reset_blocks():
            for pb_block, block_pose in self.orig_poses.items():
                pb_block.set_base_link_pose(block_pose)
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
    def place_blocks(self):
        ys = np.linspace(-.4, .4, self.num_blocks)
        x = 0.3
        xy_points = [(x, y) for y in ys]
        pb_blocks = {}
        orig_poses = {}
        for block_num, xy_point in zip(range(1, self.num_blocks+1), xy_points):
            file_name = block_to_urdf(block_num)
            pb_block = pb_robot.body.createBody(os.path.join('models', file_name))
            # NOTE: for now assumes no relative rotation between robot base/world frame and object
            z = pb_robot.placements.stable_z(pb_block, self.panda.table)
            block_pose = ((*xy_point, z), (0., 0., 0., 1.))
            pb_block.set_base_link_pose(block_pose)
            pb_blocks[pb_block] = block_num
            orig_poses[pb_block] = block_pose
        return pb_blocks, orig_poses


    def get_pddl_info(self, pddl_model_type, logger):
        if self.use_panda:
            robot = self.panda.planning_robot
            optimistic_domain_pddl = read('domains/ordered_blocks/panda_domain/optimistic/domain.pddl')
            optimistic_stream_pddl = read('domains/ordered_blocks/panda_domain/optimistic/streams.pddl')
            optimistic_stream_map = {
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
            if pddl_model_type == 'optimistic':
                domain_pddl = optimistic_domain_pddl
                stream_pddl = optimistic_stream_pddl
                stream_map = optimistic_stream_map
            elif pddl_model_type == 'learned':
                pass # TODO
        else:
            optimistic_domain_pddl = read('domains/ordered_blocks/discrete_domain/optimistic/domain.pddl')
            optimistic_stream_pddl = None
            optimistic_stream_map = {}
            if pddl_model_type == 'optimistic':
                domain_pddl = optimistic_domain_pddl
                stream_pddl = optimistic_stream_pddl
                stream_map = optimistic_stream_map
            elif pddl_model_type == 'learned':
                domain_pddl = read('domains/ordered_blocks/discrete_domain/learned/domain.pddl')
                stream_pddl = read('domains/ordered_blocks/discrete_domain/learned/streams.pddl')
                stream_map = {'TrustModel': get_trust_model(self, logger)}
        constant_map = {}
        optimistic_pddl_info = [optimistic_domain_pddl,
                                constant_map,
                                optimistic_stream_pddl,
                                optimistic_stream_map]
        pddl_info = [domain_pddl, constant_map, stream_pddl, stream_map]
        return optimistic_pddl_info, pddl_info


    def generate_random_goal(self, feasible=False):
        random_top_block = random.choice(list(self.pb_blocks))
        if feasible:
            top_block_num = self.pb_blocks[random_top_block]
            random_height = np.random.randint(2, top_block_num+1)
        else:
            random_height = np.random.randint(2, self.num_blocks+1)
        random_height = 2
        return ('height%s' % int_to_str(random_height), random_top_block)


    # TODO: is there a way to sample random actions using PDDL code?
    def random_actions(self):
        state = self.init_state
        # TODO: this doesn't work
        if self.use_panda:
            robot = self.panda.planning_robot

            # pick action
            grasp_fn = get_grasp_gen(robot)
            pick_fn = get_ik_fn(robot,
                                self.fixed,
                                approach_frame='gripper',
                                backoff_frame='global')
            block = list(self.pb_blocks.values())[0]
            pose = self.init_state[0][2]
            grasps = grasp_fn(block)
            i = 0
            while True:
                try:
                    print('attempt %i' % i)
                    grasp_i = np.random.randint(len(grasps))
                    grasp = grasps[grasp_i][0]
                    pinit_conf, final_conf, traj = pick_fn(block, pose, grasp)
                    break
                except:
                    i += 1
                    pass
            pick_pre = ('pickkin', block, pose, grasp, pinit_conf, final_conf, traj)
            pick_action = Action(name='pick', args=(block, pose, grasp, pinit_conf, final_conf, traj))

            # move free action
            init_conf = self.init_state[-2][1]
            free_fn = get_free_motion_gen(robot, self.fixed)
            free_traj = free_fn(init_conf, pinit_conf)
            free_pre = ('freemotion', init_conf, free_traj, pinit_conf)
            free_action = Action(name='move_free', args=(init_conf, pinit_conf, free_traj))

            return [free_action, pick_action], [free_pre, pick_pre]
        else:
            action = None
            simple_state = get_simple_state(state)
            table_blocks = [bn for bn in range(1, self.num_blocks+1)
                    if ('ontable', bn) in simple_state and ('clear', bn) in simple_state]
            if len(table_blocks) > 0:
                top_block_idx = np.random.choice(len(table_blocks))
                top_block_num = table_blocks[top_block_idx]
                possible_bottom_blocks = []
                for bn in range(1, self.num_blocks+1):
                    if ('clear', bn) in simple_state and bn != top_block_num:
                        possible_bottom_blocks.append(bn)
                bottom_block_idx = np.random.choice(len(possible_bottom_blocks))
                bottom_block_num = possible_bottom_blocks[bottom_block_idx]
                action = Action(name='stack', args=(top_block_num, bottom_block_num))
            return [action], []


    def state_to_vec(self, state):
        simple_state = get_simple_state(state)
        def block_on_top(bottom_block, state):
            for top_block in self.pb_blocks:
                if ('on', top_block, bottom_block) in simple_state:
                    return True, top_block
            return False, None

        object_features = np.expand_dims(np.arange(self.num_blocks+1), 1)
        edge_features = np.zeros((self.num_blocks+1, self.num_blocks+1, 1))
        # for each block on the table, recursively check which blocks are on top of it
        for block in self.pb_blocks:
            if ('on', block, self.panda.table) in simple_state:
                edge_features[0, self.pb_blocks[block], 0] = 1.
                is_block_on_top, top_block = block_on_top(block, state)
                bottom_block = block
                while is_block_on_top:
                    edge_features[self.pb_blocks[bottom_block], self.pb_blocks[top_block], 0] = 1.
                    bottom_block = top_block
                    is_block_on_top, top_block = block_on_top(bottom_block, state)
        return object_features, edge_features


    def action_to_vec(self, action):
        if action.name == 'place':
            top_block_num = self.pb_blocks[action.args[0]]
            bottom_block_num = self.pb_blocks[action.args[2]]
        elif action.name == 'stack':
            top_block_num = action.args[0]
            bottom_block_num = action.args[1]
        return np.array([top_block_num, bottom_block_num])


    # init keys for all potential actions
    def all_optimistic_actions(self):
        actions = []
        for bb in range(1, self.num_blocks+1):
            for bt in range(1, self.num_blocks+1):
                if bb != bt:
                    action = Action(name='stack', args=(bt, bb))
                    actions.append(action)
        return actions


    def action_args_to_action(self, top_block_num, bottom_block_num):
        return Action(name='stack', args=(top_block_num, bottom_block_num))


    def transition(self, pddl_state, fd_state, pddl_action, fd_action):
        new_fd_state = copy(fd_state)
        if self.valid_transition(pddl_action):
            apply_action(new_fd_state, fd_action) # apply action in PDDL model
            if self.use_panda:
                print('Executing action: ', pddl_action)
                self.panda.execute_action(pddl_action, self.fixed, world_obstacles=list(self.pb_blocks))
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
        return new_pddl_state, new_fd_state, self.valid_transition(pddl_action)


    # NOTE: in physical domains, to evaluate if a transition matched the optimistic model,
    # we will have to see if the resulting physical state matches the resulting PDDL
    # state. This will require a method of going from the physical world to a PDDL
    # respresentation of the state.
    def valid_transition(self, pddl_action):
        if pddl_action.name == 'stack':
            return pddl_action.args[0] == pddl_action.args[1]+1
        else:
            if pddl_action.name == 'place':
                return self.pb_blocks[pddl_action.args[0]] == self.pb_blocks[pddl_action.args[2]]+1
            else:
                return True # all other actions are valid

int_to_str_dict = {2:'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
            7: 'seven', 8: 'eight'}
def int_to_str(int):
    return int_to_str_dict[int]

block_colors = [(255, 0 , 0, 1), (255, 1, 0, 1), (255, 255, 0, 1),
                    (0, 255, 0, 1), (0, 0, 255, 1), (148, 0, 130, 1), (1, 0, 211, 1)]
def block_to_urdf(block_num):
    I = 0.001
    side = 0.05
    mass = 0.1
    color = block_colors[(block_num % len(block_colors)) -1]
    link_urdf = odio_urdf.Link(str(block_num),
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

    block_urdf = odio_urdf.Robot(link_urdf, name=str(block_num))
    file_name = '%i.urdf' % block_num
    path = os.path.join('pb_robot', 'models', file_name)
    with open(path, 'w') as handle:
        handle.write(str(block_urdf))
    return file_name
