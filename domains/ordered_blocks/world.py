import os
import odio_urdf
import numpy as np
from copy import copy

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
    def init(domain_args, pddl_model_type, logger=None):
        num_blocks = int(domain_args[0])
        use_panda = domain_args[1] == 'True'
        world = OrderedBlocksWorld(num_blocks, use_panda)
        opt_pddl_info, pddl_info = world.get_pddl_info(pddl_model_type, logger)
        return world, opt_pddl_info, pddl_info


    def __init__(self, num_blocks, use_panda):
        self.num_blocks = num_blocks  # NOTE: 1 < num_blocks <= 8
        self.use_panda = use_panda

        if self.use_panda:
            self.panda = PandaAgent()
            self.panda.plan()
            self.pb_blocks = self.place_blocks()
            self.panda.execute()
            self.pb_blocks = self.place_blocks()


    # world frame aligns with the robot base
    def place_blocks(self):
        ys = np.linspace(-.4, .4, self.num_blocks)
        x = 0.3
        xy_points = [(x, y) for y in ys]
        pb_blocks = []
        for block_num, xy_point in zip(range(1, self.num_blocks+1), xy_points):
            file_name = block_to_urdf(block_num)
            pb_block = pb_robot.body.createBody(os.path.join('models', file_name))
            # NOTE: for now assumes no relative rotation between robot base/world frame and object
            z_point = pb_block.get_dimensions()[2]/2
            pb_block.set_point([*xy_point, z_point])
            pb_blocks.append(pb_blocks)
        return pb_blocks


    def get_pddl_info(self, pddl_model_type, logger):
        if self.use_panda:
            fixed = self.pb_blocks+self.panda.fixed
            robot = self.panda.planning_robot
            optimistic_domain_pddl = read('domains/ordered_blocks/panda_domain/optimistic/domain.pddl')
            optimistic_stream_pddl = read('domains/ordered_blocks/panda_domain/optimistic/streams.pddl')
            optimistic_stream_map = {
                'plan-free-motion': from_fn(get_free_motion_gen(robot,
                                                                fixed)),
                'plan-holding-motion': from_fn(get_holding_motion_gen(robot,
                                                                        fixed)),
                'pick-inverse-kinematics': from_fn(get_ik_fn(robot,
                                                            fixed,
                                                            approach_frame='gripper',
                                                            backoff_frame='global')),
                'place-inverse-kinematics': from_fn(get_ik_fn(robot,
                                                                fixed,
                                                                approach_frame='global',
                                                                backoff_frame='gripper')),
                'sample-pose-block': from_fn(get_pose_gen_block(fixed)),
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


    def get_init_state(self):
        pddl_state = []
        for bi in range(1, self.num_blocks+1):
            pddl_state += [('clear', bi), ('ontable', bi), ('block', bi)]
        if self.use_panda:
            pddl_state += self.panda.get_init_state()
        return pddl_state


    def generate_random_goal(self, feasible=False):
        random_top_block = np.random.randint(2, self.num_blocks+1)
        if feasible:
            random_height = np.random.randint(2, random_top_block+1)
        else:
            random_height = np.random.randint(2, self.num_blocks+1)
        return ('height%s' % int_to_str(random_height), random_top_block)


    def generate_curriculum_goal(self, current_t, max_t, new=False):
        '''
        Old method is to increase the maximum random block num used in a random
        goal in each time zone. New method is to increase the maximum random
        height used to generate a random goal in each time zone. It also shortens
        the time zones.
        '''
        if current_t >= max_t:
            return self.generate_random_goal()
        if not new:
            max_blocks = range(2, self.num_blocks+1)
            t_times = np.linspace(0, max_t, len(max_blocks)+1)
            for t0, t1 in zip(t_times[:-1], t_times[1:]):
                if current_t >= t0 and current_t < t1:
                    t_index = np.where(t_times==t0)[0][0]
                    break
            max_block = max_blocks[t_index]
            random_top_block = np.random.randint(2, max_block+1)
            random_height = np.random.randint(2, max_block+1)
            return ('height%s' % int_to_str(random_height), random_top_block)
        else:
            max_heights = range(2, self.num_blocks+1)
            t_times = np.linspace(0, max_t, len(max_heights)+1)
            for t0, t1 in zip(t_times[:-1], t_times[1:]):
                if current_t >= t0 and current_t < t1:
                    t_index = np.where(t_times==t0)[0][0]
                    break
            max_height = max_heights[t_index]
            #random_height = np.random.randint(2, max_height+1)
            random_height = max_height
            random_top_block = np.random.randint(2, self.num_blocks+1)
            return ('height%s' % int_to_str(random_height), random_top_block)

    # TODO: is there a way to sample random actions using PDDL code?
    def random_optimistic_action(self, state):
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
        return action


    def state_to_vec(self, state):
        simple_state = get_simple_state(state)
        def block_on_top(bottom_block_num, state):
            for top_block_num in range(1, self.num_blocks):
                if ('on', top_block_num, bottom_block_num) in simple_state:
                    return True, top_block_num
            return False, None

        object_features = np.expand_dims(np.arange(self.num_blocks+1), 1)
        edge_features = np.zeros((self.num_blocks+1, self.num_blocks+1, 1))
        # for each block on the table, recursively check which blocks are on top of it
        for block_num in range(1, self.num_blocks+1):
            if ('ontable', block_num) in simple_state:
                edge_features[0, block_num, 0] = 1.
                bottom_block_num = block_num
                is_block_on_top, top_block_num = block_on_top(bottom_block_num, state)
                while is_block_on_top:
                    edge_features[bottom_block_num, top_block_num, 0] = 1.
                    bottom_block_num = top_block_num
                    is_block_on_top, top_block_num = block_on_top(bottom_block_num, state)
        return object_features, edge_features


    def action_to_vec(self, action):
        return np.array([action.args[0], action.args[1]])


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


    # NOTE: in physical domains, to evaluate if a transition matched the optimistic model,
    # we will have to see if the resulting physical state matches the resulting PDDL
    # state. This will require a method of going from the physical world to a PDDL
    # respresentation of the state.
    def transition(self, pddl_state, fd_state, pddl_action, fd_action):
        new_fd_state = copy(fd_state)
        if self.valid_transition(pddl_action):
            apply_action(new_fd_state, fd_action) # apply action in PDDL model
            if self.use_panda:
                pass # TODO: call panda controllers to place blocks
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
        return new_pddl_state, new_fd_state, self.valid_transition(pddl_action)


    def valid_transition(self, pddl_action):
        return pddl_action.args[0] == pddl_action.args[1]+1

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
