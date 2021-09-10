import os
import odio_urdf
import numpy as np
from copy import copy

import pb_robot

from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.language.generator import from_fn
from pddlstream.algorithms.downward import fact_from_fd, apply_action

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import predicate_in_state
from domains.ordered_blocks.learned_pddl.primitives import get_trust_model

class OrderedBlocksWorld:
    def __init__(self, args):
        self.num_blocks = int(args[0])  # NOTE: must be greater than 1!
        self.use_panda = bool(args[1])

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


    def get_pddl_info(self, planning_model_type, logger=None):
        #if self.use_panda:
        #    if planning_model_type == 'optimistic':
        #        pass # TODO: make new domain that includes pick and place
        #    if planning_model_type == 'learned':
        #        pass # TODO
        #else:
        if planning_model_type == 'optimistic':
            domain_pddl = read('domains/ordered_blocks/optimistic_pddl/domain.pddl')
            stream_pddl = None
            stream_map = {}
        elif planning_model_type == 'learned':
            domain_pddl = read('domains/ordered_blocks/learned_pddl/domain.pddl')
            stream_pddl = read('domains/ordered_blocks/learned_pddl/streams.pddl')
            stream_map = {'get-trust-model': from_fn(get_trust_model(self, logger))}
        constant_map = {}
        return [domain_pddl, constant_map, stream_pddl, stream_map]

    def get_init_state(self):
        pddl_state = []
        for bi in range(1, self.num_blocks+1):
            pddl_state += [('clear', bi), ('ontable', bi), ('block', bi)]
        if self.use_panda:
            pddl_state += self.panda.get_init_state()
        return pddl_state

    def generate_random_goal(self):
        top_block_num = np.random.randint(2, self.num_blocks+1)
        return ('on', top_block_num, top_block_num-1)

    # TODO: is there a way to sample random actions using PDDL code?
    def random_action(self, state):
        action = None
        table_blocks = [bn for bn in range(1, self.num_blocks+1)
                if predicate_in_state(('ontable', bn), state) and predicate_in_state(('clear', bn), state)]
        if len(table_blocks) > 0:
            top_block_idx = np.random.choice(len(table_blocks))
            top_block_num = table_blocks[top_block_idx]
            possible_bottom_blocks = []
            for bn in range(1, self.num_blocks+1):
                if predicate_in_state(('clear', bn), state) and bn != top_block_num:
                    possible_bottom_blocks.append(bn)
            bottom_block_idx = np.random.choice(len(possible_bottom_blocks))
            bottom_block_num = possible_bottom_blocks[bottom_block_idx]
            action = Action(name='stack', args=(top_block_num, bottom_block_num))
        return action

    # TODO: remove this check from random-actions as it assumes domain information
    # should instead just explore for a number of time steps
    def valid_actions_exist(self, state):
        # for each clear block on the table (which will be placed as a top block)
        table_blocks = []
        for block_num in range(1, self.num_blocks+1):
            if predicate_in_state(('ontable', block_num), state) and \
                        predicate_in_state(('clear', block_num), state):
                table_blocks.append(block_num)
        # check if the correct corresponding bottom block is clear
        for table_block in table_blocks:
            if table_block != 1:
                bottom_block_num = table_block-1
                if predicate_in_state(('clear', bottom_block_num), state):
                    return True
        return False

    def state_to_vec(self, state):
        def block_on_top(bottom_block_num, state):
            for top_block_num in range(1, self.num_blocks):
                if predicate_in_state(('on', top_block_num, bottom_block_num), state):
                    return True, top_block_num
            return False, None

        object_features = np.expand_dims(np.arange(self.num_blocks+1), 1)
        edge_features = np.zeros((self.num_blocks+1, self.num_blocks+1, 1))
        # for each block on the table, recursively check which blocks are on top of it
        for block_num in range(1, self.num_blocks+1):
            if predicate_in_state(('ontable', block_num), state):
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
    def all_optimistic_actions(self, num_blocks):
        pos_actions = []
        neg_actions = []
        for bb in range(1, num_blocks+1):
            for bt in range(1, num_blocks+1):
                if bt == bb+1:
                    pos_actions.append(str(bt)+','+str(bb))
                elif bt != bb:
                    neg_actions.append(str(bt)+','+str(bb))
        return pos_actions, neg_actions

    def action_args_to_action(self, top_block_num, bottom_block_num):
        return Action(name='stack', args=(top_block_num, bottom_block_num))


    # NOTE: in physical domains, to evaluate if a transition matched the optimistic model,
    # we will have to see if the resulting physical state matches the resulting PDDL
    # state. This will require a method of going from the physical world to a PDDL
    # respresentation of the state.
    def transition(self, pddl_state, fd_state, pddl_action, fd_action):
        new_fd_state = copy(fd_state)
        valid_transition = pddl_action.args[0] == pddl_action.args[1]+1
        if valid_transition:
            apply_action(new_fd_state, fd_action) # apply action in PDDL model
            if self.use_panda:
                pass # TODO: call panda controllers to place blocks
        pddl_state = [fact_from_fd(sfd) for sfd in fd_state]
        new_pddl_state = [fact_from_fd(sfd) for sfd in new_fd_state]
        return new_pddl_state, new_fd_state, valid_transition


block_colors = [(255, 0 , 0, 1), (255, 1, 0, 1), (255, 255, 0, 1),
                    (0, 255, 0, 1), (0, 0, 255, 1), (148, 0, 130, 1), (1, 0, 211, 1)]
def block_to_urdf(block_num):
    I = 0.001
    side = 0.05
    mass = 0.1
    color = block_colors[(block_num % len(block_colors)) -1]
    print(color)
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