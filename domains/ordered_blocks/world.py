import os
import numpy as np
import random
import matplotlib.pyplot as plt

import pb_robot

from pddlstream.language.constants import Action
from pddlstream.utils import read
from pddlstream.language.generator import from_list_fn, from_fn

from panda_wrapper.panda_agent import PandaAgent
from tamp.utils import get_simple_state, get_learned_pddl, block_to_urdf
from domains.ordered_blocks.panda.primitives import get_free_motion_gen, \
    get_holding_motion_gen, get_ik_fn, get_pose_gen_block, get_grasp_gen
from domains.ordered_blocks.add_to_primitives import get_trust_model
from learning.utils import model_forward


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
            self.obstacles = list(self.pb_blocks)
        else:
            self.nb_blocks = {}
            for n in range(1,self.num_blocks+1):
                nb_block = NumberedBlock(n)
                self.nb_blocks[nb_block] = n
            self.table = Table()

        self.blocks = self.pb_blocks if self.use_panda else self.nb_blocks
        self.table = self.panda.table if self.use_panda else self.table
        self.init_state = self.get_init_state()

        # GNN model params
        self.n_of_in = 1
        self.n_ef_in = 1
        self.n_af_in = 2


    def get_init_state(self):
        pddl_state = []
        shuffled_blocks = list(self.blocks)
        random.shuffle(shuffled_blocks)
        for b in shuffled_blocks:
            pddl_state += [('clear', b),
                            ('block', b),
                            ('on', b, self.table)]
            if self.use_panda:
                pose = pb_robot.vobj.BodyPose(b, b.get_base_link_pose())
                pddl_state += [('pose', b, pose),
                                ('atpose', b, pose)]
        pddl_state += [('table', self.table)]
        if self.use_panda:
            pddl_state += self.panda.get_init_state()
        return pddl_state


    # NOTE: this reset looks like it works but then planning fails
    def reset(self):
        def reset_blocks():
            for pb_block, block_pose in self.orig_poses.items():
                pb_block.set_base_link_pose(block_pose)
        if self.use_panda:
            self.panda.plan()
            reset_blocks()
            self.panda.execute()
            reset_blocks()
            self.panda.reset()


    def disconnect(self):
        if self.use_panda:
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
            fname = '%i.urdf' % block_num
            block_to_urdf(str(block_num),
                            os.path.join('pb_robot/models', fname),
                            block_colors[(block_num % len(block_colors))][1])
            pb_block = pb_robot.body.createBody(os.path.join('models', fname))
            # NOTE: for now assumes no relative rotation between robot base/world frame and object
            z = pb_robot.placements.stable_z(pb_block, self.panda.table)
            block_pose = ((*xy_point, z), (0., 0., 0., 1.))
            pb_block.set_base_link_pose(block_pose)
            pb_blocks[pb_block] = block_num
            orig_poses[pb_block] = block_pose
        return pb_blocks, orig_poses


    def get_pddl_info(self, pddl_model_type, logger):
        add_to_domain_path = 'domains/ordered_blocks/add_to_domain.pddl'
        add_to_streams_path = 'domains/ordered_blocks/add_to_streams.pddl'
        if self.use_panda:
            robot = self.panda.planning_robot
            opt_domain_pddl_path = 'domains/ordered_blocks/panda/domain.pddl'
            opt_streams_pddl_path = 'domains/ordered_blocks/panda/streams.pddl'
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
        else:
            opt_domain_pddl_path = 'domains/ordered_blocks/discrete/domain.pddl'
            opt_streams_pddl_path = None
            opt_streams_map = {}

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


    def generate_random_goal(self, feasible=False):
        random_top_block = random.choice(list(self.blocks))
        if feasible:
            top_block_num = self.blocks[random_top_block]
            random_height = np.random.randint(2, top_block_num+1)
        else:
            random_height = np.random.randint(2, self.num_blocks+1)
        goal = ('height%s' % int_to_str(random_height), random_top_block)
        return goal


    # TODO: is there a way to sample random actions using PDDL code?
    def random_actions(self, state):
        def random_block(cond):
            blocks = list(self.blocks)
            random.shuffle(blocks)
            for block in blocks:
                if cond(block):
                    if self.use_panda:
                        for fluent in state:
                            if fluent[0] == 'atpose' and fluent[1] == block:
                                pose = fluent[2]
                                return block, pose
                    else:
                        return block, None
            return None, None

        clear = lambda block : ('clear', block) in state
        on_table_clear = lambda block : (('on', block, self.table) in state) and clear(block)

        # TODO: this only picks and places a single block. want it to work until
        # infeasible action is attempted
        timeout = 15
        if self.use_panda:
            # pick action
            grasp_fn = get_grasp_gen(self.panda.planning_robot)
            pick_fn = get_ik_fn(self.panda.planning_robot,
                                self.fixed,
                                approach_frame='gripper',
                                backoff_frame='global')

            top_block, top_block_pose = random_block(on_table_clear)
            if not top_block:
                return [], []

            grasps = grasp_fn(top_block)
            t = 0
            while t < timeout:
                try:
                    grasp_i = np.random.randint(len(grasps))
                    grasp = grasps[grasp_i][0]
                    pick_init_conf, pick_final_conf, pick_traj = pick_fn(top_block,
                                                                    top_block_pose,
                                                                    grasp)
                    print('Found successful pick grasp and traj.')
                    break
                except:
                    t += 1
                    print('Searching for pick grasp and traj.')
                    pass

            pick_pre = ('pickkin',
                        top_block,
                        top_block_pose,
                        grasp,
                        pick_init_conf,
                        pick_final_conf,
                        pick_traj)
            pick_action = Action(name='pick', args=(top_block,
                                                    top_block_pose,
                                                    self.panda.table,
                                                    grasp,
                                                    pick_init_conf,
                                                    pick_final_conf,
                                                    pick_traj))

            # place action
            place_fn = get_ik_fn(self.panda.planning_robot,
                                    self.fixed,
                                    approach_frame='global',
                                    backoff_frame='gripper')
            supported_fn = get_pose_gen_block(self.fixed)

            bottom_block = top_block
            while bottom_block == top_block:
                bottom_block, bottom_block_pose = random_block(clear)
                if not bottom_block:
                    return [], []

            t = 0
            while t < timeout:
                try:
                    top_block_place_pose = supported_fn(top_block, bottom_block, bottom_block_pose)[0]
                    place_init_conf, place_final_conf, place_traj = place_fn(top_block,
                                                                    top_block_place_pose,
                                                                    grasp)
                    print('Found successful place traj.')
                    break
                except:
                    t += 1
                    print('Searching for place traj.')
                    pass

            place_pre = ('placekin',
                        top_block,
                        top_block_place_pose,
                        grasp,
                        place_init_conf,
                        place_final_conf,
                        place_traj)
            supported_pre = ('supported',
                            top_block,
                            top_block_place_pose,
                            bottom_block,
                            bottom_block_pose)
            place_action = Action(name='place', args=(top_block,
                                                        top_block_place_pose,
                                                        bottom_block,
                                                        bottom_block_pose,
                                                        grasp,
                                                        place_init_conf,
                                                        place_final_conf,
                                                        place_traj))

            # move free action
            move_free_fn = get_free_motion_gen(self.panda.planning_robot, self.fixed)

            for fluent in state:
                if fluent[0] == 'atconf':
                    init_conf = fluent[1]

            t = 0
            while t < timeout:
                try:
                    move_free_traj = move_free_fn(init_conf, pick_init_conf)[0]
                    print('Found successful move free traj.')
                    break
                except:
                    t += 1
                    print('Searching for move free traj.')
                    pass

            move_free_pre = ('freemotion', init_conf, move_free_traj, pick_init_conf)
            move_free_action = Action(name='move_free', args=(init_conf, pick_init_conf, move_free_traj))

            # move holding action
            move_holding_fn = get_holding_motion_gen(self.panda.planning_robot, self.fixed)

            t = 0
            while t < timeout:
                try:
                    move_holding_traj = move_holding_fn(pick_final_conf, place_init_conf, top_block, grasp)[0]
                    print('Found successful move holding traj.')
                    break
                except:
                    t += 1
                    print('Searching for move holding traj.')
                    pass

            move_holding_pre = ('holdingmotion',
                                pick_final_conf,
                                move_holding_traj,
                                place_init_conf,
                                top_block,
                                grasp)
            move_holding_action = Action(name='move_holding', args=(pick_final_conf,
                                                                    place_init_conf,
                                                                    top_block,
                                                                    grasp,
                                                                    move_holding_traj))

            actions = [move_free_action, pick_action, move_holding_action, place_action]
            add_fluents = [move_free_pre, pick_pre, place_pre, supported_pre, move_holding_pre]
            return actions, add_fluents
        else:
            top_block, _ = random_block(on_table_clear)
            if top_block:
                bottom_block = top_block
                while bottom_block == top_block:
                    bottom_block, _ = random_block(clear)
                if bottom_block:
                    return [Action(name='pickplace', args=(top_block, self.table, bottom_block))], []
            return [], []


    def state_to_vec(self, state, num_blocks=None):
        if not num_blocks: num_blocks = self.num_blocks
        assert num_blocks <= self.num_blocks, 'can\'t ask for more block states than ' \
                                                'there are blocks in the world'
        blocks = {block : block_num for block, block_num in self.blocks.items() \
                                                if block_num <= num_blocks}
        state = get_simple_state(state)
        def block_on_top(bottom_block):
            for top_block in blocks:
                if ('on', top_block, bottom_block) in state:
                    return True, top_block
            return False, None

        object_features = np.expand_dims(np.arange(num_blocks+1), self.n_of_in)
        edge_features = np.zeros((num_blocks+1, num_blocks+1, self.n_ef_in))
        # for each block on the table, recursively check which blocks are on top of it
        for block in blocks:
            if ('on', block, self.table) in state:
                edge_features[0, blocks[block], 0] = 1.
                is_block_on_top, top_block = block_on_top(block)
                bottom_block = block
                while is_block_on_top:
                    edge_features[blocks[bottom_block], blocks[top_block], 0] = 1.
                    bottom_block = top_block
                    is_block_on_top, top_block = block_on_top(bottom_block)
        return object_features, edge_features


    def action_to_vec(self, action):
        # NOTE this is a bit hacky. should get indices from param names ?bt and ?bb
        top_block_num = self.blocks[action.args[0]]
        bottom_block_num = self.blocks[action.args[2]]
        return np.array([top_block_num, bottom_block_num])


    # init keys for all potential actions
    def all_optimistic_actions(self, num_blocks=None):
        if self.use_panda:
            print('WARNING: world.all_optimistic_actions() only enumerates place' \
                    'actions and doesn\'t ground all variables')
        actions = []
        if not num_blocks: num_blocks = self.num_blocks
        assert num_blocks <= self.num_blocks, 'can\'t ask for more block actions than ' \
                                                'there are blocks in the world'
        blocks = {block : block_num for block, block_num in self.blocks.items() \
                                                if block_num <= num_blocks}
        for bb in blocks:
            for bt in blocks:
                if bb != bt:
                    if self.use_panda:
                        action = Action(name='place', args=(bt, None, bb, None, None, None, None, None))
                    else:
                        action = Action(name='pickplace', args=(bt, self.table, bb))
                    actions.append(action)
        return actions


    def action_args_to_action(self, top_block, bottom_block):
        return Action(name='pickplace', args=(top_block, self.table, bottom_block))


    # NOTE: in physical domains, to evaluate if a transition matched the optimistic model,
    # we will have to see if the resulting physical state matches the resulting PDDL
    # state. This will require a method of going from the physical world to a PDDL
    # respresentation of the state.
    # NOTE this is a bit hacky. should get indices from param names ?bt and ?bb
    def valid_transition(self, new_pddl_state, pddl_action):
        if 'place' in pddl_action.name:
            return self.blocks[pddl_action.args[0]] == self.blocks[pddl_action.args[2]]+1
        else:
            return True # all other actions are valid


    def add_goal_text(self, goal):
        height_str = goal[0]
        for hi, hstr in int_to_str_dict.items():
            if hstr == height_str[-1*len(hstr):]:
                height = hi
        print(height, '!!')
        top_block_num = self.blocks[goal[1]]
        goal_feas = top_block_num >= height
        if self.use_panda:
            self.panda.add_text('Planning for Goal: (%s, %s)' % (goal[0], block_colors[self.blocks[goal[1]]][0]),
                position=(0, -1, 1),
                size=1.5,
                color = (0, 255, 0, 1) if goal_feas else (255, 0 , 0, 1))


    def plot_model_accuracy(self, i, model):
        fig, ax = plt.subplots()
        # NOTE: we test all actions from initial state assuming that the network is ignoring the state
        preds = np.zeros((self.num_blocks, self.num_blocks))
        all_actions = self.all_optimistic_actions()
        orig_use_panda = self.use_panda
        self.use_panda = False
        init_state = self.get_init_state()
        vof, vef = self.state_to_vec(init_state)
        for action in all_actions:
            va = self.action_to_vec(action)
            model_pred = model_forward(model, [vof, vef, va]).squeeze()#.round().squeeze()
            preds[va[0]-1][va[1]-1] = model_pred
        im = ax.imshow(preds, cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1)
        ax.set_aspect('equal')
        fig.colorbar(im, orientation='vertical')
        ax.set_title('Feasibility Predictions, Iteration %i' % i)
        ax.set_xlabel('Bottom Block')
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['red', 'orange', 'yellow', 'green'])
        ax.set_ylabel('Top Block')
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(['red', 'orange', 'yellow', 'green'])
        #plt.show()
        self.use_panda = orig_use_panda


class NumberedBlock:
    def __init__(self, num):
        self.num = num

    def __repr__(self):
        return '{}'.format(self.num % 1000)

class Table:
    def __init__(self):
        pass

    def __repr__(self):
        return 'table'

int_to_str_dict = {2:'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
            7: 'seven', 8: 'eight'}
def int_to_str(int):
    return int_to_str_dict[int]

block_colors = {1:('red', (255, 0 , 0, 1)),
                2: ('orange', (255, 1, 0, 1)),
                3: ('yellow', (255, 255, 0, 1)),
                4: ('green', (0, 255, 0, 1)),
                5: ('blue', (0, 0, 255, 1)),
                6: ('indigo', (148, 0, 130, 1)),
                7: ('violet', (1, 0, 211, 1))}
