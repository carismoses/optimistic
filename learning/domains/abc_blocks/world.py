from copy import copy, deepcopy
import csv
import numpy as np

from tamp.predicates import On

TABLE_NUM = 0

### Object Classes

class Object:
    def __init__(self, num):
        self.num = num


class Block(Object):
    def __init__(self, num):
        super(Block, self).__init__(num)
        self.name = 'block_%i' % num


class Table(Object):
    def __init__(self, num):
        super(Table, self).__init__(num) # make 1 when using * as 0
        self.name = 'table'


### World and State Classes

class ABCBlocksWorld:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.num_objects = num_blocks + 1 # table is also an object
        self.min_block = 1                # 0 is the table
        self.max_block = num_blocks

        self.table = Table(TABLE_NUM)
        self._blocks = {i: Block(i) for i in range(self.min_block, self.max_block+1)}

    def get_init_state(self):
        raise NotImplementedError

    def transition(self, state, action):
        raise NotImplementedError

    def random_policy(self, state):
        action = None
        top_block_num = np.random.choice(list(self._blocks))
        bottom_block_num = np.random.choice(list(self._blocks))
        return (bottom_block_num, top_block_num)

    def reward(self, state, goal):
        return 1 if self.is_goal_state(state, goal) else 0

    def is_goal_state(self, state, goal):
        raise NotImplementedError


# Ground Truth Blocks World
class ABCBlocksWorldGT(ABCBlocksWorld):
    def __init__(self, num_blocks):
        super().__init__(num_blocks)

    def get_init_state(self):
        return LogicalState(self._blocks, self.num_objects, self.table)

    def transition(self, state, action):
        new_state = state.copy()
        if action is not None:
            bottom_block_num = action[0]
            top_block_num = action[1]
            # can only stack blocks by increments of one and top block must be on table
            if top_block_num == bottom_block_num + 1 and \
                top_block_num not in state.stacked_blocks:
                # add both if bottom block is on table and this is the start of the stack
                if bottom_block_num not in state.stacked_blocks and \
                                        len(state.stacked_blocks) == 0:
                    new_state.stacked_blocks.append(bottom_block_num)
                    new_state.stacked_blocks.append(top_block_num)
                # if bottom block is on top of stack, can stack top block (can only build one stack at a time for now)
                elif bottom_block_num == state.stacked_blocks[-1]:
                    new_state.stacked_blocks.append(top_block_num)
        #print('new stack:', new_state.stacked_blocks)
        return new_state

    # attempt to stack a block that is currently on the table
    def random_remaining_policy(self, state):
        action = None
        remaining_blocks = list(set(self._blocks.keys()).difference(set(self.stacked_blocks)))
        if len(remaining_blocks) > 0:
            top_block_idx = np.random.choice(len(remaining_blocks))
            top_block_num = remaining_blocks[top_block_idx].num
            if len(state.stacked_blocks) > 0:
                bottom_block_num = state.stacked_blocks[-1]
                action = (bottom_block_num, top_block_num)
            else:
                possible_bottom_blocks = list(set(remaining_blocks).difference(set([top_block])))
                if len(possible_bottom_blocks) > 0:
                    bottom_block_idx = np.random.choice(len(possible_bottom_blocks))
                    bottom_block_num = possible_bottom_blocks[bottom_block_idx].num
                    action = (bottom_block_num, top_block_num)
        return action

    def expert_policy(self, state):
        action = None
        if len(state.stacked_blocks) > 0:
            bottom_block_num = state.stacked_blocks[-1]
            if bottom_block_num != self.max_block:
                top_block_num = bottom_block_num + 1
                action = (bottom_block_num, top_block_num)
        else: # start stack
            action = (self.min_block, self.min_block+1)
        return action

    # goal state is already logical
    def is_goal_state(self, state, goal):
        in_goal = True
        for goal_pred in goal:
            in_goal = in_goal and goal_pred.in_state(state.as_logical())
        return in_goal

# When using learned model for transitions, edge states won't always make sense as logical states,
# so need a separate World class (eg. 2 blocks can be on top of one in a vectorized edge state)

# Learned Blocks World
class ABCBlocksWorldLearned(ABCBlocksWorld):
    def __init__(self, args, num_blocks):
        super().__init__(num_blocks)
        self.object_features = np.expand_dims(np.arange(self.num_objects), 1) # for now object features are static
        if args.model_path:
            self.model_path = args.model_path

    def get_init_state(self):
        edge_features = np.zeros((self.num_objects, self.num_objects, 1))
        # edge_feature[i, j, 0] == 1 if j on i, else 0
        # initially everything on table
        for block_num in blocks.keys():
            edge_features[self.table.num, block_num, 0] = 1.
        return self.object_features, edge_features

    def transition(self, state, action):
        object_features, edge_features = state
        model = torch.load_state(self.model_path)
        delta_edge_features = model(object_features, edge_features, vec_action)
        new_state = edge_features + delta_edge_features
        return new_state

    def is_goal_state(self, state, goal):
        goal_idxs = np.where(goal == 1)
        goal_reached = True
        for goal_idx in zip(goal_idxs.T):
            # NOTE: This only works when vactorized states are 2D
            goal_reached = goal_reached and (round(state[goal_idx[0]][goal_idx[1]]) == 1)
        return goal_reached


# In the ground truth world the state is separate from world so that we can get different
# state representations (logical and vectorized)
class LogicalState:
    def __init__(self, blocks, num_objects, table):
        self.table = table
        self.blocks = blocks
        self.stacked_blocks = []
        self.num_objects = num_objects

    def as_logical(self, pprint=False):
        logical_state = []

        # stacked blocks
        if len(self.stacked_blocks) > 0:
            logical_state.append(On(self.table, self.blocks[self.stacked_blocks[0]]))
            if pprint:
                print(self.table.num, self.blocks[self.stacked_blocks[0]].num)
        for bottom_block_num, top_block_num in zip(self.stacked_blocks[:-1], self.stacked_blocks[1:]):
            logical_state.append(On(self.blocks[bottom_block_num], self.blocks[top_block_num]))
            if pprint:
                print(bottom_block_num, top_block_num)

        # remaining blocks on table
        for block_num, block in self.blocks.items():
            if block_num not in self.stacked_blocks:
                logical_state.append(On(self.table, block))
                if pprint:
                    print(self.table.num, block_num)
        if pprint:
            print('---')
        return logical_state

    def as_vec(self):
        object_features = np.expand_dims(np.arange(self.num_objects), 1)
        edge_features = np.zeros((self.num_objects, self.num_objects, 1))

        # edge_feature[i, j, 0] == 1 if j on i, else 0
        for predicate in self.as_logical():
            bottom_i = predicate.bottom.num
            top_i = predicate.top.num
            edge_features[bottom_i, top_i, 0] = 1.

        return object_features, edge_features

    def copy(self):
        copy_state = LogicalState(self.blocks, self.num_objects, self.table)
        copy_state.stacked_blocks = deepcopy(self.stacked_blocks)
        return copy_state

### Helper Functions
def parse_goals_csv(self, goal_file_path):
    def ground_obj(obj_str):
        if obj_str == 'table':
            return self._table
        elif obj_str == '*':
            return self._star
        else:
            return self._blocks[int(obj_str)]
    goals = []
    with open(goal_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            pred_name = row[0]
            pred_args = row[1:]
            if pred_name == 'On':
                goals.append([On(ground_obj(pred_args[0]), ground_obj(pred_args[1]))])
    return goals
