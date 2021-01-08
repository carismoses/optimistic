from collections import namedtuple
import argparse
import pickle
import sys
import numpy as np

from planning.tree import Tree
from planning.problems import Tallest, Overhang, Deconstruct
from block_utils import Object
from learning.active.utils import ActiveExperimentLogger

def plan(timeout, blocks, problem, model):
    tree = Tree(blocks)
    for t in range(timeout):
        parent_node_id = tree.get_exp_best_node_expand()
        #print(t, len(tree.nodes[parent_node_id]['tower']), tree.nodes[parent_node_id]['value'])
        sys.stdout.write("Search progress: %i   \r" % (t) )
        sys.stdout.flush()
        new_nodes = problem.sample_actions(tree.nodes[parent_node_id], model)
        for node in new_nodes:
            tree.expand(parent_node_id, node)
    return tree

def plan_mcts(timeout, blocks, problem, model, c=np.sqrt(2)):
    tree = Tree(blocks)
    tallest_tower = [0]
    highest_exp_height = [0]
    highest_value = [0]
    for t in range(timeout):
        #sys.stdout.write("Search progress: %i   \r" % (t) )
        #sys.stdout.flush()
        parent_node_id = tree.traverse(c)
        print(t, len(tree.nodes[parent_node_id]['tower']), tree.nodes[parent_node_id]['value'])
        new_node = problem.sample_action(tree.nodes[parent_node_id], model, discrete=True)
        
        new_node_id = tree.expand(parent_node_id, new_node)
        rollout_value = tree.rollout(new_node_id, problem, model)
        tree.backpropagate(new_node_id, rollout_value)
        
        if len(new_node['tower'])>tallest_tower[-1]:
            tallest_tower.append(len(new_node['tower']))
        else:
            tallest_tower.append(tallest_tower[-1])
            
        if new_node['exp_reward'] > highest_exp_height[-1]:
            highest_exp_height.append(new_node['exp_reward'])
        else:
            highest_exp_height.append(highest_exp_height[-1])
            
        if new_node['value'] > highest_value[-1]:
            highest_value.append(new_node['value'])
        else:
            highest_value.append(highest_value[-1])
    return tallest_tower, highest_exp_height, highest_value, tree
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--block-set-fname', 
                        type=str, 
                        default='',
                        help='path to the block set file. if not set, random 5 blocks generated.')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    parser.add_argument('--timeout',
                        type=int,
                        default=1000,
                        help='max number of iterations to run planner for')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--max-height',
                        type=int,
                        default=5,
                        help='number of blocks in desired tower')
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    n_blocks = 5
    tx = 99
    
    if args.block_set_fname is not '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        block_set = [Object.random(f'obj_{ix}') for ix in range(n_blocks)]
        
    if args.problem == 'tallest':
        problem = Tallest(args.max_height)
    elif args.problem == 'overhang':
        problem = Overhang()
    elif args.problem == 'deconstruct':
        problem = Deconstruct()
        
    logger = ActiveExperimentLogger(args.exp_path)
    ensemble = logger.get_ensemble(tx)
    
    tallest_tower, highest_exp_height, highest_value, tree = \
        plan_mcts(args.timeout, block_set, problem, ensemble, c=np.sqrt(2))
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(3)
    
    xs = list(range(len(tallest_tower)))
    ax[0].plot(xs, tallest_tower, label='tallest tower')
    ax[1].plot(xs, highest_exp_height, label='highest expected height')
    ax[2].plot(xs, highest_value, label='highest node value')
    
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    
    plt.show()