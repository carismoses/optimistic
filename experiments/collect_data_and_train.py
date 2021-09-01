import torch
import numpy as np
import argparse
from argparse import Namespace
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from pprint import pformat

from tamp.logic import subset
from domains.ordered_blocks.world import OrderedBlocksWorldGT, OrderedBlocksWorldOpt, generate_random_goal
from learning.datasets import TransDataset, HeurDataset
from learning.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from planning import plan
from learning.train import train

# TODO: this doesn't make len(dataset) == args.max_transitions exactly
# because sequences are added in chunks that might push it past the limit
# but will be close

def train_class(args, trans_dataset, heur_dataset, logger):
    # save initial (empty) dataset
    logger.save_trans_dataset(trans_dataset, i=0)

    # initialize and save model
    trans_model = TransitionGNN(n_of_in=1,
                                n_ef_in=1,
                                n_af_in=2,
                                n_hidden=16,
                                pred_type=args.pred_type)
    logger.save_trans_model(trans_model, i=0)

    # initialize planner
    if args.data_collection_mode == 'random-goals-opt':
        args.model_type = 'opt'
    elif args.data_collection_mode == 'random-goals-learned':
        args.model_type = 'learned'

    i = 0
    true_world = OrderedBlocksWorldGT(args.num_blocks)
    while len(trans_dataset) < args.max_transitions:
        print('Iteration %i |dataset| = %i' % (i, len(trans_dataset)))
        if args.data_collection_mode in ['random-goals-opt', 'random-goals-learned']:
            # generate plan to reach random goal
            goal = generate_random_goal(true_world)
            tree, found_plan = plan.run(goal, args, logger.exp_path, model_i=i)
            if found_plan: # found_plan is None if not found
                trajectory = true_world.execute_plan(found_plan)
                logger.save_planning_data(tree, goal, found_plan, i=i)
            else:
                trajectory = random_action_sequence(true_world)
        elif args.data_collection_mode == 'random-actions':
            trajectory = random_action_sequence(true_world)
            goal = None

        # add to dataset and save
        print('Adding trajectory to dataset.')
        add_sequence_to_dataset(args, trans_dataset, heur_dataset, trajectory, goal)

        # initialize and train new model
        if args.train:
            trans_model = TransitionGNN(n_of_in=1,
                                        n_ef_in=1,
                                        n_af_in=2,
                                        n_hidden=16,
                                        pred_type=args.pred_type)
            trans_dataset.set_pred_type(args.pred_type)
            print('Training model.')
            trans_dataloader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=True)
            train(trans_dataloader, trans_model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

        # save new model and dataset
        i += 1
        logger.save_trans_dataset(trans_dataset, i=i)
        logger.save_trans_model(trans_model, i=i)
        print('Saved model to %s' % logger.exp_path)

def random_action_sequence(true_world):
    new_state = true_world.get_init_state()
    valid_actions = True
    action_sequence = []
    while valid_actions:
        state = new_state
        vec_action = true_world.random_policy(state)
        new_state = true_world.transition(state, vec_action)
        action_sequence.append((state, vec_action))
        valid_actions = true_world.expert_policy(new_state) is not None
    vec_action = np.zeros(2)
    action_sequence.append((new_state, vec_action))
    return action_sequence


def add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal):
    opt_world = OrderedBlocksWorldOpt(args.num_blocks)
    def helper(sequence, seq_goal, add_to_trans):
        n = len(sequence)
        object_features, goal_edge_features = seq_goal.as_vec()
        for i in range(n):
            state, vec_action = sequence[i]
            object_features, edge_features = state.as_vec()
            heur_dataset.add_to_dataset(object_features, edge_features, goal_edge_features, n-i-1)
            # add_to_trans: only add to transition model once per sequence (not hindsight subsequences)
            # i < n-1: don't add last action to trans dataset as it doesn't do anything
            if add_to_trans and i < n-1:
                next_state, _ = sequence[i+1]
                object_features, next_edge_features = next_state.as_vec()
                delta_edge_features = next_edge_features-edge_features
                optimistic_accuracy = 1 if opt_world.transition(state, vec_action).is_equal(next_state) \
                                        else 0
                trans_dataset.add_to_dataset(object_features, edge_features, vec_action, next_edge_features, delta_edge_features, optimistic_accuracy)

    # make each reached state a goal (hindsight experience replay)
    for goal_i, (hindsight_goal, _) in enumerate(action_sequence):
        helper(action_sequence[:goal_i+1], hindsight_goal, goal_i == len(action_sequence)-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks'],
                        default='ordered_blocks',
                        help='domain to generate data from')
    parser.add_argument('--num-blocks',
                        type=int,
                        default=4)
    parser.add_argument('--max-transitions',
                        type=int,
                        default=100,
                        help='max number of transitions to save to transition dataset')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save datasets and models to')
    parser.add_argument('--data-collection-mode',
                        type=str,
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned'],
                        required=True,
                        help='method of data collection')
    parser.add_argument('--N',
                        type=int,
                        default=1,
                        help='number of data collection/training runs to perform')
    # Training params
    parser.add_argument('--pred-type',
                        type=str,
                        choices=['delta_state', 'full_state', 'class'],
                        default='class')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=300,
                        help='training epochs')
    parser.add_argument('--n-hidden',
                        type=int,
                        default=16,
                        help='number of hidden units in network')
    parser.add_argument('--train',
                        type=bool,
                        default=True,
                        help='Sometimes just want to collect dataset, dont want to train')
    # Planning params
    parser.add_argument('--num-branches',
                        type=int,
                        default=10,
                        help='number of actions to try from each node')
    parser.add_argument('--timeout',
                        type=int,
                        default=100,
                        help='Iterations to run MCTS')
    parser.add_argument('--c',
                        type=int,
                        default=.01,
                        help='UCB parameter to balance exploration and exploitation')
    parser.add_argument('--max-ro',
                        type=int,
                        default=10,
                        help='Maximum number of random rollout steps')
    parser.add_argument('--value-fn',
                        type=str,
                        choices=['rollout', 'learned'],
                        default='rollout',
                        help='use random model rollouts to estimate node value or learned value')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if not args.domain == 'ordered_blocks':
        NotImplementedError('Only Ordered Blocks world works')

    if not args.pred_type == 'class':
        NotImplementedError('Prediction types != class have not been tested in a while')

    if args.value_fn == 'learned':
        NotImplementedError('Using the learned heuristic instead of rollouts has not been tested in a while')

    paths = []
    for n in range(args.N):
        print('Run %i/%i' % (n+1, args.N))
        trans_dataset = TransDataset()
        heur_dataset = HeurDataset()
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        train_class(args, trans_dataset, heur_dataset, logger)
        paths.append(logger.exp_path)

    # print out all paths
    print('%i runs saved to :' % args.N)
    print('['+pformat(paths[0])+',')
    [print(pformat(path)+',') for path in paths[1:-1]]
    print(pformat(paths[-1])+']')
