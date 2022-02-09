import matplotlib.pyplot as plt
import numpy as np
import argparse

from experiments.utils import ExperimentLogger


dataset_paths = ['logs/experiments/test_dataset_progress0p0-20220127-194636',
                    'logs/experiments/test_dataset_progress0p1-20220127-195002',
                    'logs/experiments/test_dataset_progress0p2-20220127-195037',
                    'logs/experiments/test_dataset_progress0p3-20220127-195109',
                    'logs/experiments/test_dataset_progress0p4-20220127-200542',
                    'logs/experiments/test-dataset-gp5',
                    'logs/experiments/test_dataset_progress0p6-20220127-201713',
                    'logs/experiments/test_dataset_progress0p7-20220127-202811',
                    'logs/experiments/test_dataset_progress0p8-20220127-195212',
                    'logs/experiments/test_dataset_progress0p9-20220127-202903',
                    'logs/experiments/test_dataset_progress1p0-20220127-202921']

def train():
    for path in dataset_paths:
        logger = ExperimentLogger(path)
        dataset = logger.load_dataset('trans')
        for i in range(5):

            n_total = len(dataset)
            n_successes = sum([y for x,y in dataset])
            percent_success = n_successes / n_total
            print(path)
            print('    %f' % percent_success)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # if restarting an experiment, only need to set the following 2 arguments
    # (all other args will be taken from the initial run)
    parser.add_argument('--restart',
                        action='store_true',
                        help='use if want to restart from a crash (must also pass in exp-path)')
    parser.add_argument('--exp-path',
                        type=str,
                        help='the exp-path to restart from')

    # Data collection args
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--dataset-path',
                        type=str,
                        help='domain to generate data from')
    parser.add_argument('--max-actions',
                        type=int,
                        default=400,
                        help='max number of actions for the robot to attempt')
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--data-collection-mode',
                        type=str,
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned', \
                                'sequential-plans', 'sequential-goals', 'engineered-goals-dist', \
                                'engineered-goals-size'],
                        help='method of data collection')
    parser.add_argument('--train-freq',
                        type=int,
                        default=10,
                        help='number of actions between model training')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--n-seq-plans',
                        type=int,
                        default=100,
                        help='number of plans used to generate search space for sequential methods')
    # Training args
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
                        default=32,
                        help='number of hidden units in network')
    parser.add_argument('--n-layers',
                        type=int,
                        default=5,
                        help='number of layers in GNN node and edge networks')
    parser.add_argument('--n-models',
                        type=int,
                        default=5,
                        help='number of models in ensemble')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    train()
