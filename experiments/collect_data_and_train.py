import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.utils import split_and_move_data
from learning.datasets import TransDataset
from experiments.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble
from learning.train import train
from experiments.strategies import collect_trajectory_wrapper
from domains.tools.world import ToolsWorld


def train_class(args, logger, n_actions):
    pddl_model_type = 'learned' if 'learned' in args.data_collection_mode else 'optimistic'

    # get model params
    n_of_in, n_ef_in, n_af_in = ToolsWorld.get_model_params()
    base_args = {'n_of_in': n_of_in,
                'n_ef_in': n_ef_in,
                'n_af_in': n_af_in,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    while n_actions < args.max_actions:
        print('# actions = %i, |current dataset| = %i' % (n_actions, len(logger.load_trans_dataset('curr'))))

        # train at training freq
        if len(logger.load_trans_dataset('curr')) >= args.train_freq:
            # split current data into train and val and train
            train_dataset, val_dataset = split_and_move_data(logger, args.val_ratio)

            # initialize and train new model
            ensemble = Ensemble(TransitionGNN,
                                    base_args,
                                    args.n_models)
            print('Training ensemble.')
            trans_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            if val_dataset:
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            else:
                val_dataloader = None
            for model in ensemble.models:
                train(trans_dataloader, model, val_dataloader=val_dataloader, \
                        n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

            # save model and accuracy plots
            logger.save_trans_model(ensemble, i=n_actions)
            print('Saved model to %s' % logger.exp_path)

        progress = None
        trajectory = collect_trajectory_wrapper(args,
                                                pddl_model_type,
                                                logger,
                                                progress,
                                                separate_process= not args.single_process)
        n_actions += len(trajectory)

        if not trajectory:
            print('Trajectory collection failed.')
        else:
            print('Successfully collected trajectory.')
            if all([t_seg[3] for t_seg in trajectory]):
                print('All feasible actions.')
            else:
                print('Infeasible action attempted.')


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
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks', 'tools'],
                        default='tools',
                        help='domain to generate data from')
    parser.add_argument('--domain-args',
                        nargs='+',
                        help='arguments to pass into desired domain')
    parser.add_argument('--max-actions',
                        type=int,
                        default=400,
                        help='max number of (ALL) actions for the robot to attempt')
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--data-collection-mode',
                        type=str,
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned', \
                                'sequential-plans', 'sequential-goals'],
                        help='method of data collection')
    parser.add_argument('--train-freq',
                        type=int,
                        default=10,
                        help='number of actions (IN DATASET) between model training')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--n-seq-plans',
                        type=int,
                        default=100,
                        help='number of plans used to generate search space for sequential methods')
    parser.add_argument('--single-process',
                        action='store_true',
                        help='set if want to run trajectory collection in the same process')
    parser.add_argument('--initial-dataset-path',
                        type=str,
                        help='path to initial dataset to start with')

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
    parser.add_argument('--val-ratio',
                        type=float,
                        default=.2,
                        help='ratio of aquired data to save to validation dataset')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.restart:
        assert args.exp_path, 'Must set the --exp-path to restart experiment'
        logger = ExperimentLogger(args.exp_path)
        _, n_trained_actions = logger.load_trans_dataset('train', ret_i=True)
        _, n_curr_actions = logger.load_trans_dataset('curr', ret_i=True)
        n_actions = n_trained_actions + n_curr_actions
        args = logger.args
    else:
        assert args.exp_name, 'Must set the --exp-name to start a new experiments'
        assert args.data_collection_mode, 'Must set the --data-collection-mode when starting a new experiment'
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        n_actions = 0
        if args.initial_dataset_path:
            init_logger = ExperimentLogger(args.initial_dataset_path)
            dataset, n_actions = init_logger.load_trans_dataset('', ret_i=True)
            logger.save_trans_dataset(dataset, 'curr', i=n_actions)

    train_class(args, logger, n_actions)
    print('Run saved to %s' % logger.exp_path)
