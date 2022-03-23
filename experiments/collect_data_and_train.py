import argparse


from experiments.utils import ExperimentLogger
from experiments.strategies import collect_trajectory_wrapper
from experiments.skeletons import merge_skeletons
from domains.tools.world import MODEL_INPUT_DIMS
from learning.utils import MLP, train_model
from learning.models.ensemble import Ensembles
from learning.datasets import OptDictDataset


def train_class(args, logger, n_actions):
    pddl_model_type = 'learned' if 'learned' in args.data_collection_mode else 'optimistic'

    # get model params
    base_args = {'n_in': MODEL_INPUT_DIMS,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    # if new exp, save a randomly initialized model
    if n_actions == 0:
        dataset = OptDictDataset(args.objects)
        logger.save_trans_dataset(dataset, '', i=n_actions)
        model = Ensembles(MLP, base_args, args.n_models, args.objects)
        logger.save_trans_model(model, i=n_actions)

    # merge skeleton files
    if args.samples_from_file:
        merge_skeletons(args.skel_nums)

    while n_actions < args.max_actions:
        dataset = logger.load_trans_dataset('')
        n_dataset_actions = len(dataset)
        print('# actions = %i, |current dataset| = %i' % (n_actions, n_dataset_actions))

        # train at training freq
        if not n_dataset_actions % args.train_freq:
            model = Ensembles(MLP, base_args, args.n_models, args.objects)
            train_model(model, dataset, args)

            # save model and accuracy plots
            logger.save_trans_model(model, i=n_actions)
            print('Saved model to %s' % logger.exp_path)

        trajectory = collect_trajectory_wrapper(args,
                                                pddl_model_type,
                                                logger,
                                                separate_process = not args.single_process)
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

    # World args
    parser.add_argument('--domain',
                        type=str,
                        choices=['ordered_blocks', 'tools'],
                        default='tools',
                        help='domain to generate data from')
    parser.add_argument('--vis',
                        action='store_true',
                        help='use to visualize robot executions.')
    parser.add_argument('--objects',
                        nargs='+',
                        type=str,
                        choices=['yellow_block', 'blue_block'],
                        default=['yellow_block', 'blue_block'])

    # Data collection args
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--max-actions',
                        type=int,
                        default=400,
                        help='max number of (ALL) actions for the robot to attempt')
    parser.add_argument('--data-collection-mode',
                        type=str,
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned', \
                                'sequential-plans', 'sequential-goals'],
                        help='method of data collection')
    parser.add_argument('--samples-from-file',
                        action='store_true',
                        help='set if want to use pre-generated samples for BALD search')
    parser.add_argument('--skel-nums',
                        type=int,
                        nargs='+')
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
    parser.add_argument('--train-freq',
                        type=int,
                        default=1,
                        help='number of actions (IN DATASET) between model training')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='training batch size')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=300,
                        help='training epochs')
    parser.add_argument('--n-hidden',
                        type=int,
                        default=48,
                        help='number of hidden units in network')
    parser.add_argument('--n-layers',
                        type=int,
                        default=5,
                        help='number of layers in GNN node and edge networks')
    parser.add_argument('--n-models',
                        type=int,
                        default=5,
                        help='number of models in ensemble')
    parser.add_argument('--early-stop',
                        type=str,
                        default='False',
                        choices=['True', 'False'],
                        help='stop training models when training loss below a threshold')

    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    args.early_stop = args.early_stop == 'True'

    if args.restart:
        assert args.exp_path, 'Must set the --exp-path to restart experiment'
        logger = ExperimentLogger(args.exp_path)
        _, n_actions = logger.load_trans_dataset('', ret_i=True)
        args = logger.args
    else:
        assert args.exp_name, 'Must set the --exp-name to start a new experiments'
        assert args.data_collection_mode, 'Must set the --data-collection-mode when starting a new experiment'
        logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
        n_actions = 0
        if args.initial_dataset_path:
            init_logger = ExperimentLogger(args.initial_dataset_path)
            dataset, n_actions = init_logger.load_trans_dataset('', ret_i=True)
            logger.save_trans_dataset(dataset, '', i=n_actions)

    train_class(args, logger, n_actions)
    print('Run saved to %s' % logger.exp_path)
