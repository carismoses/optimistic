import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.utils import ExperimentLogger
from experiments.strategies import collect_trajectory_wrapper
from domains.tools.world import N_MC_IN, CONTACT_TYPES
from learning.models.ensemble import Ensembles
from learning.models.mlp import MLP
from learning.train import train
from learning.datasets import get_balanced_dataset


def train_class(args, logger, n_actions):
    pddl_model_type = 'learned' if 'learned' in args.data_collection_mode else 'optimistic'

    # get model params
    base_args = {'n_in': N_MC_IN,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}

    # if new exp, save a randomly initialized model
    if n_actions == 0:
        ensembles = Ensembles(MLP, base_args, args.n_models, CONTACT_TYPES)
        logger.save_trans_model(ensembles, i=n_actions)

    while n_actions < args.max_actions:
        dataset = logger.load_trans_dataset('')
        n_dataset_actions = len(dataset)
        print('# actions = %i, |current dataset| = %i' % (n_actions, n_dataset_actions))

        # train at training freq
        if not n_dataset_actions % args.train_freq:
            # if train_bal then balance dataset, save, and train
            if args.train_bal:
                print('Balancing datasets before training')
                bal_dataset = get_balanced_dataset(dataset, CONTACT_TYPES)
                logger.save_bal_dataset(bal_dataset, 'bal', i=n_actions)
                dataset = bal_dataset
            ensembles = Ensembles(MLP, base_args, args.n_models, CONTACT_TYPES)
            for type in CONTACT_TYPES:
                if len(dataset[type]) > 0:
                    print('Training %s ensemble.' % type)
                    dataloader = DataLoader(dataset[type], batch_size=args.batch_size, shuffle=True)
                    for model in ensembles.ensembles[type].models:
                        train(dataloader, model, n_epochs=args.n_epochs, loss_fn=F.binary_cross_entropy)

            # save model and accuracy plots
            logger.save_trans_model(ensembles, i=n_actions)
            print('Saved model to %s' % logger.exp_path)

            '''
            # visualize balanced dataset and model accuracy
            from domains.tools.primitives import get_contact_gen
            from evaluate.plot_value_fns import get_model_accuracy_fn
            from domains.utils import init_world
            import matplotlib.pyplot as plt
            world = init_world('tools',
                                None,
                                'optimistic',
                                False,
                                None)
            contacts_fn = get_contact_gen(world.panda.planning_robot)
            contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)
            mean_fn = get_model_accuracy_fn(ensembles, 'mean')
            std_fn = get_model_accuracy_fn(ensembles, 'std')
            for type in CONTACT_TYPES:
                fig, axes = plt.subplots(3, figsize=(4.5, 8))
                world.vis_dense_plot(type, axes[0], [-1, 1], [-1, 1], 0, 1, value_fn=mean_fn)
                world.vis_dense_plot(type, axes[1], [-1, 1], [-1, 1], None, None, value_fn=std_fn)
                for ai in range(3):
                    world.vis_dataset(axes[ai], bal_dataset.datasets[type], linestyle='-')
                for contact in contacts:
                    cont = contact[0]
                    if cont.type == type:
                        world.vis_tool_ax(cont, axes[2], frame='cont')
                axes[0].set_title('Mean Ensemble Predictions')
                axes[1].set_title('Std Ensemble Predictions')
            plt.show()
            '''

        progress = None
        trajectory = collect_trajectory_wrapper(args,
                                                pddl_model_type,
                                                logger,
                                                progress,
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
    parser.add_argument('--train-bal',
                        action='store_true',
                        help='if want to train only on balanced datasets')

    # Training args
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
        _, n_actions = logger.load_trans_dataset('', ret_i=True)
        #_, n_curr_actions = logger.load_trans_dataset('', ret_i=True)
        #n_actions = n_trained_actions + n_curr_actions
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
