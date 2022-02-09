from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import Namespace

from learning.datasets import TransDataset
from experiments.utils import ExperimentLogger
from learning.models.gnn import TransitionGNN
from learning.train import train
from domains.tools.world import ToolsWorld


batch_size = 16
n_epochs = 300
n_hidden = 32
n_layers = 5
n_of_in = 1
n_af_in = 7
n_ef_in = 3

if __name__ == '__main__':
    dataset_paths = ['logs/experiments/yellow100opt-20211129-151544',
                        'logs/experiments/yellow100opt-20211129-155524',
                        'logs/experiments/yellow100opt-20211129-162804',
                        'logs/experiments/yellow100opt-20211129-170739',
                        'logs/experiments/yellow100opt-20211129-174541']
    dataset_index = 100
    this_i = 500

    # load datasets
    full_dataset = TransDataset()
    for path in dataset_paths:
        test_dataset_logger = ExperimentLogger(path)
        test_dataset = test_dataset_logger.load_dataset('trans', i=dataset_index)
        for si in range(len(test_dataset)):
            sample = test_dataset.__getitem__(si, full_info=True)
            full_dataset.add_to_dataset(*sample)

    # set up logger
    args = Namespace(exp_name='test')
    logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')

    # initialize model
    trans_model = TransitionGNN(n_of_in=n_of_in,
                                n_ef_in=n_ef_in,
                                n_af_in=n_af_in,
                                n_hidden=n_hidden,
                                n_layers=n_layers)

    # train model and save model and dataset
    print('Training model.')
    trans_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    train(trans_dataloader, trans_model, n_epochs=n_epochs, loss_fn=F.binary_cross_entropy)
    logger.save_dataset(full_dataset, 'trans', i=this_i)
    logger.save_model(trans_model, 'trans', i=this_i)

    # plot accuracy
    world, opt_pddl_info, pddl_info = ToolsWorld.init([],
                                                    'learned',
                                                    False,
                                                    logger)
    world.plot_model_accuracy(this_i, trans_model, logger)
