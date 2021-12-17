import re
import os
import pickle
import time
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np

from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble


def model_forward(model, inputs, single_batch=False):
    if single_batch:
        single_inputs = inputs
        inputs = [torch.tensor(input[None, :], dtype=torch.float64) \
                                            for input in single_inputs]
    if torch.cuda.is_available():
        model.cuda()
        inputs = [inpi.cuda() for inpi in inputs]

    output = model.forward(inputs)
    if torch.cuda.is_available():
        output = output.cpu()
    return output.detach().numpy()


class ExperimentLogger:

    def __init__(self, exp_path):
        self.exp_path = exp_path

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

    @staticmethod
    def setup_experiment_directory(args, root_folder):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = os.path.join('logs', root_folder)
        if not os.path.exists(root): os.makedirs(root)

        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        # add suffix if the exp_path already exists
        if os.path.exists(exp_path):
            suff = 1
            while os.path.exists(exp_path+'_'+str(suff)):
                suff += 1
            exp_path = exp_path+'_'+str(suff)

        os.mkdir(exp_path)
        if root_folder == 'experiments':
            os.mkdir(os.path.join(exp_path, 'datasets'))
            os.mkdir(os.path.join(exp_path, 'models'))
            os.mkdir(os.path.join(exp_path, 'figures'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ExperimentLogger(exp_path)

    # Datasets
    def save_dataset(self, dataset, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def save_trans_dataset(self, dataset, i=None):
        if i is not None:
            fname = 'trans_dataset_%i.pkl' % i
        else:
            fname = 'trans_dataset.pkl'
        self.save_dataset(dataset, fname)

    def save_balanced_dataset(self, dataset):
        self.save_dataset(dataset, 'balanced_dataset.pkl')

    def load_dataset(self, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset

    def load_trans_dataset(self, i=None, balanced=False):
        # NOTE: returns the highest numbered model if i is not given
        if i is not None:
            fname = 'trans_dataset_%i.pkl' % i
        else:
            model_files = os.listdir(os.path.join(self.exp_path, 'datasets'))
            if len(model_files) == 0:
                raise Exception('No datasets found on args.exp_path.')
            txs = []
            for file in model_files:
                matches = re.match(r'trans_dataset_(.*).pkl', file)
                if matches: # sometimes system files are saved here, don't parse these
                    txs += [int(matches.group(1))]
            i = max(txs)
            fname = 'trans_dataset_%i.pkl' % i
        if balanced:
            fname = 'balanced_dataset.pkl'
        return self.load_dataset(fname)

    def get_dataset_iterator(self):
        model_files = os.listdir(os.path.join(self.exp_path, 'datasets'))
        if len(model_files) == 0:
            raise Exception('No datasets found on args.exp_path.')
        txs = []
        datasets = []
        for file in model_files:
            matches = re.match(r'trans_dataset_(.*).pkl', file)
            if matches: # sometimes system files are saved here, don't parse these
                txs += [int(matches.group(1))]
                datasets += [file]
        sorted_indices = np.argsort(txs)
        sorted_dataset_names = [datasets[idx] for idx in sorted_indices]
        sorted_datasets = [(self.load_dataset(fname),i) for fname,i in zip(sorted_dataset_names, np.sort(txs))]
        return iter(sorted_datasets)

    # Models
    def save_model(self, model, fname):
        torch.save(model.state_dict(), os.path.join(self.exp_path, 'models', fname))

    def save_trans_model(self, model, i=None):
        if i is not None:
            fname = 'trans_model_%i.pt' % i
        else:
            fname = 'trans_model.pt'
        self.save_model(model, fname)

    def load_trans_model(self, world, i=None):
        # NOTE: returns the highest numbered model if i is not given
        if i is not None:
            fname = 'trans_model_%i.pt' % i
        else:
            model_files = os.listdir(os.path.join(self.exp_path, 'models'))
            if len(model_files) == 0:
                raise Exception('No models found on args.exp_path.')
            txs = []
            for file in model_files:
                matches = re.match(r'trans_model_(.*).pt', file)
                if matches: # sometimes system files are saved here, don't parse these
                    txs += [int(matches.group(1))]
            if len(txs) == 0:
                #print('Returning trans_model.pt. No numbered models found on path: %s' % self.exp_path)
                fname = 'trans_model.pt'
            else:
                i = max(txs)
                fname = 'trans_model_%i.pt' % i
                #print('Loading model %s.' % fname)

        base_args = {'n_of_in': world.n_of_in,
                    'n_ef_in': world.n_ef_in,
                    'n_af_in': world.n_af_in,
                    'n_hidden': self.args.n_hidden,
                    'n_layers': self.args.n_layers}
        model = Ensemble(TransitionGNN,
                        base_args,
                        self.args.n_models)
        model.load_state_dict(torch.load(os.path.join(self.exp_path, 'models', fname)))
        return model

    # Planning info
    def save_planning_data(self, tree, goal, plan, i=None):
        with open(os.path.join(self.exp_path, 'plans', 'plan_data_%i.pkl' % i), 'wb') as handle:
            pickle.dump([tree, goal, plan], handle)

    # args
    def load_args(self):
        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            args = pickle.load(handle)
        return args

    # plot data
    def save_plot_data(self, plot_data):
        with open(os.path.join(self.exp_path, 'plot_data.pkl'), 'wb') as handle:
            pickle.dump(plot_data, handle)

    def load_plot_data(self):
        with open(os.path.join(self.exp_path, 'plot_data.pkl'), 'rb') as handle:
            plot_data = pickle.load(handle)
        return plot_data

    # figures
    def save_figure(self, filename, dir=''):
        full_path = os.path.join(self.exp_path, 'figures', dir)
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        plt.savefig(os.path.join(full_path, filename))
