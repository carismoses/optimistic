import re
import os
import pickle
import time
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace

#from domains.tools.world import ToolsWorld
from learning.models.gnn import TransitionGNN
from learning.models.ensemble import Ensemble, OptimisticEnsemble


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


def add_trajectory_to_dataset(domain, dataset_logger, trajectory, world):
    if len(trajectory) > 0: print('Adding trajectory to dataset.')
    dataset, n_actions = dataset_logger.load_trans_dataset(ret_i=True)
    for (state, pddl_action, next_state, opt_accuracy) in trajectory:
        if (pddl_action.name == 'move_contact' and domain == 'tools') or \
            (pddl_action.name in ['place', 'pickplace'] and domain == 'ordered_blocks'):
            object_features, edge_features = world.state_to_vec(state)
            action_features = world.action_to_vec(pddl_action)
            # assume object features don't change for now
            _, next_edge_features = world.state_to_vec(next_state)
            delta_edge_features = next_edge_features-edge_features
            dataset.add_to_dataset(object_features,
                                    edge_features,
                                    action_features,
                                    next_edge_features,
                                    delta_edge_features,
                                    opt_accuracy)
        n_actions += 1
        dataset_logger.save_trans_dataset(dataset, i=n_actions)
    return n_actions


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

    def add_model_args(self, add_args):
        # add args and save
        del self.args.debug   # args.debug will be in both args sets and cause an error
        self.args = Namespace(**vars(self.args), **vars(add_args))
        os.remove(os.path.join(self.exp_path, 'args.pkl'))
        with open(os.path.join(self.exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(self.args, handle)

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

    def remove_dataset(self, i):
        os.remove(os.path.join(self.exp_path, 'datasets', 'trans_dataset_%i.pkl'%i))

    def load_trans_dataset(self, i=None, balanced=False, ret_i=False):
        # NOTE: returns the highest numbered model if i is not given
        if i is not None:
            fname = 'trans_dataset_%i.pkl' % i
        else:
            data_files = os.listdir(os.path.join(self.exp_path, 'datasets'))
            if len(data_files) == 0:
                raise Exception('No datasets found on args.exp_path.')
            txs = []
            for file in data_files:
                matches = re.match(r'trans_dataset_(.*).pkl', file)
                if matches: # sometimes system files are saved here, don't parse these
                    txs += [int(matches.group(1))]
            i = max(txs)
            fname = 'trans_dataset_%i.pkl' % i
        if balanced:
            fname = 'balanced_dataset.pkl'
        if ret_i:
            return self.load_dataset(fname), i
        return self.load_dataset(fname)

    def get_dataset_iterator(self):
        found_files, txs = self.get_dir_indices('datasets')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_datasets = [(self.load_dataset(fname),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_datasets)

    def get_model_iterator(self):
        found_files, txs = self.get_dir_indices('models')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_models = [(self.load_trans_model(i=i),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_models)

    def get_dir_indices(self, dir):
        files = os.listdir(os.path.join(self.exp_path, dir))
        if len(files) == 0:
            print('No files found on path args.exp_path/%s.' % dir)
        if dir == 'datasets':
            file_name = r'trans_dataset_(.*).pkl'
        elif dir == 'models':
            file_name = r'trans_model_(.*).pt'
        txs = []
        found_files = []
        for file in files:
            matches = re.match(file_name, file)
            if matches: # sometimes system files are saved here, don't parse these
                txs += [int(matches.group(1))]
                found_files += [file]
        return found_files, txs

    # Models
    def save_model(self, model, fname):
        torch.save(model.state_dict(), os.path.join(self.exp_path, 'models', fname))

    def save_trans_model(self, model, i=None):
        if i is not None:
            fname = 'trans_model_%i.pt' % i
        else:
            fname = 'trans_model.pt'
        self.save_model(model, fname)

    def load_trans_model(self, i=None):
        # NOTE: returns the highest numbered model if i is not given
        if i is not None:
            fname = 'trans_model_%i.pt' % i
        else:
            found_files, txs = self.get_dir_indices('models')
            if len(txs) == 0:
                #print('Returning trans_model.pt. No numbered models found on path: %s' % self.exp_path)
                fname = 'trans_model.pt'
            else:
                i = max(txs)
                fname = 'trans_model_%i.pt' % i
                #print('Loading model %s.' % fname)

        #n_of_in, n_ef_in, n_af_in = ToolsWorld.get_model_params()
        base_args = {'n_of_in': 1,#n_of_in,
                    'n_ef_in': 3,#n_ef_in,
                    'n_af_in': 7,#n_af_in,
                    'n_hidden': self.args.n_hidden,
                    'n_layers': self.args.n_layers}
        #if self.args.data_collection_mode == 'curriculum' and i == 0:
        #    model = OptimisticEnsemble(TransitionGNN,
        #                    base_args,
        #                    self.args.n_models)
        #else:
        model = Ensemble(TransitionGNN,
                            base_args,
                            self.args.n_models)
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(os.path.join(self.exp_path, 'models', fname), map_location=loc))
        return model

    # Get action count info from logger
    def get_action_count(self):
        _, txs = self.get_dir_indices('datasets')
        n_actions = max(txs)
        return n_actions

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
