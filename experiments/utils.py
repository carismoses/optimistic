import re
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
import torch

#from domains.tools.world import ToolsWorld
from learning.models.gnn import TransitionGNN
from learning.models.mlp import MLP
from learning.models.ensemble import Ensemble, OptimisticEnsemble


dataset_type_dir_map = {'trans': 'datasets', 'goal': 'goal_datasets', 'balanced': 'datasets'}
model_type_dir_map = {'trans': 'models', 'goal': 'goal_models'}

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
            os.mkdir(os.path.join(exp_path, 'goal_datasets'))
            os.mkdir(os.path.join(exp_path, 'goal_models'))

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

    ## Dataset Functions ##
    # type in ['trans', 'goal', 'balanced']
    def save_dataset(self, dataset, type, i=None):
        if i is not None:
            fname = '%s_dataset_%i.pkl' % (type, i)
        else:
            fname = '%s_dataset.pkl' % type

        dir = dataset_type_dir_map[type]
        with open(os.path.join(self.exp_path, dir, fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def remove_dataset(self, i):
        os.remove(os.path.join(self.exp_path, 'datasets', 'trans_dataset_%i.pkl'%i))

    # type in ['trans', 'goal', 'balanced']
    def load_dataset(self, type, i=None, ret_i=False):
        # NOTE: returns the highest numbered model if i is not given
        dir = dataset_type_dir_map[type]
        if i is not None:
            fname = '%s_dataset_%i.pkl' % (type, i)
        else:
            data_files = os.listdir(os.path.join(self.exp_path, dir))
            if len(data_files) == 0:
                raise Exception('No datasets found on path %s/%s: ' % (args.exp_path, dir))
            txs = []
            for file in data_files:
                matches = re.match(r'%s_dataset_(.*).pkl' % type, file)
                if matches: # sometimes system files are saved here, don't parse these
                    txs += [int(matches.group(1))]
            i = max(txs)
            fname = '%s_dataset_%i.pkl' % (type, i)

        with open(os.path.join(self.exp_path, dir, fname), 'rb') as handle:
            dataset = pickle.load(handle)
        if ret_i:
            return dataset, i
        return dataset

    # type in ['trans', 'goal', 'balanced']
    def get_dataset_iterator(self, type):
        dir = dataset_type_dir_map[type]
        found_files, txs = self.get_dir_indices(type, 'dataset')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_datasets = [(self.load_dataset(type, i=i),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_datasets)

    # type in ['trans', 'goal', 'balanced']
    def get_model_iterator(self, type):
        dir = model_type_dir_map[type]
        found_files, txs = self.get_dir_indices(type, 'model')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_models = [(self.load_model(type, i=i),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_models)

    # type in ['trans', 'goal', 'balanced']
    # dir_type in ['dataset', 'model']
    def get_dir_indices(self, type, dir_type):
        if dir_type == 'dataset': dir, ending = dataset_type_dir_map[type], 'pkl'
        if dir_type == 'model': dir, ending = model_type_dir_map[type], 'pt'
        files = os.listdir(os.path.join(self.exp_path, dir))
        if len(files) == 0:
            print('No files found on path %s/%s.' % (self.exp_path, dir))
        file_name = r'%s_%s_(.*).%s' % (type, dir_type, ending)
        txs = []
        found_files = []
        for file in files:
            matches = re.match(file_name, file)
            if matches: # sometimes system files are saved here, don't parse these
                txs += [int(matches.group(1))]
                found_files += [file]
        return found_files, txs

    ## Model Functions ##
    def save_model(self, model, type, i=None):
        if i is not None:
            fname = '%s_model_%i.pt' % (type, i)
        else:
            fname = '%s_model.pt' % type
        dir = model_type_dir_map[type]
        torch.save(model.state_dict(), os.path.join(self.exp_path, dir, fname))

    def load_model(self, type, i=None):
        # NOTE: returns the highest numbered model if i is not given
        dir = model_type_dir_map[type]
        if i is not None:
            fname = '%s_model_%i.pt' % (type, i)
        else:
            found_files, txs = self.get_dir_indices(type, 'model')
            if len(txs) == 0:
                #print('Returning trans_model.pt. No numbered models found on path: %s' % self.exp_path)
                fname = '%s_model.pt' % type
            else:
                i = max(txs)
                fname = '%s_model_%i.pt' % (type, i)
                #print('Loading model %s.' % fname)

        if type == 'trans':
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
        if type == 'goal':
            goal_args = {'n_in': 3,
                        'n_hidden': self.args.n_hidden,
                        'n_layers': self.args.n_layers}
            model = Ensemble(MLP,
                            goal_args,
                            self.args.n_models)
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(os.path.join(self.exp_path, dir, fname), map_location=loc))
        return model

    # Get action count info from logger
    def get_action_count(self):
        _, txs = self.get_dir_indices('trans', 'dataset')
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
