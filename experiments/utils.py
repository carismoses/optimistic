import re
import os
import dill as pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
import torch

from domains.tools.world import MODEL_INPUT_DIMS
from learning.utils import MLP
from learning.models.ensemble import Ensembles


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
            os.mkdir(os.path.join(exp_path, 'datasets', 'train'))
            os.mkdir(os.path.join(exp_path, 'datasets', 'val'))
            os.mkdir(os.path.join(exp_path, 'datasets', 'curr'))
            os.mkdir(os.path.join(exp_path, 'datasets', 'bal'))
            os.mkdir(os.path.join(exp_path, 'models'))
            os.mkdir(os.path.join(exp_path, 'figures'))
            os.mkdir(os.path.join(exp_path, 'eval_trajs'))
            os.mkdir(os.path.join(exp_path, 'trajs'))
            os.mkdir(os.path.join(exp_path, 'plans'))
            os.mkdir(os.path.join(exp_path, 'goals'))

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

    def save_trans_dataset(self, dataset, dir, i):
        fname = os.path.join(dir, 'trans_dataset_%i.pkl' % i)
        self.save_dataset(dataset, fname)

    def save_bal_dataset(self, dataset, dir, i):
        fname = os.path.join(dir, 'bal_dataset_%i.pkl' % i)
        self.save_dataset(dataset, fname)

    def load_dataset(self, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset

    def remove_dataset(self, dir, i):
        os.remove(os.path.join(self.exp_path, 'datasets', dir, 'trans_dataset_%i.pkl' % i))

    def load_trans_dataset(self, dir, i=None, ret_i=False):
        # NOTE: returns the highest numbered model if i is not given
        if i is not None:
            fname = os.path.join(dir, 'trans_dataset_%i.pkl' % i)
            dataset = self.load_dataset(fname)
        else:
            found_files, txs = self.get_dir_indices('datasets/%s' % dir)
            if len(txs) > 0:
                i = max(txs)
                fname = os.path.join(dir, 'trans_dataset_%i.pkl' % i)
                dataset = self.load_dataset(fname)
            else:
                print('No NUMBERED datasets on path %s/datasets/%s. Returning new empty dataset.' % (self.exp_path, dir))
                print('All datasets must be numbered')
                assert False, 'No datasets found on path %s' % os.path.join(self.exp_path, 'datasets')
        if ret_i:
            return dataset, i
        return dataset

    def get_dataset_iterator(self, dir):
        found_files, txs = self.get_dir_indices(os.path.join('datasets', dir))
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_datasets = [(self.load_dataset(os.path.join(dir, fname)), i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_datasets)

    def get_model_iterator(self):
        found_files, txs = self.get_dir_indices('models')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_models = [(self.load_trans_model(i=i),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_models)

    def get_trajectories_iterator(self):
        found_files, txs = self.get_dir_indices('eval_trajs')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_trajs = [(self.load_trajectories(i=i),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_trajs)

    def get_plan_iterator(self):
        found_files, txs = self.get_dir_indices('plans')
        sorted_indices = np.argsort(txs)
        sorted_file_names = [found_files[idx] for idx in sorted_indices]
        sorted_plans = [(self.load_plan(i=i),i) for fname,i in zip(sorted_file_names, np.sort(txs))]
        return iter(sorted_plans)

    def get_dir_indices(self, dir):
        files = os.listdir(os.path.join(self.exp_path, dir))
        if len(files) == 0:
            print('No files found on path %s.' % os.path.join(self.exp_path, dir))
        if 'datasets' in dir:
            file_name = r'trans_dataset_(.*).pkl'
        elif dir == 'models':
            file_name = r'trans_model_(.*).pt'
        elif dir == 'eval_trajs':
            file_name = r'trajs_(.*).pkl'
        elif dir == 'plans':
            file_name = r'plan_(.*).pkl'
        elif dir == 'trajs':
            file_name = r'traj_(.*).pkl'
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

    def load_trans_model(self, i=None, ret_i=False):
        # NOTE: returns the highest numbered model if i is not given
        if i is not None:
            fname = 'trans_model_%i.pt' % i
        else:
            found_files, txs = self.get_dir_indices('models')
            if len(txs) == 0:
                fname = None
                print('Returning randomly initialized model.')
            else:
                i = max(txs)
                fname = 'trans_model_%i.pt' % i
                #print('Loading model %s.' % fname)

        base_args = {'n_in': MODEL_INPUT_DIMS,
                    'n_hidden': self.args.n_hidden,
                    'n_layers': self.args.n_layers}
        assert fname, 'No models found on path' % os.path.join(self.exp_path, 'models')
        model = model = Ensembles(MLP, base_args, self.args.n_models, self.args.objects)
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(os.path.join(self.exp_path, 'models', fname), map_location=loc))
        if ret_i:
            return model, i
        else:
            return model

    # save trajectory data
    def save_trajectories(self, trajectories, i):
        import dill
        path = os.path.join(self.exp_path, 'eval_trajs')
        file_name = 'trajs_%i.pkl' % i
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, file_name), 'wb') as handle:
            dill.dump(trajectories, handle)

    def load_trajectories(self, i):
        path = os.path.join(self.exp_path, 'eval_trajs')
        file_name = 'trajs_%i.pkl' % i
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, file_name), 'rb') as handle:
            trajectories = pickle.load(handle)
        return trajectories


    # add plans
    def add_to_plans(self, plan):
        found_files, txs = self.get_dir_indices('plans')
        if len(txs) == 0:
            i = 0
        else:
            i = max(txs)+1
        path = os.path.join(self.exp_path, 'plans', 'plan_%i.pkl'%i)
        with open(os.path.join(path), 'wb') as handle:
            pickle.dump(plan, handle)


    def load_plan(self, i=None):
        if i is None:
            found_files, txs = self.get_dir_indices('plans')
            assert len(txs) > 0, 'No plans on path %s' % os.path.join(self.exp_path, 'plans')
            i = max(txs)+1 
        path = os.path.join(self.exp_path, 'plans', 'plan_%i.pkl'%i)
        with open(os.path.join(path), 'rb') as handle:
            plans = pickle.load(handle)
        return plans


    # add executed trajectories
    def add_to_trajs(self, traj):
        found_files, txs = self.get_dir_indices('trajs')
        if len(txs) == 0:
            i = 0
        else:
            i = txs[-1]
        path = os.path.join(self.exp_path, 'trajs', 'traj_%i.pkl'%i)
        with open(os.path.join(path), 'wb') as handle:
            pickle.dump(traj, handle)


    # info on abstract plans that fail to plan trajectories
    def add_to_failed_plans(self, new_datapoints):
        datapoints = self.load_failed_plans()
        for datapoint in new_datapoints:
            datapoints.append(datapoint)
        self.save_failed_plans(datapoints)

    def load_failed_plans(self):
        # TODO: change dir and filename so doesn't conflict with goals fns
        dir = 'goals'
        fname = 'failed_goals.pkl'
        ##
        path = os.path.join(self.exp_path, dir)
        if os.path.exists(os.path.join(path, fname)):
            with open(os.path.join(path, fname), 'rb') as handle:
                datapoints = pickle.load(handle)
            return datapoints
        else:
            print('No datapoints found on path %s. Returning empty list' % path)
            return []

    def save_failed_plans(self, datapoints):
        # TODO: change dir and filename so doesn't conflict with goals fns
        dir = 'goals'
        fname = 'failed_goals.pkl'
        ##
        path = os.path.join(self.exp_path, dir)
        with open(os.path.join(path, fname), 'wb') as handle:
            pickle.dump(datapoints, handle)

    # goal info (for goal datasets to calculate plan success rate)
    def save_goals(self, goals):
        path = os.path.join(self.exp_path, 'goals')
        file_name = 'goals.pkl'
        with open(os.path.join(path, file_name), 'wb') as handle:
            pickle.dump(goals, handle)

    def load_goals(self):
        path = os.path.join(self.exp_path, 'goals')
        file_name = 'goals.pkl'
        if os.path.exists(os.path.join(path, file_name)):
            with open(os.path.join(path, file_name), 'rb') as handle:
                goals = pickle.load(handle)
            return goals
        else:
            print('No goals found on path %s' % path)

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
