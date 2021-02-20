import re
import os
import pickle
import time
import torch
import datetime

from torch.utils.data import DataLoader

from learning.models.ensemble import Ensemble
from learning.models.mlp_dropout import MLP


class ExperimentLogger:

    def __init__(self, exp_path):
        self.exp_path = exp_path

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/active/experiments'
        if not os.path.exists(root): os.makedirs(root)

        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ExperimentLogger(exp_path)

    def save_dataset(self, dataset, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def load_dataset(self, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset

    def get_figure_path(self, fname):
        return os.path.join(self.exp_path, 'figures', fname)

    def save_model(self, model, fname):
        torch.save(model.state_dict(), os.path.join(self.exp_path, 'models', fname))

    def load_model(self, fname):
        model = MLP(n_hidden=self.args.n_hidden, dropout=self.args.dropout)
        model.load_state_dict(torch.load(os.path.join(self.exp_path, 'models', fname)))
        return model

    def get_ensemble(self):
        ensemble = []
        for mx in range(0, self.args.n_models):
            ensemble.append(self.load_model('net_%d.pt' % mx))
        return ensemble

class ActiveExperimentLogger:

    def __init__(self, exp_path):
        self.exp_path = exp_path
        self.acquisition_step = 0
        self.tower_counter = 0

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

    @staticmethod
    def get_experiments_logger(exp_path, args):
        logger = ActiveExperimentLogger(exp_path)
        dataset_path = os.path.join(exp_path, 'datasets')
        dataset_files = os.listdir(dataset_path)
        if len(dataset_files) == 0:
            raise Exception('No datasets found on args.exp_path. Cannot restart active training.')
        txs = []
        for file in dataset_files:
            matches = re.match(r'active_(.*).pkl', file)
            if matches: # sometimes system files are saved here, don't parse these
                txs += [int(matches.group(1))]
        logger.acquisition_step = max(txs)
        
        # save potentially new args
        with open(os.path.join(exp_path, 'args_restart.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return logger

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/experiments/logs'
        if not os.path.exists(root): os.makedirs(root)

        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'towers'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))
        os.mkdir(os.path.join(exp_path, 'val_datasets'))
        os.mkdir(os.path.join(exp_path, 'acquisition_data'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ActiveExperimentLogger(exp_path)

    def save_dataset(self, dataset, tx):
        fname = 'active_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)
            
    def save_val_dataset(self, val_dataset, tx):
        fname = 'val_active_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'val_datasets', fname), 'wb') as handle:
            pickle.dump(val_dataset, handle)

    def load_dataset(self, tx):
        fname = 'active_%d.pkl' % tx 
        path = os.path.join(self.exp_path, 'datasets', fname)
        try:
            with open(path, 'rb') as handle:
                dataset = pickle.load(handle)
            return dataset
        except:
            print('active_%d.pkl not found on path' % tx)
            return None
            
    def load_val_dataset(self, tx):
        fname = 'val_active_%d.pkl' % tx 
        path = os.path.join(self.exp_path, 'val_datasets', fname)
        try:
            with open(path, 'rb') as handle:
                val_dataset = pickle.load(handle)
            return val_dataset
        except:
            print('val_active_%d.pkl not found on path' % tx)
            return None

    def get_figure_path(self, fname):
        if not os.path.exists(os.path.join(self.exp_path, 'figures')): 
            os.mkdir(os.path.join(self.exp_path, 'figures'))
        return os.path.join(self.exp_path, 'figures', fname)

    def get_towers_path(self, fname):
        return os.path.join(self.exp_path, 'towers', fname)

    def get_ensemble(self, tx):
        """ Load an ensemble from the logging structure.
        :param tx: The active learning iteration of which ensemble to load.
        :return: learning.models.Ensemble object.
        """
        # Load metadata and initialize ensemble.
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'rb') as handle:
            metadata = pickle.load(handle)
        ensemble = Ensemble(base_model=metadata['base_model'],
                            base_args=metadata['base_args'],
                            n_models=metadata['n_models'])

        # Load ensemble weights.
        path = os.path.join(self.exp_path, 'models', 'ensemble_%d.pt' % tx)
        try:
            ensemble.load_state_dict(torch.load(path, map_location='cpu'))
            return ensemble
        except:
            print('ensemble_%d.pkl not found on path' % tx)
            return None

    def save_ensemble(self, ensemble, tx):
        """ Save an ensemble within the logging directory. The weights
        will be saved to <exp_name>/models/ensemble_<tx>.pt. Model metadata that
        is needed to initialize the Ensemble class while loading is 
        save to <exp_name>/models/metadata.pkl.

        :ensemble: A learning.model.Ensemble object.
        :tx: The active learning timestep these models represent.
        """
        # Save ensemble metadata.
        metadata = {'base_model': ensemble.base_model,
                    'base_args': ensemble.base_args,
                    'n_models': ensemble.n_models}
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(metadata, handle)

        # Save ensemble weights.
        path = os.path.join(self.exp_path, 'models', 'ensemble_%d.pt' % tx)
        torch.save(ensemble.state_dict(), os.path.join(path))

    def get_towers_data(self, tx):
        tower_data = []
        for tower_file in os.listdir(os.path.join(self.exp_path, 'towers')):
            matches = re.match(r'labeled_tower_(.*)_(.*)_(.*)_(.*).pkl', tower_file)
            if matches: # sometimes other system files get saved here (eg. .DStore on a mac). don't parse these
                tower_tx = int(matches.group(4))
                if tower_tx == tx:
                    with open(self.get_towers_path(tower_file), 'rb') as handle:
                        tower_tx_data = pickle.load(handle)
                    tower_data += [tower_tx_data]
        return tower_data

    def save_towers_data(self, block_tower, block_ids, label):
        fname = 'labeled_tower_{:%Y-%m-%d_%H-%M-%S}_'.format(datetime.datetime.now())\
                +str(self.tower_counter)+'_'+str(self.acquisition_step)+'.pkl'
        path = self.get_towers_path(fname)
        with open(path, 'wb') as handle:
            pickle.dump([block_tower, block_ids, label], handle)
        self.tower_counter += 1

    def save_acquisition_data(self, new_data, samples, tx):
        data = {
            'acquired_data': new_data,
            'samples': samples
        }
        path = os.path.join(self.exp_path, 'acquisition_data', 'acquired_%d.pkl' % tx)
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
        self.acquisition_step = tx+1
        self.tower_counter = 0
        self.remove_unlabeled_acquisition_data()
        
    def remove_unlabeled_acquisition_data(self):
        os.remove(os.path.join(self.exp_path, 'acquired_processing.pkl'))
        
    def save_unlabeled_acquisition_data(self, data):
        path = os.path.join(self.exp_path, 'acquired_processing.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
            
    def get_unlabeled_acquisition_data(self):
        path = os.path.join(self.exp_path, 'acquired_processing.pkl')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data
    
    def load_acquisition_data(self, tx):
        path = os.path.join(self.exp_path, 'acquisition_data', 'acquired_%d.pkl' % tx)
        try:
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
            return data['acquired_data'], data['samples']
        except:
            print('acquired_%d.pkl not found on path' % tx)
            return None, None