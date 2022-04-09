from copy import copy
import torch
import torch.nn as nn


class Ensembles(nn.Module):
    def __init__(self, base_model, base_args, n_models, objects, wi):
        # actions_and_objects is a list of (action, object) tuples
        # where action in {pick, push-poke, push-push_pull}
        # object in {yellow_block, blue_block}
        super(Ensembles, self).__init__()
        self.base_model = base_model
        self.all_n_in = copy(base_args['n_in'])
        self.base_args = copy(base_args)
        self.n_models = n_models
        self.actions = ['pick', 'move_contact-poke', 'move_contact-push_pull', 'move_holding']
        self.objects = objects
        self.weight_init_sds = {'move_contact-push_pull': wi}#('move_contact-poke', 'blue_block'): 9,
                            #('move_holding', 'yellow_block'): 5}
        self.reset()


    def reset(self):
        def reset_dict(action, obj, grasp):
            if action not in self.ensembles:
                self.ensembles[action] = nn.ModuleDict()
            if obj not in self.ensembles[action]:
                self.ensembles[action][obj] = nn.ModuleDict()

            self.base_args['n_in'] = self.all_n_in[action]
            if (action, obj) in self.weight_init_sds:
                self.base_args['winit_sd'] = self.weight_init_sds[(action, obj)]
            else:
                self.base_args['winit_sd'] = 7
            ensemble = Ensemble(self.base_model,
                                self.base_args,
                                self.n_models)
            ensemble.reset()
            self.ensembles[action][obj][grasp] = ensemble

        self.ensembles = nn.ModuleDict()
        for action in self.actions:
            for obj in self.objects:
                if 'move_contact' in action:
                    for grasp in ['p1', 'n1']:
                        reset_dict(action, obj, grasp)
                else:
                    grasp = 'None'
                    reset_dict(action, obj, grasp)


    def forward(self, x, action, obj):
        self.ensembles[action][obj].forward()


class Ensemble(nn.Module):
    """ A helper class to represent a collection of models.

    Intended usage:
    This class is designed to be used for active learning. It only needs
    to be initialized once outside of the active loop. Every time we want
    new models, we only need to call reset().

    To save an ensemble, save it as a regular PyTorch model. Only a single
    file is needed for the whole ensemble.

    When loading, an ensemble with the same base parameters will need to
    be used. After that the forward function can be used to get predictions
    from all models.
    """
    def __init__(self, base_model, base_args, n_models):
        """ Save relevant information to reinitialize the ensemble later on.
        :param base_model: The class of a single ensemble member.
        :param base_args: The arguments used to initialize the base member.
        :param n_models: The number of models in the ensemble.
        """
        super(Ensemble, self).__init__()
        self.base_model = base_model
        self.base_args = base_args
        self.n_models = n_models

    def reset(self):
        """ Initialize (or re-initialize) all the models in the ensemble."""
        self.models = nn.ModuleList([self.base_model(**self.base_args) for _ in range(self.n_models)])
        for model in self.models:
            model.reset()

    def forward(self, x):
        """ Return a prediction for each model in the ensemble.
        :param x: (N, *) Input tensor compatible with the base_model.
        :return: (N, n_models), class prediction for each model.
        """
        preds = [self.models[ix].forward(x).unsqueeze(0) for ix in range(self.n_models)]
        return torch.cat(preds, axis=0)
