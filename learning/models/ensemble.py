import torch
import torch.nn as nn


class Ensembles(nn.Module):
    def __init__(self, base_model, base_args, n_models, classes):
        super(Ensembles, self).__init__()
        self.base_model = base_model
        self.base_args = base_args
        self.n_models = n_models
        self.classes = classes
        self.reset()

    def reset(self):
        self.ensembles = {}
        for class_name in self.classes:
            ensemble = Ensemble(self.base_model,
                                self.base_args,
                                self.n_models)
            ensemble.reset()
            self.ensembles[class_name] = ensemble

    def forward(self, x, class_name):
        self.ensembles[class_name].forward()


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
        self.reset()

    def reset(self):
        """ Initialize (or re-initialize) all the models in the ensemble."""
        self.models = nn.ModuleList([self.base_model(**self.base_args) for _ in range(self.n_models)])

    def forward(self, x):
        """ Return a prediction for each model in the ensemble.
        :param x: (N, *) Input tensor compatible with the base_model.
        :return: (N, n_models), class prediction for each model.
        """
        preds = [self.models[ix].forward(x) for ix in range(self.n_models)]
        return torch.cat(preds)


class OptimisticEnsemble(Ensemble):
    """ Just predicts 1 for all inputs """
    def __init__(self, base_model, base_args, n_models):
        super(OptimisticEnsemble, self).__init__(base_model, base_args, n_models)

    def forward(self, x):
        return torch.ones(self.n_models)
