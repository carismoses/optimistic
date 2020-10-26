import numpy as np
import torch


def bald(predictions, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    mp_c1 = torch.mean(predictions, dim=1)
    mp_c0 = torch.mean(1 - predictions, dim=1)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=1)

    bald = m_ent + ent
    return bald

def choose_acquisition_data(samples, ensemble, n_acquire, strategy, data_pred_fn):
    """ Choose data points with the highest acquisition score
    :param samples: (N,2) An array of unlabelled datapoints which to evaluate.
    :param ensemble: A list of models. 
    :param n_acquire: The number of data points to acquire.
    :param strategy: ['random', 'bald'] The objective to use to choose new datapoints.
    :param data_pred_fn: A handler to get predictions specific on the dataset type.
    :return: (n_acquire, 2) - the samples which to label.
    """
    # Get predictions for each model of the ensemble. 
    preds = data_pred_fn(samples, ensemble)

    # Get the acquisition score for each.
    if strategy == 'bald':
        scores = bald(preds).cpu().numpy()
    elif strategy == 'random':
        scores = np.ones((preds.shape[0],), dtype='float32')
        
    # Return the n_acquire points with the highest score.
    acquire_indices = np.argsort(scores)[::-1][:n_acquire]
    return samples[acquire_indices, :]

def acquire_datapoints(ensemble, n_samples, n_acquire, strategy, data_sampler_fn, data_label_fn, data_pred_fn):
    """ Get new datapoints given the current ensemble.
    Uses function handlers for domain specific components (e.g., sampling unlabeled data).
    :n_samples: How many unlabeled samples to generate.
    :n_acquire: How many samples to acquire labels for.
    :strategy: Which acquisition function to use.
    :data_sampler_fn: Function handler: n_samples -> Dataset
    :data_label_fn:
    :data_pred_fn:
    :return: (n_acquire, 2), (n_acquire,) - x,y tuples of the new datapoints.
    """
    unlabeled_pool = data_sampler_fn(n_samples)
    xs = choose_acquisition_data(unlabeled_pool, ensemble, n_acquire, strategy, data_pred_fn)
    ys = data_label_fn(xs)
    return xs, ys, unlabeled_pool
