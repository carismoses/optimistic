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

def subtower_bald(samples, ensemble, data_pred_fn, greedy=False):
    """  subtower_bald is the sum of the bald scores for each subtower

    If "greedy" is True, we just sum the BALD scores. this does not
    consider that a tower might fall over before we collect an
    observation for every block. If "greedy" is False, then we
    multiply the BALD score for each subtower by the predicted probability
    that we collect an observation for that subtower.

    Arguments:
        samples {dict} -- sampled towers of different sizes
        ensemble {Ensemble} -- the model
        data_pred_fn {function} -- uses the model to label towers

    Keyword Arguments:
        greedy {bool} -- don't consider constructability (default: {False})

    Returns:
        torch.Tensor -- scores
    """
    scores = []

    # for each tower size in the sampled towers
    for k in samples.keys():
        n_blocks = int(k.strip('block'))
        data = samples[k]['towers']

        # predict subtower constructability with the ensemble
        subtower_preds = []
        subtower_scores = []
        for i in range(2, n_blocks+1):
            subtowers = {f'{i}block': {'towers': data[:,:i,:],
                                       'labels': np.zeros(data.shape[0])}}
            subtower_preds.append(data_pred_fn(subtowers, ensemble))
            subtower_scores.append(bald(subtower_preds[-1]))

        subtower_preds = torch.stack(subtower_preds)
        subtower_scores = torch.stack(subtower_scores)

        # NOTE(izzy): the version of "greedy" that i've implmented here is the
        # one that fits most readily with our existing infrastructure, but it
        # isn't quite right. what we're doing here is sampling a bunch of
        # towers, adding the bald scores for all the subtowers, and picking
        # the best one. what would really be "greedy" is adding blocks to those
        # towers one at a time.

        if greedy:
            # if we're being greedy, then we're not considering the fact that
            # a tower might fall over before we get to observe the outcome of
            # a taller tower, so we just sum the acquisition scores for each
            # subtower
            scores.append(subtower_scores.sum(axis=0))
        else:
            # to avoid being greedy, we need to consider that building risky
            # towers might prevent us from observing every block in that tower,
            # so we multiply the bald score for each added block by the
            # predicted probability that the tower will reach that height
            two_block_scores = subtower_scores[0]
            constructability = subtower_preds.mean(axis=2)
            n_block_expected_scores = (constructability[:-1] * subtower_scores[1:]).sum(axis=0)
            scores.append(two_block_scores + n_block_expected_scores)

    return torch.cat(scores)

def choose_acquisition_data(samples, ensemble, n_acquire, strategy, data_pred_fn, data_subset_fn):
    """ Choose data points with the highest acquisition score
    :param samples: (N,2) An array of unlabelled datapoints which to evaluate.
    :param ensemble: A list of models.
    :param n_acquire: The number of data points to acquire.
    :param strategy: ['random', 'bald'] The objective to use to choose new datapoints.
    :param data_pred_fn: A handler to get predictions specific on the dataset type.
    :prarm data_subset_fn: A handler to select fewer datapoints.
    :return: (n_acquire, 2) - the samples which to label.
    """
    # Get predictions for each model of the ensemble.
    preds = data_pred_fn(samples, ensemble)
    # print('samples', [samples[k]['towers'].shape for k in samples.keys()])
    # print('preds', preds.shape)

    # Get the acquisition score for each.
    if strategy == 'bald':
        scores = bald(preds).cpu().numpy()
    elif strategy == 'random':
        scores = np.random.uniform(size=preds.shape[0]).astype('float32')
    elif strategy == 'subtower':
        scores = subtower_bald(samples, ensemble, data_pred_fn, greedy=False).cpu().numpy()
    elif strategy == 'subtower-greedy':
        scores = subtower_bald(samples, ensemble, data_pred_fn, greedy=True).cpu().numpy()

    # Return the n_acquire points with the highest score.
    acquire_indices = np.argsort(scores)[::-1][:n_acquire]
    return data_subset_fn(samples, acquire_indices)

def acquire_datapoints(ensemble, n_samples, n_acquire, strategy, data_sampler_fn, \
        data_label_fn, data_pred_fn, data_subset_fn, exec_mode, agent, logger, xy_noise):
    """ Get new datapoints given the current ensemble.
    Uses function handlers for domain specific components (e.g., sampling unlabeled data).
    :param n_samples: How many unlabeled samples to generate.
    :param n_acquire: How many samples to acquire labels for.
    :param strategy: Which acquisition function to use.
    :param data_sampler_fn: Function handler: n_samples -> Dataset
    :param data_label_fn:
    :param data_pred_fn:
    :param data_subset_fn:
    :param exec_mode: in ['simple-model', 'noisy-model', 'sim', 'real']. Method for labeling data
    :param agent: PandaAgent or None (if exec_mode == 'simple-model' or 'noisy-model')
    :return: (n_acquire, 2), (n_acquire,) - x,y tuples of the new datapoints.
    """
    unlabeled_pool = data_sampler_fn(n_samples)
    xs = choose_acquisition_data(unlabeled_pool, ensemble, n_acquire, strategy, data_pred_fn, data_subset_fn)
    new_data = data_label_fn(xs, exec_mode, agent, logger, xy_noise, save_tower=True)
    return new_data, unlabeled_pool
