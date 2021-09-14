from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def recc_dict():
    return defaultdict(recc_dict)

def plot_results(success_data, all_test_num_blocks, title, xlabel, ylabel, logger):
    # plot colors
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # plot all results
    figure, axis = plt.subplots()
    for i, (method, method_successes) in enumerate(success_data.items()):
        method_avgs = []
        method_mins = []
        method_maxs = []
        for test_num_blocks, num_block_successes in method_successes.items():
            if method == 'opt':
                num_block_success_data = num_block_successes
            else:
                num_block_success_data = [data for model_path, data in num_block_successes.items()]
            method_avgs.append(np.mean(num_block_success_data))
            method_mins.append(np.mean(num_block_success_data)-np.std(num_block_success_data))
            method_maxs.append(np.mean(num_block_success_data)+np.std(num_block_success_data))
        if method == 'opt':
            axis.plot(all_test_num_blocks, method_avgs, color=cs[i], label=method, linestyle='-.')
        else:
            axis.plot(all_test_num_blocks, method_avgs, color=cs[i], label=method)
        axis.fill_between(all_test_num_blocks, method_mins, method_maxs, color=cs[i], alpha=0.1)

    axis.set_xticks(all_test_num_blocks)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1.1)
    axis.legend(title='Method')

    plt.savefig('%s/%s.png' % (logger.exp_path, title))
    print('Saving figures to %s.' % logger.exp_path)
