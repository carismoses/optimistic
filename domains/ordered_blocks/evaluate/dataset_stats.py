import argparse
import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import ExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

### Parameters
# This plots a histogram of the labels (feasible/infeasible), tower heights, and
# actions attempted in a set of models all trained using the same method (set below
# with dataset_exp_paths from results_paths.py)
from domains.ordered_blocks.results_paths import model_paths
dataset_exp_paths = model_paths['random-goals-learned']
###

labels = {0: [], 1: []}
pos_actions = {}
neg_actions = {}
heights = {}

# NOTE: this only works if all data_exp_paths have the same num blocks:
dataset_logger = ExperimentLogger(dataset_exp_paths[0])
num_blocks = int(dataset_logger.load_args().domain_args[0])

# init keys for all potential keys
for bb in range(1, num_blocks+1):
    for bt in range(1, num_blocks+1):
        if bt == bb+1:
            pos_actions[str(bt)+','+str(bb)] = []
        else:
            neg_actions[str(bt)+','+str(bb)] = []

for th in range(1,num_blocks+1): heights[th] = []

# store values for each dataset
for dataset_exp_path in dataset_exp_paths:
    dataset_logger = ExperimentLogger(dataset_exp_path)
    dataset = dataset_logger.load_dataset('trans')

    ds_pos_actions, ds_neg_actions, ds_labels, ds_heights = [], [], [], []
    for x,y in dataset:
        label = int(y.detach().numpy())
        ds_labels.append(label)
        str_action = ','.join([str(int(a.detach())) for a in x[2]])
        if label == 1: ds_pos_actions.append(str_action)
        if label == 0: ds_neg_actions.append(str_action)
        vef = x[1].detach().numpy()
        ds_heights.append(vef[1:,1:].sum()+1)  # one block is stacked on table

    for d, ds_list in zip([labels, pos_actions, neg_actions, heights], [ds_labels, ds_pos_actions, ds_neg_actions, ds_heights]):
        for key in d:
            d[key].append(ds_list.count(key))

# calc mean and std dev over all datasets
for d in [labels, pos_actions, neg_actions, heights]:
    for key in d:
        mean = np.mean(d[key])
        std = np.std(d[key])
        d[key] = [mean, std]

plt.ion()

# Label Frequency
plt.figure()
plt.bar(labels.keys(), [ms[0] for ms in labels.values()], yerr=[ms[1] for ms in labels.values()])
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.ylim(0, 90)
#plt.savefig('1')
#plt.close()

# Transition Frequency
plt.figure()
for actions, label in zip([pos_actions, neg_actions], ['Positive', 'Negative']):
    plt.bar(list(actions.keys()),
            [ms[0] for ms in actions.values()],
            label=label,
            color='r' if label=='Negative' else 'g',
            yerr=[ms[1] for ms in actions.values()])
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.ylim(0, 37)
plt.legend()
#plt.savefig('2')
#plt.close()

# Tower Heights
plt.figure()
plt.bar(heights.keys(), [ms[0] for ms in heights.values()], yerr=[ms[1] for ms in heights.values()])
plt.xlabel('Tower Heights')
plt.ylabel('Frequency')
plt.ylim(0, 100)
#plt.savefig('3')
#plt.close()

plt.show()
input('enter to close')
plt.close()
