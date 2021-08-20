import argparse
import matplotlib.pyplot as plt
import numpy as np

from learning.active.utils import GoalConditionedExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

### Parameters
# random-actions

dataset_exp_paths = ['learning/experiments/logs/datasets/random-actions-100-20210818-224843',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_1',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_2',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_3',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_4',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_5',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_6',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_7',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_8',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_1',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_2',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_3',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_4',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_5',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_6',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_7',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_8',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224845',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224845_1']

# random-goals-opt
'''
dataset_exp_paths = ['learning/experiments/logs/datasets/random-goals-opt-100-20210818-222521',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-222730',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-222943',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-223211',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-223422',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-223639',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-223852',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-224121',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-224355',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-224654',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-224925',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-225219',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-225516',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-225858',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-230212',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-230525',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-230847',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-231139',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-231415',
                    'learning/experiments/logs/datasets/random-goals-opt-100-20210818-231715']
'''
# random-goals-learned'
'''
dataset_exp_paths = ['learning/experiments/logs/datasets/random-goals-learned-100-20210818-222536',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-223019',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-223632',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-224302',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-224900',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-225521',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-230454',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-231310',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-231830',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-232404',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-233004',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-233550',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210818-235648',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-001847',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-002409',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-004515',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-004958',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-005427',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-010024',
                    'learning/experiments/logs/datasets/random-goals-learned-100-20210819-010509']
'''
###
labels = {0: [], 1: []}
pos_actions = {}
neg_actions = {}
heights = {}

# NOTE: this only works if all data_exp_paths have the same num blocks:
dataset_logger = GoalConditionedExperimentLogger(dataset_exp_paths[0])
num_blocks = dataset_logger.load_args().num_blocks

# init keys for all potential keys
for bb in range(1, num_blocks+1):
    for bt in range(1, num_blocks+1):
        if bt == bb+1:
            pos_actions[str(bb)+','+str(bt)] = []
        else:
            neg_actions[str(bb)+','+str(bt)] = []

for th in range(1,num_blocks+1): heights[th] = []

# store values for each dataset
for dataset_exp_path in dataset_exp_paths:
    dataset_logger = GoalConditionedExperimentLogger(dataset_exp_path)
    max_i=False
    if 'goals' in dataset_exp_path:
        max_i=True
    dataset = dataset_logger.load_trans_dataset(max_i=max_i)
    dataset.set_pred_type('class')

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

# Label Frequency
plt.figure()
plt.bar(labels.keys(), [ms[0] for ms in labels.values()], yerr=[ms[1] for ms in labels.values()])
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.ylim(0, 90)
plt.savefig('1')
plt.close()

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
plt.savefig('2')
plt.close()

# Tower Heights
plt.figure()
plt.bar(heights.keys(), [ms[0] for ms in heights.values()], yerr=[ms[1] for ms in heights.values()])
plt.xlabel('Tower Heights')
plt.ylabel('Frequency')
plt.ylim(0, 100)
plt.savefig('3')
plt.close()
