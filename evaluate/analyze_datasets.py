from experiments.utils import ExperimentLogger
import matplotlib.pyplot as plt
import numpy as np

all_dataset_paths = {'0p1': ['logs/experiments/unbalanced_0p1-20220201-201256',
                            'logs/experiments/unbalanced_0p1-20220201-210230',
                            'logs/experiments/unbalanced_0p1-20220201-214929'],
                    '0p4': ['logs/experiments/unbalanced_0p4-20220201-201310',
                            'logs/experiments/unbalanced_0p4-20220201-205943',
                            'logs/experiments/unbalanced_0p4-20220201-214749'],
                    '0p7': ['logs/experiments/unbalanced_0p7-20220201-201316',
                            'logs/experiments/unbalanced_0p7-20220201-210017',
                            'logs/experiments/unbalanced_0p7-20220201-214557'],
                    '1p0': ['logs/experiments/unbalanced_1p0-20220201-201329',
                            'logs/experiments/unbalanced_1p0-20220201-205417',
                            'logs/experiments/unbalanced_1p0-20220201-214002'],}

fig, ax = plt.subplots()
cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for pi, (progress, dataset_paths) in enumerate(all_dataset_paths.items()):
    print('Progress: %s' % progress)
    all_succ = []
    for di, dataset_path in enumerate(dataset_paths):
        print('Dataset path: %s' % dataset_path)
        succ = []
        n_actions = []
        dataset_logger = ExperimentLogger(dataset_path)
        for dataset, dii in dataset_logger.get_dataset_iterator():
            if len(dataset) > 0:
                n_successes = sum([y for x,y in dataset])
                percent_success = n_successes / len(dataset)
                succ.append(percent_success)
                n_actions.append(dii)
        all_succ.append(succ)

    condition = len(set([len(s_list) for s_list in all_succ])) == 1
    assert condition, 'all datasets for progress %s do not contain the same amount of actions/models' % progress
    all_succ = np.array(all_succ)
    avg_s = np.mean(all_succ, axis=0)
    std_s = np.std(all_succ, axis=0)
    ax.plot(n_actions, avg_s, color=cs[pi % len(cs)], label=progress)
    ax.fill_between(n_actions, avg_s-std_s, avg_s+std_s, color=cs[pi % len(cs)], alpha=0.1)

ax.set_xlabel('Number of Executed Actions')
ax.set_ylabel('Percent Success')
ax.set_title('Percent Success over Data Collection Time for varying Goal Progresses')
ax.legend(title='Goal Progress')
ax.set_ylim([0,1])
plt.savefig('percent_success.svg', format='svg')

