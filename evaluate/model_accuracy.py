import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld

## Params
# poke paths
all_model_paths = {'sequential-goals': ['sequential-goals-20220228-231704',
                                        'sequential-goals-20220228-231714',
                                        'sequential-goals-20220302-233003'],
                    'sequential-goals-early-stop': ['sequential-goals-20220302-011936',
                                                    'sequential-goals-20220302-011949',
                                                    'sequential-goals-20220302-232923']}

# push paths
'''
all_model_paths = {'sequential-goals': ['sequential-goals-20220302-233044',
                                        'sequential-goals-20220302-233108',
                                        'sequential-goals-20220302-233121'],
                    'sequential-goals-early-stop': ['sequential-goals-20220302-233227',
                                                    'sequential-goals-20220302-233239',
                                                    'sequential-goals-20220302-233248']}
'''
test_dataset_path = 'logs/experiments/60_random_goals_balanced-20220226-041103'
max_actions = None

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset('')
    gts = {}
    for type, dataset in test_dataset.datasets.items():
        gts[type] = [int(y) for _,y in dataset]
    type = 'poke'
    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)
        all_accuracies = []
        for mi, model_path in enumerate(model_paths):
            print('Model path: %s' % model_path)
            accuracies = []
            n_actions = []
            model_logger = ExperimentLogger('logs/experiments/'+model_path)
            for ensembles, mii in model_logger.get_model_iterator():
                model_accuracies = []
                #for type, dataset in test_dataset.datasets.items():
                dataset = test_dataset.datasets[type]
                model_preds = [model_forward(type, ensembles, x, single_batch=True).squeeze().mean().round() \
                                    for x,_ in dataset]
                model_accuracies.append([(pred == gt) for pred, gt in zip(model_preds, gts[type])])
                accuracies.append(np.mean(model_accuracies))
                n_actions.append(mii)
            all_accuracies.append(accuracies)

        list_lens = [len(acc_list) for acc_list in all_accuracies]
        condition = not len(set(list_lens)) == 1
        if condition:
            print('all models for method %s do not contain the same amount of actions/models' % method)
            last_i = min(list_lens)
            all_accuracies = [all_accs[:last_i] for all_accs in all_accuracies]
            n_actions = n_actions[:last_i]
            print('Shortening all sublists lengths to action %i' % n_actions[-1])
        if max_actions:
            print('Shortening all sublists lengths to action %i' % max_actions)
            last_i = n_actions.index(max_actions)
            all_accuracies = [all_accs[:last_i] for all_accs in all_accuracies]
            n_actions = n_actions[:last_i]
        all_accuracies = np.array(all_accuracies)
        avg_accs = np.mean(all_accuracies, axis=0)
        std_accs = np.std(all_accuracies, axis=0)
        ax.plot(n_actions, avg_accs, color=cs[pi % len(cs)], label=method)
        ax.fill_between(n_actions, avg_accs-std_accs, avg_accs+std_accs, color=cs[pi % len(cs)], alpha=0.1)

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Model Accuracy over Training Time')
    ax.legend(title='Method', loc='lower right')
    ax.set_ylim([0.3,1])
    plt.savefig('model_accuracy.svg', format='svg')
    #plt.show()
