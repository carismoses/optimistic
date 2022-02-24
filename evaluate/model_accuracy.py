import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from domains.utils import init_world

## Params
all_model_paths = {'sequential-goals-0': ['logs/experiments/sequential-goals-20220224-043346'],
                    'sequential-goals-1': ['logs/experiments/sequential-goals-20220224-043412'],
                    'sequential-goals-2': ['logs/experiments/sequential-goals-20220224-144200'],
                    #'sequential-goals-bal-0': ['logs/experiments/sequential-goals-6init-bal-20220224-143800'],
                    #'sequential-goals-bal-1': ['logs/experiments/sequential-goals-6init-bal-20220224-143923'],
                    #'sequential-goals-bal-2': ['logs/experiments/sequential-goals-6init-bal-20220224-143951'],
                    'random-goals-opt': ['logs/experiments/random-goals-opt-20220223-183356', 
                                        'logs/experiments/random-goals-opt-20220223-183458',
                                        'logs/experiments/random-goals-opt-20220223-183443']}

test_dataset_path = 'logs/experiments/90_random_goals_balanced-20220223-162637'

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset('')
    gts = {}
    for type, dataset in test_dataset.datasets.items():
        gts[type] = [int(y) for _,y in dataset]

    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)
        all_accuracies = []
        for mi, model_path in enumerate(model_paths):
            print('Model path: %s' % model_path)
            accuracies = []
            n_actions = []
            model_logger = ExperimentLogger(model_path)
            for ensembles, mii in model_logger.get_model_iterator():
                model_accuracies = []
                for type, dataset in test_dataset.datasets.items():
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
            print('Shortening all sublists to lengths to action %i' % n_actions[-1])
        all_accuracies = np.array(all_accuracies)
        avg_accs = np.mean(all_accuracies, axis=0)
        std_accs = np.std(all_accuracies, axis=0)
        ax.plot(n_actions, avg_accs, color=cs[pi % len(cs)], label=method)
        ax.fill_between(n_actions, avg_accs-std_accs, avg_accs+std_accs, color=cs[pi % len(cs)], alpha=0.1)

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Model Accuracy over Training Time')
    ax.legend(title='Method')
    ax.set_ylim([0.3,1])
    plt.savefig('model_accuracy.svg', format='svg')
    #plt.show()

