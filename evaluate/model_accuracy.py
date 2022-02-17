import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from domains.utils import init_world

## Params

all_model_paths = {'sequential-goals-val-and-init': ['logs/experiments/sequential-goals-20220216-182615',
                                        'logs/experiments/sequential-goals-20220216-183036',
                                        'logs/experiments/sequential-goals-20220216-183101'],
                   'sequential-goals-init': ['logs/experiments/sequential-goals-noval-20220216-205218',
                                        'logs/experiments/sequential-goals-noval-20220216-205236',
                                        'logs/experiments/sequential-goals-noval-20220217-040812'],
                   'sequential-goals-old': ['logs/experiments/sequential-goals-20220208-025351',
                                        'logs/experiments/sequential-goals-20220208-025356',
                                        'logs/experiments/sequential-goals-20220208-025405'],
                   #'random-actions': ['logs/experiments/random-actions-20220207-192035',
                    #                    'logs/experiments/random-actions-20220207-205436',
                    #                    'logs/experiments/random-actions-20220207-223028'],
                   #'random-goals': ['logs/experiments/random-goals-opt-20220207-192020',
                    #                    'logs/experiments/random-goals-opt-20220207-204938',
                    #                    'logs/experiments/random-goals-opt-20220207-221248'],
                   #'sequential-actions': ['logs/experiments/sequential-plans-20220208-011124',
                    #                    'logs/experiments/sequential-plans-20220208-141044',
                    #                    'logs/experiments/sequential-plans-20220208-230530'],
                   }

batch_size = 16
n_epochs = 300
n_hidden = 32
n_layers = 5
n_of_in = 1
n_af_in = 7
n_ef_in = 3
legend_title = 'Method'
test_dataset_path = 'logs/experiments/100_random_goals_balanced-20220217-041010'

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset('')
    gts = [int(y) for _,y in test_dataset]
    
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
            for model, mii in model_logger.get_model_iterator():
                preds = [model_forward(model, x, single_batch=True).squeeze().mean().round() for x,_ in test_dataset]
                accuracy = np.mean([(pred == gt) for pred, gt in zip(preds, gts)])
                accuracies.append(accuracy)
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
    ax.legend(title=legend_title)
    ax.set_ylim([0.3,1])
    plt.savefig('model_accuracy.svg', format='svg')
    #plt.show()

