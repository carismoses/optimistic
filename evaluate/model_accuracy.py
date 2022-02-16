import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from domains.utils import init_world

## Params
'''
all_model_paths = {'0p1': ['logs/experiments/unbalanced_0p1-20220201-201256',
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
'''
all_model_paths = {'old': ['logs/experiments/sequential_goals-20220201-001349'],
                   'old1': ['logs/experiments/sequential_goals-20220201-001157'],
                   'new': ['logs/experiments/sequential_goals-20220131-223904'],
                   'new1': ['logs/experiments/sequential_goals-20220131-223918']}
test_dataset_paths = {'0p0': 'logs/experiments/balanced_dataset_0p0-20220128-215624',
                        '0p1': 'logs/experiments/balanced_dataset_0p1-20220128-215512',
                        '0p2': 'logs/experiments/balanced_dataset_0p2-20220128-215323',
                        '0p3': 'logs/experiments/balanced_dataset_0p3-20220128-215401',
                        '0p4': 'logs/experiments/balanced_dataset_0p4-20220128-193801',
                        '0p5': 'logs/experiments/balanced_dataset_0p5-20220128-193727',
                        '0p6': 'logs/experiments/balanced_dataset_0p6-20220128-193821',
                        '0p7': 'logs/experiments/balanced_dataset_0p7-20220128-214728',
                        '0p8': 'logs/experiments/balanced_dataset_0p8-20220128-214759',
                        '0p9': 'logs/experiments/balanced_dataset_0p9-20220128-214825',
                        '1p0': 'logs/experiments/balanced_dataset_1p0-20220128-214857'}
batch_size = 16
n_epochs = 300
n_hidden = 32
n_layers = 5
n_of_in = 1
n_af_in = 7
n_ef_in = 3
legend_title = 'Method' # or 'Progress'

if __name__ == '__main__':
    import pdb; pdb.set_trace()

    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)
        all_accuracies = []
        test_dataset_method = method if method in test_dataset_paths else '1p0'
        test_dataset_logger = ExperimentLogger(test_dataset_paths[test_dataset_method])
        test_dataset = test_dataset_logger.load_trans_dataset('')
        gts = [int(y) for _,y in test_dataset]
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

        condition = len(set([len(acc_list) for acc_list in all_accuracies])) == 1
        assert condition, 'all models for method %s do not contain the same amount of actions/models' % method
        all_accuracies = np.array(all_accuracies)
        avg_accs = np.mean(all_accuracies, axis=0)
        std_accs = np.std(all_accuracies, axis=0)
        ax.plot(n_actions, avg_accs, color=cs[pi % len(cs)], label=method)
        ax.fill_between(n_actions, avg_accs-std_accs, avg_accs+std_accs, color=cs[pi % len(cs)], alpha=0.1)

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Model Accuracy over Training Time')
    ax.legend(title=legend_title)
    ax.set_ylim([0.43,1])
    plt.savefig('model_accuracy.svg', format='svg')
    #plt.show()
