import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld

## Params

all_model_paths = {'sequential-goals': ['seq-20220408-192725',
                                        'seq-20220408-193202',
                                        'seq-20220408-193207']}#,
'''
                    'random-goals': ['random-goals-opt-20220326-155213',
                                        'random-goals-opt-20220326-155206',
                                        'random-goals-opt-20220326-155154'],
                    'sequential-actions': ['seq-act-20220404-132835',
                                        'seq-act-20220404-132901',
                                        'seq-act-20220404-132911'],
                    'random-actions': ['random-actions-20220403-024343',
                                        'random-actions-20220403-024354',
                                        'random-actions-20220403-024404'],
                    }
'''

all_actions = ['pick', 'move_contact-poke', 'move_contact-push_pull', 'move_holding']
all_objs = ['yellow_block', 'blue_block']
all_grasps = ['p1', 'n1']
test_dataset_path = 'logs/experiments/seq50-20220404-131632'
max_actions = None

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset('')
    gts = {}
    for action in all_actions:
        gts[action] = {}
        for obj in all_objs:
            if 'move_contact' in action:
                for grasp in all_grasps:
                    gts[action][obj][grasp] = [int(y) for _,y in test_dataset.datasets[action][obj][grasp]]
            else:
                gts[action][obj]['None'] = [int(y) for _,y in test_dataset.datasets[action][obj]['None']]

    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)

        # get max_actions and interpolate so all lists same length
        if not max_actions:
            all_txs = []
            for mi, model_path in enumerate(model_paths):
                model_logger = ExperimentLogger('logs/experiments/'+model_path)
                _, txs = model_logger.get_dir_indices('models')
                all_txs.append(txs)
            max_actions = min([sorted(txs)[-1] for txs in all_txs])

        # calculate all accuracies
        all_accuracies = []
        all_action_steps = []
        for mi, model_path in enumerate(model_paths):
            print('Model path: %s' % model_path)
            accuracies = []
            n_actions = []
            model_logger = ExperimentLogger('logs/experiments/'+model_path)
            _, txs = model_logger.get_dir_indices('models')
            #for ensembles, mii in model_logger.get_model_iterator():
            for mii in sorted(txs)[::20]:
                if mii < max_actions:
                    ensembles = model_logger.load_trans_model(i=mii)
                    model_accuracies = []
                    for action in all_actions:
                        for obj in all_objs:
                            if 'move_contact' in action:
                                for grasp in all_grasps:
                                    dataset = test_dataset.datasets[action][obj][grasp]
                                    model_preds = [model_forward(ensembles, x, action, obj, grasp, single_batch=True).squeeze().mean().round() \
                                                for x,_ in dataset]
                                    model_accuracies += [(pred == gt) for pred, gt in zip(model_preds, gts[action][obj][grasp])]
                            else:
                                grasp = 'None'
                                dataset = test_dataset.datasets[action][obj][grasp]
                                model_preds = [model_forward(ensembles, x, action, obj, grasp, single_batch=True).squeeze().mean().round() \
                                            for x,_ in dataset]
                                model_accuracies += [(pred == gt) for pred, gt in zip(model_preds, gts[action][obj][grasp])]
                    accuracies.append(np.mean(model_accuracies))
                    n_actions.append(mii)
            all_accuracies.append(accuracies)
            all_action_steps.append(n_actions)

        xs = np.arange(max_actions)
        full_accuracies = []
        for model_actions, model_accs in zip(all_action_steps, all_accuracies):
            interp_model_accs = np.interp(xs, model_actions, model_accs)
            full_accuracies.append(interp_model_accs)

        # plot avg and standard dev
        all_accuracies = np.array(full_accuracies)
        avg_accs = np.mean(all_accuracies, axis=0)
        std_accs = np.std(all_accuracies, axis=0)
        ax.plot(xs, avg_accs, color=cs[pi % len(cs)], label=method)
        ax.fill_between(xs, avg_accs-std_accs, avg_accs+std_accs, color=cs[pi % len(cs)], alpha=0.1)

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Model Accuracy over Training Time')
    ax.legend(title='Method', loc='lower right')
    ax.set_ylim([0.3,1])
    plt.savefig('model_accuracy.svg', format='svg')
    #plt.show()

