import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from learning.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from domains.utils import init_world

## Params
model_paths = {'0p0': 'logs/experiments/test_dataset_progress0p0-20220127-194636',
                '0p1': 'logs/experiments/test_dataset_progress0p1-20220127-195002',
                '0p2': 'logs/experiments/test_dataset_progress0p2-20220127-195037',
                '0p3': 'logs/experiments/test_dataset_progress0p3-20220127-195109',
                '0p4': 'logs/experiments/test_dataset_progress0p4-20220127-200542',
                #'0p5': 'logs/experiments/test_dataset_progress0p5-20220128-214234',
                '0p6': 'logs/experiments/test_dataset_progress0p6-20220127-201713',
                '0p7': 'logs/experiments/test_dataset_progress0p7-20220127-202811',
                '0p8': 'logs/experiments/test_dataset_progress0p8-20220127-195212',
                '0p9': 'logs/experiments/test_dataset_progress0p9-20220127-202903',
                '1p0': 'logs/experiments/test_dataset_progress1p0-20220127-202921'}

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
max_actions = 300



if __name__ == '__main__':
    import pdb; pdb.set_trace()

    fig, ax = plt.subplots()

    for progress, model_path in model_paths.items():
        model_logger = ExperimentLogger(model_path)
        model_logger.args.data_collection_mode = 'random-goals-opt'
        model_logger.args.n_models = 1
        test_dataset_logger = ExperimentLogger(test_dataset_paths[progress])
        test_dataset = test_dataset_logger.load_trans_dataset()
        gts = [int(y) for _,y in test_dataset]

        accuracies = []
        n_actions = []
        world = init_world('tools',
                       None,
                       'optimistic',
                       False,
                       model_logger)
        for model, mi in model_logger.get_model_iterator(world):
            n_actions.append(mi)
            preds = [model_forward(model, x, single_batch=True).squeeze().mean().round() for x,_ in test_dataset]
            accuracy = np.mean([(pred == gt) for pred, gt in zip(preds, gts)])
            accuracies.append(accuracy)

        ax.plot(n_actions, accuracies, label=progress)

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Model Accuracy over Training Time for varying Goal Progresses')
    ax.legend(title='Goal Progress')
    ax.set_ylim([0.43,1])
    plt.savefig('model_accuracy')
    #plt.show()

