import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools

from learning.utils import ExperimentLogger
from learning.datasets import model_forward

def calc_time_step_accs(test_num_blocks, test_dataset_path, model_paths):
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset()
    test_dataset.set_pred_type('class')

    all_model_accuracies = []
    for mi, model_path in enumerate(model_paths):
        logger = ExperimentLogger(model_path)
        accuracies = []
        i = 0       # current model and dataset index
        max_T = 0   # current time step (there are potentially multiple timesteps between i's)
        for t in itertools.count(1):
            if t > max_T:
                i += 1
                try:
                    trans_model = logger.load_trans_model(i=i)
                    train_dataset = logger.load_trans_dataset(i=i)
                    max_T = len(train_dataset)
                except:
                    break
            preds = [model_forward(trans_model, x).round().squeeze() for x,y in test_dataset]
            gts = [y.numpy().squeeze() for x,y in test_dataset]
            accuracy = np.mean([(pred == gt) for pred, gt in zip(preds, gts)])
            accuracies.append(accuracy)
        all_model_accuracies.append(accuracies)

    # plot all accuracies
    max_T = max([len(ma) for ma in all_model_accuracies])
    avg_accuracies = []
    std_accuracies = []
    for t in range(max_T):
        all_t_accuracies = []
        for model_accuracies in all_model_accuracies:
            if t >= len(model_accuracies):
                all_t_accuracies.append(model_accuracies[-1])
            else:
                all_t_accuracies.append(model_accuracies[t])
        avg_accuracies.append(np.mean(all_t_accuracies))
        std_accuracies.append(np.std(all_t_accuracies))

    return np.array(avg_accuracies), np.array(std_accuracies)

def calc_opt_accuracy(test_dataset_path):
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset()
    test_dataset.set_pred_type('class')
    accuracies = []
    for x, y in test_dataset:
        # get all relevant transition info
        vof, vef, va = [xi.detach().numpy() for xi in x]
        model_pred = 1. # optimistic model always thinks it's right
        gt_pred = y.numpy().squeeze()
        accuracy = int(np.array_equal(gt_pred, model_pred))
        accuracies.append(accuracy)
    return np.mean(accuracies)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='where to save exp data')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    #### Parameters ####
    # import train and test datasets
    from domains.ordered_blocks.test_datasets import test_datasets
    from domains.ordered_blocks.results_paths import model_paths
    compare_opt = True  # if want to compare against the optimistic model
    ########

    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # run for each method and model
    for test_num_blocks, test_dataset_path in test_datasets.items():
        if test_num_blocks in [2]:
            fig, ax = plt.subplots()
            for mi, (method, method_model_paths) in enumerate(model_paths.items()):
                avg_acc, std_acc = calc_time_step_accs(test_num_blocks, test_dataset_path, method_model_paths)
                ax.plot(avg_acc, color=cs[mi], label=method)
                ax.fill_between(avg_acc-std_acc, avg_acc+std_acc, color=cs[mi], alpha=0.1)
            if compare_opt:
                final_accuracy = calc_opt_accuracy(test_dataset_path)
                ax.plot([0, len(avg_acc)], [final_accuracy, final_accuracy], color= cs[mi+1], label='opt')
            ax.set_title('Accuracy over Time for %i Blocks' % test_num_blocks)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Training Timesteps')
            ax.set_ylim(0, 1.1)
            ax.legend(title='Method')
    plt.show()
