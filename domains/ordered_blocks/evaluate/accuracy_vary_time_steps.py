import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools

from experiments.utils import ExperimentLogger
from learning.utils import model_forward
from domains.ordered_blocks.world import OrderedBlocksWorld
from evaluate.utils import recc_dict

all_time_step_accs = recc_dict()

def calc_time_step_accs(test_num_blocks, test_dataset, method, model_paths, world):
    all_model_accuracies = []
    # NOTE: this assumes all action validity can be correctly determined from the initial state!
    # this won't always be the case!
    vof, vef = world.state_to_vec(world.get_init_state(), num_blocks=test_num_blocks)
    for model_path in model_paths:
        logger = ExperimentLogger(model_path)
        accuracies = []
        i = 0       # current model and dataset index (within a model_path)
        max_T = 0   # current max time step given i
        for t in itertools.count(1):
            if t > max_T:
                i += 1
                try:
                    trans_model = logger.load_trans_model(i=i)
                    train_dataset = logger.load_trans_dataset(i=i)
                    max_T = len(train_dataset)
                except:
                    print('Model %s ends at timestep %i' % (model_path, max_T))
                    break
            preds, gts = [], []
            for action, label in test_dataset:
                va = world.action_to_vec(action)
                x = [vof, vef, va]
                preds.append(model_forward(trans_model, x).round().squeeze())
                gts.append(label)
            accuracy = np.mean([(pred == gt) for pred, gt in zip(preds, gts)])
            accuracies.append(accuracy)
        all_model_accuracies.append(accuracies)
        all_time_step_accs[test_num_blocks][method][model_path] = accuracies

    # calculate avg and std accuracies accross all models
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

def calc_opt_accuracy(test_dataset):
    # NOTE: this assumes all action validity can be correctly determined from the initial state!
    # this won't always be the case!
    accuracies = []
    for _, gt_pred in test_dataset:
        model_pred = 1. # optimistic model always thinks it's right
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
    # This plots the accuracy of the models in results_path.py where the x axis
    # is the number actions trained on so far

    # import train
    from domains.ordered_blocks.results_paths import model_paths
    compare_opt = True  # if want to compare against the optimistic model
    all_test_num_blocks = [2, 3, 4, 5, 6, 7, 8, 9]
    ########

    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    logger = ExperimentLogger.setup_experiment_directory(args, 'model_accuracy')

    # generate dataset for each test_num_blocks
    world = OrderedBlocksWorld(max(all_test_num_blocks), use_panda=False, vis=False)
    test_datasets = {}
    for test_num_blocks in all_test_num_blocks:
        num_blocks_dataset = []
        all_opt_actions = world.all_optimistic_actions(num_blocks=test_num_blocks)
        for opt_action in all_opt_actions:
            num_blocks_dataset.append([opt_action, world.valid_transition(opt_action)])
        test_datasets[test_num_blocks] = num_blocks_dataset

    # run for each method and model
    for test_num_blocks, test_dataset in test_datasets.items():
        if test_num_blocks in all_test_num_blocks:
            fig, ax = plt.subplots()
            for mi, (method, method_model_paths) in enumerate(model_paths.items()):
                avg_acc, std_acc = calc_time_step_accs(test_num_blocks,
                                                        test_dataset,
                                                        method,
                                                        method_model_paths,
                                                        world)
                xs = np.arange(len(avg_acc))
                ax.plot(xs, avg_acc, color=cs[mi], label=method)
                ax.fill_between(xs, avg_acc-std_acc, avg_acc+std_acc, color=cs[mi], alpha=0.1)
            if compare_opt:
                final_accuracy = calc_opt_accuracy(test_dataset)
                ax.plot([0, len(avg_acc)], [final_accuracy, final_accuracy], color= cs[mi+1], label='opt')
            title = 'Accuracy over Time for %i Blocks' % test_num_blocks
            ax.set_title(title)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Training Timesteps')
            ax.set_ylim(0, 1.1)
            ax.legend(title='Method')

            # save plots to logger
            plt.savefig('%s/%s.png' % (logger.exp_path, title))
            print('Saving figures to %s.' % logger.exp_path)

    # Save data to logger
    logger.save_plot_data(all_time_step_accs)
    print('Saving data to %s.' % logger.exp_path)
