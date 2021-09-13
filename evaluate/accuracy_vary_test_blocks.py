import argparse
import numpy as np
import matplotlib.pyplot as plt

from domains.ordered_blocks.world import OrderedBlocksWorld
from learning.utils import ExperimentLogger
from learning.datasets import model_forward
from evaluate.utils import recc_dict, plot_results

def calc_full_trans_accuracy(model_type, test_num_blocks, model):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    accuracies = {}
    # NOTE: we test all actions from initial state assuming that the network is ignoring the state
    world = OrderedBlocksWorld(test_num_blocks, False)
    pos_actions, neg_actions = world.all_optimistic_actions(test_num_blocks)
    init_state = world.get_init_state()
    vof, vef = world.state_to_vec(init_state)
    for gt_pred, actions in zip([1, 0], [pos_actions, neg_actions]):
        for action in actions:
            if model_type == 'opt':
                model_pred = 1. # optimistic model always thinks it's right
            else:
                int_action = [int(action[0]), int(action[-1])]
                model_pred = model_forward(model, [vof, vef, int_action]).round().squeeze()
            accuracies[(gt_pred, action)] = int(np.array_equal(gt_pred, model_pred))
    return accuracies

def plot_full_accuracies(method, method_full_trans_success_data):
    fig, axes = plt.subplots(len(method_full_trans_success_data), 1, sharex=True)

    def plot_accs(avg_acc, std_acc, action, nb, ni):
        axes[ni].bar(action,
                    avg_acc,
                    color='r' if avg_acc<0.5 else 'g',
                    yerr=std_acc)
        axes[ni].set_ylabel('Num Blocks = %i' % nb)

    for ni, (num_blocks, num_blocks_data) in enumerate(method_full_trans_success_data.items()):
        if method == 'opt':
            for (label, action), accs in num_blocks_data.items():
                avg_acc = accs
                std_acc = 0
                plot_accs(avg_acc, std_acc, action, num_blocks, ni)
        else:
            all_models = list(num_blocks_data.keys())
            # NOTE: all models have same keys
            for (label, action), accs in num_blocks_data[all_models[0]].items():
                all_accs = [num_blocks_data[model][(label, action)] for model in all_models]
                avg_acc = np.mean(all_accs)
                std_acc = np.std(all_accs)
                plot_accs(avg_acc, std_acc, action, num_blocks, ni)

    title = 'Accuracy of Method %s on Different Test Set Num Blocks' % method
    fig.suptitle(title)
    plt.xlabel('Actions')
    plt.savefig('%s/%s.png' % (logger.exp_path, title))


def calc_trans_accuracy(model_type, test_dataset, test_num_blocks, model=None):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    accuracies = []

    for x, y in test_dataset:
        # get all relevant transition info
        vof, vef, va = [xi.detach().numpy() for xi in x]
        if model_type == 'opt':
            model_pred = 1. # optimistic model always thinks it's right
        else:
            model_pred = model_forward(model, x).round().squeeze()
        gt_pred = y.numpy().squeeze()
        accuracy = int(np.array_equal(gt_pred, model_pred))
        accuracies.append(accuracy)
    final_accuracy = np.mean(accuracies)
    return final_accuracy


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

    trans_success_data = recc_dict()
    full_trans_success_data = recc_dict()

    # run for each method and model
    for method, method_model_paths in model_paths.items():
        for test_num_blocks in test_datasets:
            for model_path in method_model_paths:
                model_logger = ExperimentLogger(model_path)
                model_args = model_logger.load_args()
                trans_model = model_logger.load_trans_model()
                test_dataset_path = test_datasets[test_num_blocks]
                test_dataset_logger = ExperimentLogger(test_dataset_path)
                test_trans_dataset = test_dataset_logger.load_trans_dataset()
                test_trans_dataset.set_pred_type(trans_model.pred_type)
                trans_accuracy = calc_trans_accuracy('learned',
                                                test_trans_dataset,
                                                test_num_blocks,
                                                model=trans_model)
                full_trans_success_data[method][test_num_blocks][model_path] = \
                                        calc_full_trans_accuracy('learned',
                                                                test_num_blocks,
                                                                model=trans_model)
                trans_success_data[method][test_num_blocks][model_path] = trans_accuracy

    if compare_opt:
        for test_num_blocks in test_datasets:
            test_dataset_path = test_datasets[test_num_blocks]
            test_dataset_logger = ExperimentLogger(test_dataset_path)
            test_trans_dataset = test_dataset_logger.load_trans_dataset()
            test_trans_dataset.set_pred_type('full_state')
            trans_success_data['opt'][test_num_blocks] = calc_trans_accuracy('opt',
                                                                    test_trans_dataset,
                                                                    test_num_blocks)
            full_trans_success_data['opt'][test_num_blocks] = calc_full_trans_accuracy('opt',
                                                            test_num_blocks,
                                                            model=trans_model)

    # Save data to logger
    logger = ExperimentLogger.setup_experiment_directory(args, 'model_accuracy')
    logger.save_plot_data([test_datasets, trans_success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    xlabel = 'Number of Test Blocks'
    trans_title = 'Transition Model Performance with Learned\nModels in %s ' \
                        'Block World' % model_args.domain_args  # TODO: hack
    trans_ylabel = 'Average Accuracy'
    all_test_num_blocks = list(test_datasets.keys())
    plot_results(trans_success_data,
                all_test_num_blocks,
                trans_title,
                xlabel,
                trans_ylabel,
                logger)

    for method, method_data in full_trans_success_data.items():
        plot_full_accuracies(method, method_data)
