import argparse
import numpy as np
import matplotlib.pyplot as plt

from domains.ordered_blocks.world import OrderedBlocksWorld
from learning.utils import ExperimentLogger
from learning.datasets import model_forward
from evaluate.utils import recc_dict, plot_results

def calc_full_trans_accuracy(model_type, test_num_blocks, world, model):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    accuracies = {}
    # NOTE: we test all actions from initial state assuming that the network is ignoring the state
    all_actions = world.all_optimistic_actions(num_blocks=test_num_blocks)
    init_state = world.get_init_state()
    vof, vef = world.state_to_vec(init_state, num_blocks=test_num_blocks)
    for action in all_actions:
        va = world.action_to_vec(action)
        gt_pred = world.valid_transition(action)
        if model_type == 'opt':
            model_pred = 1. # optimistic model always thinks it's right
        else:
            model_pred = model_forward(model, [vof, vef, va]).round().squeeze()
        str_a = '(%i, %i)' % (action.args[0].num, action.args[2].num)
        accuracies[(gt_pred, str_a)] = int(np.array_equal(gt_pred, model_pred))
    return accuracies

def plot_full_accuracies(method, method_full_trans_success_data):
    fig, axes = plt.subplots(len(method_full_trans_success_data), 1, sharex=True)
    fig.set_size_inches(12, 7)

    def plot_accs(avg_acc, std_acc, action, nb, ni):
        axes[ni].bar(action,
                    avg_acc,
                    color='r' if avg_acc<0.5 else 'g',
                    yerr=std_acc)
        axes[ni].set_ylabel('Num Blocks = %i' % nb, rotation=0, va='center', ha='right')

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
    plt.xlabel('Action')
    # slow but works
    for tick in axes[-1].get_xticklabels():
        tick.set_rotation(90)
    plt.savefig('%s/%s.png' % (logger.exp_path, title))


def calc_trans_accuracy(model_type, test_dataset, test_num_blocks, world, model=None):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    accuracies = []
    vof, vef = world.state_to_vec(world.get_init_state(), num_blocks=test_num_blocks)
    for va, gt_pred in test_dataset:
        # get all relevant transition info
        x = [vof, vef, va]
        if model_type == 'opt':
            model_pred = 1. # optimistic model always thinks it's right
        else:
            model_pred = model_forward(model, x).round().squeeze()
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
    # import train datasets
    from domains.ordered_blocks.results_paths import model_paths
    all_test_num_blocks = [2, 3, 4, 5, 6, 7, 8]
    compare_opt = True  # if want to compare against the optimistic model
    ########

    # generate dataset for each test_num_blocks
    world = OrderedBlocksWorld(max(all_test_num_blocks), use_panda=False, vis=False)
    test_datasets = {}
    for test_num_blocks in all_test_num_blocks:
        num_blocks_dataset = []
        all_opt_actions = world.all_optimistic_actions(num_blocks=test_num_blocks)
        for opt_action in all_opt_actions:
            num_blocks_dataset.append([world.action_to_vec(opt_action),
                                        world.valid_transition(opt_action)])
        test_datasets[test_num_blocks] = num_blocks_dataset

    trans_success_data = recc_dict()
    full_trans_success_data = recc_dict()

    # run for each method and model
    for method, method_model_paths in model_paths.items():
        for test_num_blocks, test_dataset in test_datasets.items():
            for model_path in method_model_paths:
                model_logger = ExperimentLogger(model_path)
                model_args = model_logger.load_args()
                trans_model = model_logger.load_trans_model()
                trans_accuracy = calc_trans_accuracy('learned',
                                                test_dataset,
                                                test_num_blocks,
                                                world,
                                                model=trans_model)
                full_trans_success_data[method][test_num_blocks][model_path] = \
                                        calc_full_trans_accuracy('learned',
                                                                test_num_blocks,
                                                                world,
                                                                trans_model)
                trans_success_data[method][test_num_blocks][model_path] = trans_accuracy

    if compare_opt:
        for test_num_blocks, test_dataset in test_datasets.items():
            trans_success_data['opt'][test_num_blocks] = calc_trans_accuracy('opt',
                                                                    test_dataset,
                                                                    test_num_blocks,
                                                                    world)
            full_trans_success_data['opt'][test_num_blocks] = calc_full_trans_accuracy('opt',
                                                            test_num_blocks,
                                                            world,
                                                            trans_model)

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
