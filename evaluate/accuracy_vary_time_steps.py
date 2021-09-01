import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools

from tamp.predicates import On
from planning import plan
from domains.ordered_blocks.world import OrderedBlocksWorldGT, OrderedBlocksWorldOpt
from learning.utils import ExperimentLogger
from learning.datasets import model_forward
from evaluate.utils import vec_to_logical_state, plot_horiz_bars, join_strs, \
                                stacked_blocks_to_str, plot_results, recc_dict, potential_actions

def calc_time_step_accs(test_num_blocks, test_dataset_path, model_paths):
    test_dataset_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_dataset_logger.load_trans_dataset()
    test_dataset.set_pred_type('class')

    all_model_accuracies = []
    for mi, model_path in enumerate(model_paths):
        model_logger = ExperimentLogger(model_path)
        train_dataset_path = model_logger.load_args().dataset_exp_path
        train_dataset_logger = ExperimentLogger(train_dataset_path)
        accuracies = []
        i = 0       # current model and dataset index
        max_T = 0   # current time step (there are potentially multiple timesteps between i's)
        for t in itertools.count(1):
            if t > max_T:
                i += 1
                try:
                    trans_model = model_logger.load_trans_model(i=i)
                    train_dataset = train_dataset_logger.load_trans_dataset(i=i)
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
    # these datasets are generated with random exploration
    test_datasets = {#2: 'learning/experiments/logs/datasets/large-test-2-20210810-223800',
                    #3: 'learning/experiments/logs/datasets/large-test-3-20210810-223754',
                    4: 'learning/experiments/logs/datasets/large-test-4-20210810-223746',
                    #5: 'learning/experiments/logs/datasets/large-test-5-20210810-223740',
                    #6: 'learning/experiments/logs/datasets/large-test-6-20210810-223731',
                    #7:'learning/experiments/logs/datasets/large-test-7-20210811-173148',
                    8:'learning/experiments/logs/datasets/large-test-8-20210811-173210'}

    compare_opt = False  # if want to compare against the optimistic model

    explore = ['learning/experiments/logs/models/model-random-actions-20210825-095417',
                    'learning/experiments/logs/models/model-random-actions-20210825-095458',
                    'learning/experiments/logs/models/model-random-actions-20210825-095540',
                    'learning/experiments/logs/models/model-random-actions-20210825-095613',
                    'learning/experiments/logs/models/model-random-actions-20210825-095642',
                    'learning/experiments/logs/models/model-random-actions-20210825-095721',
                    'learning/experiments/logs/models/model-random-actions-20210825-095800',
                    'learning/experiments/logs/models/model-random-actions-20210825-095841',
                    'learning/experiments/logs/models/model-random-actions-20210825-095922',
                    'learning/experiments/logs/models/model-random-actions-20210825-095959',
                    'learning/experiments/logs/models/model-random-actions-20210825-100039',
                    'learning/experiments/logs/models/model-random-actions-20210825-100130',
                    'learning/experiments/logs/models/model-random-actions-20210825-100201',
                    'learning/experiments/logs/models/model-random-actions-20210825-100239',
                    'learning/experiments/logs/models/model-random-actions-20210825-100341',
                    'learning/experiments/logs/models/model-random-actions-20210825-100418',
                    'learning/experiments/logs/models/model-random-actions-20210825-100508',
                    'learning/experiments/logs/models/model-random-actions-20210825-100533',
                    'learning/experiments/logs/models/model-random-actions-20210825-100610',
                    'learning/experiments/logs/models/model-random-actions-20210825-100700']
    exploit_T_opt_deep = ['learning/experiments/logs/models/model-test-random-goals-opt-20210823-165549',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-165810',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-170034',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-170322',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-170613',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-171415',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174213',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174438',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174637',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174845',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175114',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175401',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175731',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175921',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180142',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180404',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180615',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180734',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-181009',
                    'learning/experiments/logs/models/model-test-random-goals-opt-20210823-181216']
    exploit_T_learned_deep = ['learning/experiments/logs/models/model-test-random-goals-learned-20210823-180152',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-165618',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-170031',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-170422',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-171341',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-174239',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-174658',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-175109',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-175726',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-180630',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-181136',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-181550',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-181840',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-183900',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-185941',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-190252',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-190613',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-191022',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-191356',
                    'learning/experiments/logs/models/model-test-random-goals-learned-20210823-191700']

    model_paths = {'explore': explore,
                    'exploit-T-opt': exploit_T_opt_deep,
                    'exploit-T-learned': exploit_T_learned_deep}
########

    #time_step_accs = recc_dict()
    #plan_paths = recc_dict()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # run for each method and model
    for test_num_blocks, test_dataset_path in test_datasets.items():
        fig, ax = plt.subplots()
        for mi, (method, method_model_paths) in enumerate(model_paths.items()):
            avg_acc, std_acc = calc_time_step_accs(test_num_blocks, test_dataset_path, method_model_paths)
            ax.plot(avg_acc, color=cs[mi], label=method)
            ax.fill_between(avg_acc-std_acc, avg_acc+std_acc, color=cs[mi], alpha=0.1)
            ax.set_title('Accuracy over Time for %i Blocks' % test_num_blocks)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Training Timesteps')
            ax.set_ylim(0, 1.1)
            ax.legend(title='Method')
    plt.show()
    '''
    if compare_opt:
        for test_num_blocks in test_datasets:
            test_dataset_path = test_datasets[test_num_blocks]
            test_dataset_logger = ExperimentLogger(test_dataset_path)
            test_trans_dataset = test_dataset_logger.load_trans_dataset()
            test_heur_dataset = test_dataset_logger.load_heur_dataset()
            test_trans_dataset.set_pred_type('full_state')
            trans_success_data['opt'][test_num_blocks] = calc_trans_accuracy('opt', test_trans_dataset, test_num_blocks)
            full_trans_success_data['opt'][test_num_blocks] = calc_full_trans_accuracy('opt', \
                                                            test_num_blocks, \
                                                            model=trans_model)

    # Save data to logger
    logger = ExperimentLogger.setup_experiment_directory(args, 'model_accuracy')
    logger.save_plot_data([test_datasets, trans_success_data, heur_success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    xlabel = 'Number of Test Blocks'
    trans_title = 'Transition Model Performance with Learned\nModels in %s Block World' % model_logger.load_args().num_blocks  # TODO: hack
    trans_ylabel = 'Average Accuracy'
    all_test_num_blocks = list(test_datasets.keys())
    plot_results(trans_success_data, all_test_num_blocks, trans_title, xlabel, trans_ylabel, logger)
    '''
