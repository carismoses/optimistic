import argparse
import numpy as np
import matplotlib.pyplot as plt

from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import INF

from tamp.utils import execute_plan
from domains.ordered_blocks.world import OrderedBlocksWorld
from experiments.utils import ExperimentLogger
from evaluate.utils import plot_results, recc_dict

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
    # This plots the % Success (out of num_goals set below) of the models in
    # results_path.py where the x axis is the number of blocks in the test domain

    # import train and test datasets
    from domains.ordered_blocks.test_datasets import test_datasets
    from domains.ordered_blocks.results_paths import model_paths
    compare_opt = False  # if want to compare against the optimistic model
    num_goals = 20
    ########

    success_data = recc_dict()
    rank_success_data = recc_dict()
    plan_paths = recc_dict()

    # generate random goals to evaluate planner with
    print('Generating random goals.')
    test_goals = {}
    for test_num_blocks in test_datasets:
        world = OrderedBlocksWorld(test_num_blocks, False) # TODO: don't assume no robot
        test_goals[test_num_blocks] = [world.generate_random_goal(feasible=True) for _ in range(num_goals)]
    print('Done generating goals.')

    if compare_opt:
        model_paths['opt'] = [None]

    for method_name, method_paths in model_paths.items():
        print('method', method_name)
        for test_num_blocks in test_datasets:
            print('test blocks', test_num_blocks)
            domain_args = [str(test_num_blocks), 'False'] # TODO: don't assume no robot
            for model_path in method_paths:
                print('model path', model_path)
                if method_name == 'opt':
                    world, opt_pddl_info, pddl_info = OrderedBlocksWorld.init(domain_args,
                                                                            'optimistic')
                else:
                    model_logger = ExperimentLogger(model_path)
                    world, opt_pddl_info, pddl_info = OrderedBlocksWorld.init(domain_args,
                                                                            'learned',
                                                                            model_logger)
                init_state = world.get_init_state()
                successes = 0
                for gi, goal in enumerate(test_goals[test_num_blocks]):
                    print('goal number', gi)
                    problem = tuple([*pddl_info, init_state, goal])
                    pddl_plan, cost, init_expanded = solve_focused(problem,
                                                        success_cost=INF,
                                                        max_skeletons=2,
                                                        search_sample_ratio=1.0,
                                                        max_time=INF,
                                                        verbose=False,
                                                        unit_costs=True)
                    if pddl_plan is not None:
                        trajectory, valid_transition = execute_plan(world, problem, pddl_plan, init_expanded)
                        if len(trajectory) == len(pddl_plan):
                            successes += 1
                if method_name == 'opt':
                    success_data[method_name][test_num_blocks] = successes/num_goals
                else:
                    success_data[method_name][test_num_blocks][model_path] = successes/num_goals

    # Save data to logger
    logger = ExperimentLogger.setup_experiment_directory(args, 'plan_results')
    logger.save_plot_data([success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    train_num_blocks = model_logger.load_args().domain_args[0] # HACK
    xlabel = 'Number of Test Blocks'
    trans_title = 'Planning Performance with Learned\nModels in %s Block World' % train_num_blocks
    trans_ylabel = '% Success'
    all_test_num_blocks = list(test_datasets.keys())
    plot_results(success_data, all_test_num_blocks, trans_title, xlabel, trans_ylabel, logger)
