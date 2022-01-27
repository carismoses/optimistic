from learning.utils import ExperimentLogger
from domains.utils import init_world

exp_paths = ['logs/experiments/sequential_goals-20220120-030537',
                'logs/experiments/sequential_goals-20220120-030624',
                'logs/experiments/sequential_goals-20220120-030629',
                'logs/experiments/sequential_goals-20220120-030635',
                'logs/experiments/sequential_goals-20220120-030640']

import pdb; pdb.set_trace()
for exp_path in exp_paths:
    print(exp_path)
    # get logger and set up world
    logger = ExperimentLogger(exp_path)
    world = init_world('tools',
                        None,
                        'optimistic',
                        False,
                        logger)

    # get largest dataset to see highest index
    trans_dataset = logger.load_trans_dataset()
    max_i = len(trans_dataset)
    print(max_i)

    # for each index in the dataset plot the goal
    for i in range(max_i):
        world.plot_datapoint(i)
