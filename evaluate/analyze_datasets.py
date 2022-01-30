from learning.utils import ExperimentLogger
import matplotlib.pyplot as plt
import numpy as np

dataset_paths = ['logs/experiments/test_dataset_progress0p0-20220127-194636',
                    'logs/experiments/test_dataset_progress0p1-20220127-195002',
                    'logs/experiments/test_dataset_progress0p2-20220127-195037',
                    'logs/experiments/test_dataset_progress0p3-20220127-195109',
                    'logs/experiments/test_dataset_progress0p4-20220127-200542',
                    'logs/experiments/test-dataset-gp5',
                    'logs/experiments/test_dataset_progress0p6-20220127-201713',
                    'logs/experiments/test_dataset_progress0p7-20220127-202811',
                    'logs/experiments/test_dataset_progress0p8-20220127-195212',
                    'logs/experiments/test_dataset_progress0p9-20220127-202903',
                    'logs/experiments/test_dataset_progress1p0-20220127-202921']

import pdb; pdb.set_trace()

for path in dataset_paths:
    logger = ExperimentLogger(path)
    dataset = logger.load_trans_dataset()
    n_total = len(dataset)
    n_successes = sum([y for x,y in dataset])
    percent_success = n_successes / n_total
    print(path)
    print('    %f' % percent_success)
