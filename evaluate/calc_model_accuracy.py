import numpy as np

from experiments.utils import ExperimentLogger
from learning.utils import model_forward
from domains.tools.world import CONTACT_TYPES


test_dataset_path = 'logs/experiments/90_random_goals_balanced-20220219-170059'
model_path = 'logs/experiments/90_random_goals_balanced-20220219-170056'

test_dataset_logger = ExperimentLogger(test_dataset_path)
test_dataset = test_dataset_logger.load_trans_dataset('')
gts = {}
for type, dataset in test_dataset.datasets.items():
    gts[type] = [int(y) for _,y in dataset]

model_logger = ExperimentLogger(model_path)
model = model_logger.load_trans_model()
preds = {}
for type, dataset in test_dataset.datasets.items():
    preds = [model_forward(type, model, x, single_batch=True).squeeze().mean().round() for x,_ in dataset]
    accuracy = np.mean([(pred == gt) for pred, gt in zip(preds, gts[type])])
    print('Accuracy for %s: %f' % (type, accuracy))
