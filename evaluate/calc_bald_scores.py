import matplotlib.pyplot as plt

from experiments.utils import ExperimentLogger
from experiments.strategies import bald
from learning.utils import model_forward
from domains.utils import init_world

##
#dataset_paths = {'0': ['logs/experiments/sequential_goals-20220120-030537'],
#'1':['logs/experiments/sequential_goals-20220120-030624'],
#'2':['logs/experiments/sequential_goals-20220120-030629'],
#'3':['logs/experiments/sequential_goals-20220120-030635'],
#'4':['logs/experiments/sequential_goals-20220120-030640']}
##

## sequential goals

dataset_paths = {'0': ['logs/experiments/sequential_goals-20220120-211043'],
                '1': ['logs/experiments/sequential_goals-20220120-211049'],
                '2': ['logs/experiments/sequential_goals-20220120-211054'],
                '3': ['logs/experiments/sequential_goals-20220120-211058'],
                '4': ['logs/experiments/sequential_goals-20220120-211103']}
#fig, ax = plt.subplots()

#import pdb; pdb.set_trace()
cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for ei, (exp_name, exp_path) in enumerate(dataset_paths.items()):
    exp_path = exp_path[0]
    bald_scores = []
    logger = ExperimentLogger(exp_path)
    model = logger.load_trans_model(i=0)
    next_model_i = 12
    xs = []
    for dataset, di in logger.get_dataset_iterator():
        if di > next_model_i:
            model = logger.load_trans_model(i=next_model_i)
            next_model_i += 12
        if len(dataset) > 0:
            (of, ef, af), l = dataset[-1]
            predictions = model_forward(model, [of, ef, af], single_batch=True)
            mean_prediction = predictions.mean()
            score = mean_prediction*bald(predictions)
            bald_scores.append(score)
            xs.append(di)
    plt.plot(xs, bald_scores, color=cs[ei], label=exp_name)
plt.ylim([0.0, 0.4])
plt.legend()
plt.xlabel('Num Actions')
plt.ylabel('BALD Score')
plt.title('BALD Scores')
#plt.show()
plt.savefig('bald_scores.png')
