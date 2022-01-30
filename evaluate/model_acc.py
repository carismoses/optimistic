import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from learning.utils import ExperimentLogger
from domains.tools.world import ToolsWorld


## Params
dataset_paths = {#'random-actions':
                 #   ['logs/experiments/random_actions-20220104-201317',
                 #   'logs/experiments/random_actions-20220104-202422',
                 #   'logs/experiments/random_actions-20220104-202440',
                 #   'logs/experiments/random_actions-20220104-202447',
                 #   'logs/experiments/random_actions-20220104-202453'],
                'random-goals-opt':
                    ['logs/experiments/random_goals_opt-20220104-204547',
                    'logs/experiments/random_goals_opt-20220104-203849',
                    'logs/experiments/random_goals_opt-20220104-204627',
                    'logs/experiments/random_goals_opt-20220104-204532',
                    'logs/experiments/random_goals_opt-20220104-204536'],
                'sequential-goals':
                    ['logs/experiments/sequential_goals-20220105-143004',
		    #'logs/experiments/sequential_goals-20220105-143605',
		    'logs/experiments/sequential_goals-20220105-143711',
	            'logs/experiments/sequential_goals-20220105-145239',
                    'logs/experiments/sequential_goals-20220105-145344'],
		'engineered-goals-dist':
		    ['logs/experiments/engineered_goals_dist-20220112-162941',
		    'logs/experiments/engineered_goals_dist-20220112-162947',
		    'logs/experiments/engineered_goals_dist-20220112-162956',
		    'logs/experiments/engineered_goals_dist-20220112-163004',
		    'logs/experiments/engineered_goals_dist-20220112-163058'],
		'engineered-goals-size':
		    ['logs/experiments/engineered_goals_size-20220112-172108',
		    'logs/experiments/engineered_goals_size-20220112-172115',
		    'logs/experiments/engineered_goals_size-20220112-172119',
		    'logs/experiments/engineered_goals_size-20220112-172125',
 		    'logs/experiments/engineered_goals_size-20220112-172129']
		}

#dataset_paths = {'sequential-goals':
#    ['logs/experiments/sequential-goals-20211223-055013']}
test_dataset_path = 'logs/experiments/test_dataset-20211201-123920'
##
batch_size = 16
n_epochs = 300
n_hidden = 32
n_layers = 5
n_of_in = 1
n_af_in = 7
n_ef_in = 3
max_actions = 300


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    test_logger = ExperimentLogger(test_dataset_path)
    test_dataset = test_logger.load_trans_dataset()
    # make test dataset balanced 50/50
    pos_ind, neg_ind = [], []
    for i, (x,y) in enumerate(test_dataset):
        if y == 1:
            pos_ind.append(i)
        else:
            neg_ind.append(i)
    len_list = len(pos_ind)
    remove_list = neg_ind[len_list:]
    test_dataset.remove_elements(remove_list)
    gts = [int(y) for x,y in test_dataset]
    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for mi, (method, method_paths) in enumerate(dataset_paths.items()):
        all_accs_array = np.zeros((len(method_paths), max_actions))
        print(method)
        for pi, path in enumerate(method_paths):
            print(path)
            logger = ExperimentLogger(path)
            world = ToolsWorld.init([],
                                'optimistic',
                                False,
                                logger)
            xs = []
            ys = []
            for dataset, di in logger.get_dataset_iterator():
                try:
                    model = logger.load_trans_model(world, i=di)
                except:
                    pass
                if 'actions' in method:
                    xs.append(di)
                else:
                    xs.append(4*len(dataset))
                preds = []
                #print(di)
                for tdi, (x,y) in enumerate(test_dataset):
                    pred = model_forward(model, x, single_batch=True).squeeze().mean().round()
                    #print(pred)
                    preds.append(pred)
                    #color = 'g' if pred == gts[tdi] else 'r'
                    #world.plot_datapoint(tdi, color=color, dir='model_acc/%i' % di)
                plotted_test_dataset = True
                #print(preds)
                accuracy = np.mean([(pred == gt) for pred, gt in zip(preds, gts)])
                #print(accuracy)
                ys.append(accuracy)
            print(max(xs))
            for xi in range(max_actions):
                all_accs_array[pi,xi] = np.interp(xi, xs, ys)
        avg_accs = np.mean(all_accs_array, axis=0)
        std_accs = np.std(all_accs_array, axis=0)
        xs = np.arange(max_actions)
        ax.plot(xs, avg_accs, color=cs[mi], label=method)
        ax.fill_between(xs, avg_accs-std_accs, avg_accs+std_accs, color=cs[mi], alpha=0.1)
    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Model Accuracy over Training Time')
    ax.legend()
    ax.set_ylim([0.43,1])
    plt.savefig('model_accuracy')
    #plt.show()
