import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.train import train
from learning.models.mlp import MLP
from experiments.utils import ExperimentLogger
from domains.tools.world import CONTACT_TYPES, N_MC_IN

'''
#### Parameters ####
# Calculate the model model accuracy on a given dataset

# dataset used to calculate accuracy
#exp_path = 'logs/experiments/test-tools-20211112-180443'
test_dataset_path = 'logs/experiments/90_random_goals_balanced-20220219-170059'
#test_dataset_index = 99
n_models = 7
nn_depths = [2, 5, 8]
n_epochs = 500
hiddens = [16, 32, 48]
batch_sizes = [4, 8, 16]#, 32, 64]
########

test_dataset_logger = ExperimentLogger(test_dataset_path)
test_dataset = test_dataset_logger.load_trans_dataset('')

all_losses = {}
for type in CONTACT_TYPES:
    print('Evaluating %s models.' % type)
    all_losses[type] = {}
    for nn_depth in nn_depths:
        for n_hidden in hiddens:
            for batch_size in batch_sizes:
                all_param_losses = []
                for mi in range(n_models):
                    dataset = test_dataset.datasets[type]
                    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    model = MLP(n_in=N_MC_IN,
                                n_hidden=n_hidden,
                                n_layers=nn_depth)
                    print('Training model.')
                    all_param_losses.append(train(test_dataloader,
                                                    model,
                                                    n_epochs=n_epochs,
                                                    loss_fn=F.binary_cross_entropy))
                all_losses[type][(nn_depth, n_hidden, batch_size)] = np.array(all_param_losses)
'''
import pickle
#with open('losses.pkl', 'wb') as handle:
#    pickle.dump(all_losses, handle)

with open('losses.pkl', 'rb') as handle:
    all_losses = pickle.load(handle)

#import pdb; pdb.set_trace()
pink_rgb = np.array([255,192,203])/255
orange_rgb = np.array([255, 165, 0])/255
cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k', pink_rgb, orange_rgb]
max_lines_per_plot = 3#len(cs)
for type in ['poke']:#CONTACT_TYPES:
    ncs = len(cs)
    for i, ((nn_depth, n_hidden, batch_size), losses) in enumerate(all_losses[type].items()):
        if not i % max_lines_per_plot:
            fig, ax = plt.subplots(figsize=(8,5))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            ax.set_ylabel('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_title('Training Loss, Contact Type = %s' % type)

        if nn_depth != 2:
            avg = np.mean(losses, axis=0)
            std = np.std(losses, axis=0)
            xs = np.arange(losses.shape[1])
            ax.plot(xs[:150], avg[:150], color=cs[i%ncs], label='(%i,%i,%i)'%(nn_depth,n_hidden,batch_size))
            ax.fill_between(xs[:150], (avg-std)[:150], (avg+std)[:150], color=cs[i%ncs], alpha=0.1)

            # Put a legend below current axis
            ax.legend(loc='upper center',
                        title='(# layers, # hidden units, batch size)',
                        bbox_to_anchor=(0.5, -0.11),
                        ncol=max_lines_per_plot)

plt.show()

'''
accuracies = []
for mi in itertools.count():
    try:
        model = model_logger.load_trans_model(world, i=mi)
    except:
        break
    count = 0
    avg_pred = 0
    for x, y in test_dataset:
        model_pred = np.round(model_forward(model, x).squeeze())
        avg_pred += model_pred
        if model_pred == y: count += 1
        #print(x[0][0][1])
    acc = count/len(test_dataset)
    accuracies.append(acc)
    print('Accuracy for model %i: %f' % (mi, acc))
    print('Average prediction for model %i: %f' % (mi, avg_pred/len(test_dataset)))
world.disconnect()
plt.plot(accuracies)
plt.show()
'''
