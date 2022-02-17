import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.train import train
from learning.models.gnn import TransitionGNN
from learning.models.mlp import MLP
from experiments.utils import ExperimentLogger
from learning.datasets import model_forward
from domains.tools.world import ToolsWorld


#### Parameters ####
# Calculate the model model accuracy on a given dataset

# dataset used to calculate accuracy
#exp_path = 'logs/experiments/test-tools-20211112-180443'
test_dataset_path = 'logs/experiments/test-tools-20211112-180443'
test_dataset_index = 99
n_models = 7
nn_depths = [5]
n_epochs = 500
hiddens = [32]
batch_sizes = [16, 32, 64]
########
test_dataset_logger = ExperimentLogger(test_dataset_path)
test_dataset = test_dataset_logger.load_trans_dataset(i=test_dataset_index)
world, _, _ = ToolsWorld.init(None, 'optimistic', False, None)

all_losses = {}
for nn_depth in nn_depths:
    for n_hidden in hiddens:
        for batch_size in batch_sizes:
            all_param_losses = []
            for mi in range(n_models):
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                model = MLP(n_of_in=world.n_of_in,
                                        n_ef_in=world.n_ef_in,
                                        n_af_in=world.n_af_in,
                                        n_hidden=n_hidden,
                                        n_layers=nn_depth)
                print('Training model.')
                all_param_losses.append(train(test_dataloader,
                                                model,
                                                n_epochs=n_epochs,
                                                loss_fn=F.binary_cross_entropy))
            all_losses[(nn_depth, n_hidden, batch_size)] = np.array(all_param_losses)

#import pdb; pdb.set_trace()
fig, ax = plt.subplots()
cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
ncs = len(cs)
for i, ((nn_depth, n_hidden, batch_size), losses) in enumerate(all_losses.items()):
    avg = np.mean(losses, axis=0)
    std = np.std(losses, axis=0)
    xs = np.arange(losses.shape[1])
    ax.plot(xs, avg, color=cs[i%ncs], label='(%i,%i,%i)'%(nn_depth,n_hidden,batch_size))
    ax.fill_between(xs, avg-std, avg+std, color=cs[i%ncs], alpha=0.1)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center',
            title='(# layers, # hidden units, batch size)',
            bbox_to_anchor=(0.5, -0.11),
            ncol=len(all_losses))

ax.set_ylabel('Training Loss')
ax.set_xlabel('Epoch')
ax.set_title('Training Loss')
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
