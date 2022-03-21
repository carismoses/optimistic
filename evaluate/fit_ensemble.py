'''
need to fit the following parameters to ensure ensemble correctly estimates
prediction uncertainty
- number of models
- initial NN parameters distribution
- early stopping during training versus not early stopping
'''
from argparse import Namespace
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import torch

from experiments.utils import ExperimentLogger
from learning.utils import train_model, MLP
from learning.models.ensemble import Ensembles
from domains.tools.world import MODEL_INPUT_DIMS, ToolsWorld
from domains.tools.primitives import get_contact_gen
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn

##############################################################################
## dataset of all negative labels
#dataset_path = 'logs/experiments/negative_dataset-20220307-161124'
## dataset of balanced positive negative labels
dataset_path = 'logs/experiments/balanced_test-20220307-112257'

args = Namespace(n_hidden=48,
                n_layers=5,
                model_exp_path=dataset_path,
                n_models=20, # more models definitely improve things
                batch_size=8,
                n_epochs=500,
                early_stop=True) # early stopping seems to make std devs high far from data

#import pdb; pdb.set_trace()
##############################################################################

## visualize mean, SD, and BALD score for sample space of each contact type
## make a plot for each contact type (subplot for mean, std, and tool vis)
world = ToolsWorld(False, None, contact_types=args.contact_types)
contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

def plot_bald(model, dataset, logger):
    mean_fn = get_model_accuracy_fn(model, 'mean')
    std_fn = get_model_accuracy_fn(model, 'std')
    seq_fn = get_seq_fn(model)
    ts = time.strftime('%Y%m%d-%H%M%S')
    for type in logger.args.contact_types:
        fig, axes = plt.subplots(4, figsize=(5, 10))
        world.vis_dense_plot(type, axes[0], [-.5, .6], [-.25, .45], 0, 1, value_fn=mean_fn, cell_width=0.05)
        world.vis_dense_plot(type, axes[1], [-.5, .6], [-.25, .45], 0, 1, value_fn=std_fn, cell_width=0.05)
        world.vis_dense_plot(type, axes[2], [-.5, .6], [-.25, .45], 0, 1, value_fn=seq_fn, cell_width=0.05)
        if dataset:
            for ai in range(3):
                for x, y in dataset.datasets[type]:
                    color = 'r' if y == 0 else 'g'
                    axes[ai].plot(*x, color+'.')

        for contact in contacts:
            cont = contact[0]
            if cont.type == type:
                world.vis_tool_ax(cont, axes[3], frame='cont')
            axes[3].set_xlim([-.5, .6])
            axes[3].set_ylim([-.25, .45])

            # move over so aligned with colorbar images
            div = make_axes_locatable(axes[3])
            div.append_axes("right", size="10%", pad=0.5)

        axes[0].set_title('Mean Ensemble Predictions')
        axes[1].set_title('Std Ensemble Predictions')
        axes[2].set_title('BALD Score')

        #if type == 'poke':
        fname = 'acc_%s_%s_%i.png' % (ts, type, i)
        logger.save_figure(fname, dir='bald')
        plt.close()

## load dataset
logger = ExperimentLogger(dataset_path)
dataset, i = logger.load_trans_dataset('', ret_i=True)

## train model on dataset
base_args = {'n_in': MODEL_INPUT_DIMS,
            'n_hidden': args.n_hidden,
            'n_layers': args.n_layers}
fig, ax = plt.subplots(args.n_models*len(args.contact_types), 2)

# plot initial weight distribution
model = Ensembles(MLP, base_args, args.n_models, args.actions, args.objects)
mi = 0
for ctype in args.contact_types:
    for emodel in model.ensembles[ctype].models:
        all_weights = []
        for pi, p in enumerate(emodel.parameters()):
            all_weights.append(torch.flatten(p))
        ax[mi,0].hist(torch.cat(all_weights).detach().numpy(), bins=20)
        mi += 1

# plot initial predictive distribution
plot_bald(model, None, logger)

# train model and plot training losses
all_losses = train_model(model, dataset, args, types=args.contact_types)
mi = 0
for ctype in args.contact_types:
    for losses in all_losses[ctype]:
        print(losses[-1])
        ax[mi,1].plot(losses)
        mi += 1
#plt.show()
plt.close()

## save model and accuracy plots
logger.save_trans_model(model, i=i)

# plot final model predictions
plot_bald(model, dataset, logger)
