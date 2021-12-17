import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

from torch.nn import functional as F
from torch.optim import Adam

def train(dataloader, model, val_dataloader=None, n_epochs=20, loss_fn=F.binary_cross_entropy):
    """
    :param val_dataloader: If a validation set is given, will return the model
    with the lowest validation loss.
    """
    optimizer = Adam(model.parameters(), lr=1e-3)
    if torch.cuda.is_available():
        model.cuda()

    best_loss = 1000
    best_weights = None
    it = 0
    all_accs = []
    all_losses = []
    for ex in range(n_epochs):
        epoch_losses = []
        #print('Epoch', ex)
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = [xi.cuda() for xi in x]
                y = y.cuda()
            optimizer.zero_grad()

            pred = model.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            #print('grad', [p.grad.sum() for p in list(model.parameters())])

            optimizer.step()

            # TODO: change accuracy calculation for non binary tasks
            accuracy = ((pred>0.5) == y).float().mean()

            all_accs.append(accuracy.item())
            #all_losses.append(loss.item())
            epoch_losses.append(loss.item())
            it += 1
        if val_dataloader is not None:
            val_loss = evaluate(val_dataloader, model)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                #print('Saved')
        all_losses.append(np.mean(epoch_losses))
    if val_dataloader is not None:
        model.load_state_dict(best_weights)

    '''
    fig, ax = plt.subplots()
    #ax.plot(all_accs, label='accuracy')
    ax.plot(all_losses, label='loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Loss on Training Dataset')
    ax.legend()
    plt.show()
    '''

    return all_losses
