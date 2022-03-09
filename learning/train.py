import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.optim import Adam


def evaluate(loader, model, loss_fn, val_metric='f1'):
    acc = []
    losses = []

    preds = []
    labels = []
    for x, y in loader:
        if torch.cuda.is_available():
            x = xi.cuda()
            y = y.cuda()
        pred = model.forward(x).squeeze()
        if len(pred.shape) == 0: pred = pred.unsqueeze(-1)
        loss = loss_fn(pred, y)

        with torch.no_grad():
            preds += (pred > 0.5).cpu().float().numpy().tolist()
            labels += y.cpu().numpy().tolist()
        accuracy = ((pred>0.5) == y).float().mean()
        acc.append(accuracy.item())
        losses.append(loss.item())
    if val_metric == 'loss':
        score = np.mean(losses)
    else:
        score = -f1_score(labels, preds)


    return score


def train(dataloader, model, val_dataloader=None, n_epochs=20, loss_fn=F.binary_cross_entropy, early_stop=False, n_restart=0):
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
    early_stop_tol = 0.01
    final_loss_tol = 1
    max_restarts = 5
    all_accs = []
    all_losses = []
    for ex in range(n_epochs):
        #print(ex, n_epochs)
        epoch_losses = []
        #print('Epoch', ex)
        train_loss = 0
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()

            pred = model.forward(x)
            loss = loss_fn(pred, y)
            train_loss += loss
            loss.backward()
            #print('grad', [p.grad.sum() for p in list(model.parameters())])

            optimizer.step()

            # TODO: change accuracy calculation for non binary tasks
            accuracy = ((pred>0.5) == y).float().mean()

            all_accs.append(accuracy.item())
            #all_losses.append(loss.item())
            epoch_losses.append(loss.item())
            it += 1
        all_losses.append(np.mean(epoch_losses))
        if early_stop and (train_loss < early_stop_tol):
            break
        if val_dataloader is not None:
            val_loss = evaluate(val_dataloader, model, loss_fn)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                #print('Saved')

    # reinitialize model weights and train again if didn't converge
    if all_losses[-1] > final_loss_tol and n_restart < max_restarts:
        model.reset()
        print('reset weights %i'%n_restart)
        return train(dataloader,
                        model,
                        val_dataloader=val_dataloader,
                        n_epochs=n_epochs,
                        loss_fn=loss_fn,
                        early_stop=early_stop,
                        n_restart=n_restart+1)
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
