import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import patches

#from learning.active.train import train

def train(dataloader, model, n_epochs, loss_fn=torch.nn.functional.binary_cross_entropy, plot=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    all_losses = []
    for ex in range(n_epochs):
        epoch_losses = []
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        all_losses.append(np.mean(epoch_losses))
    if plot:
        plt.ion()
        fig, ax = plt.subplots()
        ax.plot(all_losses, label='loss')
        ax.set_xlabel('Epoch')
        ax.set_title('Loss on Training Dataset')
        ax.legend()
        plt.show()
        input('enter to close plots')
        plt.close()

class AddNet(torch.nn.Module):
    def __init__(self, hidden_states, n_layers):
        super(AddNet, self).__init__()
        modules = []
        if n_layers == 1:
            modules.append(torch.nn.Linear(2, 1))
        else:
            modules.append(torch.nn.Linear(2, hidden_states))
            for li in range(1, n_layers):
                modules.append(torch.nn.ReLU())
                if li == n_layers-1:
                    modules.append(torch.nn.Linear(hidden_states, 1))
                else:
                    modules.append(torch.nn.Linear(hidden_states, hidden_states))
            self.NN = torch.nn.Sequential(*modules)

    def forward(self, x):
        """
        :param input: network inputs. First element in list is always object_features
        """
        y = torch.sigmoid(self.NN(x))
        return y

class AddDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.all_abs = torch.tensor([])
        self.all_preds = torch.tensor([])

    def __getitem__(self, ix):
        return self.all_abs[ix], self.all_preds[ix]

    def __len__(self):
        return len(self.all_abs)

    def add_to_dataset(self, a, b):
        self.all_abs = torch.cat([self.all_abs, torch.tensor([a])])
        self.all_preds = torch.cat([self.all_preds, torch.tensor([b])])

def gen_balanced_dataset(dataset, a_range, b_range, N):
    N_pos = 0
    N_neg = 0
    while N_pos+N_neg < N:
        a = np.random.randint(*a_range)
        b = np.random.randint(*b_range)
        if b == a+1 and N_pos < N/2:
            dataset.add_to_dataset([a,b], [1])
            N_pos += 1
        elif N_neg < N/2:
            dataset.add_to_dataset([a,b], [0])
            N_neg += 1
    print('Done generating balanced dataset')

#### Parameters ####
# training rule: // b == a+1 --> 1 // b != a+1 --> 0 //
# range of a and b integer values to train on
a_train_range = [55,59]
b_train_range = [56,60]
# range of a and b integer values to test on
a_test_range = range(0, 100)
b_test_range = range(0, 100)
# Number of datapoints in train set (0/1s are balanced in dataset so make sure
# there are possible 1 labels in the train ranges you select or it will hang forever)
N_train = 100
# number of models to average over to calculate accuracy
test_runs = 100
# training parameters (I found these to work well consistently)
batch_size = 5
hidden_states = 32
n_layers = 5
n_epochs = 200
####################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # make balanced dataset
    dataset = AddDataset()
    gen_balanced_dataset(dataset, a_train_range, b_train_range, N_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train and store test predictions
    preds = np.zeros((len(b_test_range), len(a_test_range), test_runs))
    for ri in range(test_runs):
        print('Run %i' % ri)
        # make model
        model = AddNet(hidden_states, n_layers)
        # train
        train(dataloader, model, n_epochs, plot=False)
        for ai, a in enumerate(a_test_range):
            for bi, b in enumerate(b_test_range):
                preds[bi, ai, ri] = model.forward(torch.tensor([a, b], dtype=torch.float32)).squeeze()
    average_predictions = np.mean(preds, axis=2)
    std_dev_predictions = np.std(preds, axis=2)

    # plot results
    plt.ion()
    for title, plot_preds in zip(['Average', 'Standard Dev'], [average_predictions, std_dev_predictions]):
        fig, ax = plt.subplots()
        im = ax.imshow(plot_preds, origin='lower')
        plt.colorbar(im)
        ax.set_xlabel('a')
        ax.set_xticks(range(len(a_test_range)))
        ax.set_xticklabels(a_test_range)
        ax.set_ylabel('b')
        ax.set_yticks(range(len(b_test_range)))
        ax.set_yticklabels(b_test_range)
        a_wh = a_train_range[1] - a_train_range[0]
        b_wh = b_train_range[1] - b_train_range[0]
        offset = [a_train_range[0]-a_test_range[0], b_train_range[0]-b_test_range[0]]
        bottom_left = np.array(offset) - 0.5
        rect = patches.Rectangle(bottom_left, a_wh, b_wh, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        ax.set_title('%s Prediction of Learning b == a + 1 (#models=%i)\n\
                    Square Indicates Training Data Region' % (title, test_runs))

    a_test_range = range(50,65)
    b_test_range = range(50,65)
    for title, plot_preds in zip(['Average', 'Standard Dev'], [average_predictions, std_dev_predictions]):
        fig, ax = plt.subplots()
        im = ax.imshow(plot_preds[a_test_range[0]:a_test_range[-1]+1,b_test_range[0]:b_test_range[-1]+1], origin='lower')
        plt.colorbar(im)
        ax.set_xlabel('a')
        ax.set_xticks(range(len(a_test_range)))
        ax.set_xticklabels(a_test_range)
        ax.set_ylabel('b')
        ax.set_yticks(range(len(b_test_range)))
        ax.set_yticklabels(b_test_range)
        a_wh = a_train_range[1] - a_train_range[0]
        b_wh = b_train_range[1] - b_train_range[0]
        offset = [a_train_range[0]-a_test_range[0], b_train_range[0]-b_test_range[0]]
        bottom_left = np.array(offset) - 0.5
        rect = patches.Rectangle(bottom_left, a_wh, b_wh, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        ax.set_title('%s Prediction of Learning b == a + 1 (#models=%i)\n\
                    Square Indicates Training Data Region' % (title, test_runs))
    input('enter to close plots')
