import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from MultiLayerNet import Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from process_data import get_states_and_values
import os


def trial(net_dims, lr=1e-3, loss=nn.MSELoss, n_epochs=1000, save_net=False, opt=torch.optim.Adam):
    """Short summary.

    Args:
        net_dims: Dimensions of each layer in neural network.
        lr: Learning rate to be used in training.
        loss: pytorch loss function to use for training.
        n_epochs: Number of epochs to train the model.
        save_net: Saves pytorch model parameters if True.

    Returns:
        type: Description of returned object.

    """

    states, values = get_states_and_values()
    train_X, test_X, train_y, test_y = split_dataset(states, values)

    net = Network(net_dims)
    criterion = loss()
    optimizer = opt(net.parameters(), lr=lr)

    train_err, test_err = train_network(train_X, train_y, test_X, test_y, net, optimizer, criterion, n_epochs=n_epochs)

    if save_net is True:
        torch.save(net, os.path.join(os.getcwd(), 'saved_models/model.pt'))

    return net, train_err, test_err


def split_dataset(states: np.ndarray, values: np.ndarray):
    """Create training/test split and store in torch Variables

    params:
        states - states of CBF
        values - values of CBF
    """

    train_X, test_X, train_y, test_y = train_test_split(states, values, test_size=0.2)

    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).float())
    test_y = Variable(torch.Tensor(test_y).float())

    return train_X, test_X, train_y, test_y

def train_network(train_X: torch.tensor, train_y: torch.tensor, test_X: torch.tensor,
    test_y: torch.tensor, net, optimizer, criterion, n_epochs=1000):
    """Short summary.

    Args:
        train_X: Description of parameter `train_X`.
        train_y: Description of parameter `train_y`.
        test_X: Description of parameter `test_X`.
        test_y: Description of parameter `test_y`.
        net: Description of parameter `net`.
        optimizer: Description of parameter `optimizer`.
        criterion: Description of parameter `criterion`.
        n_epochs: number of epochs for training

    Returns:
        type: Description of returned object.

    """

    train_errors, test_errors = [], []
    for epoch in tqdm(range(n_epochs)):

        train_err = step(train_X, train_y, net, criterion, opt=optimizer, train=True)
        train_errors.append(train_err)
        test_err = step(test_X, test_y, net, criterion, train=False)
        test_errors.append(test_err)

    return train_errors, test_errors


def step(train_X: torch.tensor, train_y: torch.tensor, net, criterion, opt=None, train=True):
    """Train neural network on Iris dataset

    params:
        train_X: training instances
        train_y: training labels
        net: torch nn.Module - neural network model
        optimizer: nn.optim - optimizer for training network
        criterion: nn.CrossEntropyLoss - loss function
    """

    out = net(train_X)
    loss = criterion(out, train_y)

    if train is True:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.detach().numpy()


if __name__ == '__main__':
    trial()
