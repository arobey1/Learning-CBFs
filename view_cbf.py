import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from train_net import trial
from process_data import get_states_and_values
from torch.autograd import Variable
import torch
from scipy.io import savemat

from SeqNet import Net
from MultiLayerNet import Network
import os

INPUT_DIM = 6
OUTPUT_DIM = 1
NET_DIMS = [INPUT_DIM, 100, 50, OUTPUT_DIM]
LR = 1e-3
N_EPOCHS = 1000
LOSS = torch.nn.MSELoss
SAVE_NET = True
OPTIMIZER = torch.optim.Adam

def main():

    states, values = get_states_and_values()
    net, train_err, test_err = trial(NET_DIMS, lr=LR, loss=LOSS, n_epochs=N_EPOCHS, save_net=SAVE_NET, opt=OPTIMIZER)
    # net = torch.load(os.path.join(os.getcwd(), 'saved_models/model.pt'))
    data = Variable(torch.tensor(states).float())
    predicted_values = net(data).detach().numpy()
    plot3d(states, values, predicted_values, together=True)
    # plot_train_errors(train_err, test_err)

    savemat(os.path.join(os.getcwd(), 'data/data_for_haimin'), mdict = {
        'states': states,
        'true_values': values,
        'learned_values': predicted_values
    })


def plot3d(states:np.ndarray, values:np.ndarray, predicted_values: np.ndarray, together=False) -> None:
    """Creates a 3D scatter plot of states and values for learned and true CBF

    Args:
        states: states for CBF
        values: values of true CBF h(x)
        predicted_values: values from learned function h(x)
        together: plots true and predicted values on same plot if true

    """

    rel_x_dist = states[:,0] - states[:,3]
    rel_y_dist = states[:,1] - states[:,4]
    fig = plt.figure()

    if together is True:
        ax = fig.gca(projection='3d')
        ax.scatter3D(rel_x_dist, rel_y_dist, predicted_values, c=predicted_values, cmap='Reds', label="Learned CBF")
        ax.scatter3D(rel_x_dist, rel_y_dist, values, c=values, cmap='Blues', alpha=0.2, label='True CBF')
        ax.set_title('True and Learned CBFs')
        ax.set_xlabel('Relative x distance')
        ax.set_ylabel('Relative y distance')
        ax.legend()
    else:
        l_ax = fig.add_subplot(1, 2, 1, projection='3d')
        l_ax.scatter3D(rel_x_dist, rel_y_dist, values, c=predicted_values, cmap='Blues')
        l_ax.set_title('True CBF')
        l_ax.set_xlabel('Relative x distance')
        l_ax.set_ylabel('Relative y distance')
        r_ax = fig.add_subplot(1, 2, 2, projection='3d')
        r_ax.scatter3D(rel_x_dist, rel_y_dist, predicted_values, c=values, cmap='Reds')
        r_ax.set_title('Learned CBF')
        r_ax.set_xlabel('Relative x distance')
        r_ax.set_ylabel('Relative y distance')

    plt.show()


def plot_train_errors(train_err, test_err):
    """Short summary.

    Args:
        train_err: list of training errors at each epoch
        test_err: list of test errors at each epoch
    """

    plt.plot(range(len(train_err)), train_err, 'r', label='Training error')
    plt.plot(range(len(test_err)), test_err, 'b', label='Test error')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch number')
    plt.ylabel('MSE')
    plt.title('Training and Test Error for Learned CBF')
    plt.show()

def calc_grad(net, x):

    x = torch.tensor(x.reshape((1, 6)).tolist(), requires_grad=True)
    out = net(x)
    out.backward()
    print(x.grad)

    return x.grad


if __name__ == '__main__':
    main()
