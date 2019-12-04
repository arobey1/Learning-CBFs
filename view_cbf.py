import matplotlib.pyplot as plt
import numpy as np
from train_net import trial
from process_data import get_states_and_values
from torch.autograd import Variable
import torch

def main():

    states, values = get_states_and_values()
    net, errors = trial()

    plt.figure()
    plt.plot(range(len(errors)), errors, label='Training error')
    plt.legend()

    data = Variable(torch.tensor(states).float())
    predicted_values = net(data).detach().numpy()

    idx = 3
    plt.figure()
    plt.scatter(states[:,idx], values, color='blue')
    plt.scatter(states[:,idx], predicted_values, color='red')
    plt.show()


if __name__ == '__main__':
    main()
