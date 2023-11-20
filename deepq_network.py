import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class DeepQNetwork(nn.Module):
    def __init__(self, layer_sizes, device):
        super(DeepQNetwork, self).__init__()
        self.shape = layer_sizes
        self.device = device
        self.seq = nn.Sequential()
        self.seq.add_module('fc1', nn.Linear(layer_sizes[0], layer_sizes[1]))
        for i in range(2, len(layer_sizes)):
            self.seq.add_module('relu' + str(i), nn.ReLU())                                 #relu
            self.seq.add_module('fc' + str(i), nn.Linear(layer_sizes[i-1], layer_sizes[i])) #fully connected

    def forward(self, x):
        return self.seq(x)