import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class DeepQNetwork(nn.Module):
    def __init__(self, layer_sizes, device):
        super(DeepQNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.to(device)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    
        