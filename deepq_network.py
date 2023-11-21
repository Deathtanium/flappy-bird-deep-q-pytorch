import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):
    def __init__(self, layer_sizes, device):
        super(DQN, self).__init__()
        self.layer_sizes = layer_sizes
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    
        

"""
DQN that implements a convolutional neural network for playing Flappy Bird with only 11 inputs as floats and 2 outputs as floats
"""
class DQNSpecial(nn.Module):
    def __init__(self,device):
        super(DQNSpecial, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

    def conv2d_size_out(size, kernel_size = 5, stride = 2):
        return (size - (kernel_size - 1) - 1) // stride  + 1
    def forward(self, x):
        x = x.to(self.device)
        x = x.view(-1, 4, 84, 84)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))        
        return x.view(x.size(0), -1)
    
