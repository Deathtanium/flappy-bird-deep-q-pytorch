import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim

from replay_memory import ReplayMemory, Transition
from deepq_network import DeepQNetwork


TARGET_UPDATE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
GAMMA = 0.99
EPS = 0.2
EPS_DECAY = 0.000
EPS_END = 0.01
BATCH_SIZE = 32
PLAY_MEMORY = 1000000
LAYERS_SIZES = [256, 256]


class DQNAgent_pytorch:
    def __init__(self, 
                 device,
                 action_count,
                 input_dims,
                 training_batch_size,
                 learn_rate=LR,
                 gamma=GAMMA,
                 eps=EPS,                                                               #rate of exploration
                 eps_decay_rate=EPS_DECAY,                                                   
                 eps_floor=EPS_END,                                                       
                 network_shape=LAYERS_SIZES,
                 pmem_buffer_size=PLAY_MEMORY
                ):
        self.device = device
        self.actions = [i for i in range(action_count)]                                 #list of all possible actions
        self.gamma = gamma
        self.eps = eps
        self.eps_end = eps_floor
        self.eps_decay = eps_decay_rate
        self.batch_size = training_batch_size
        self.action_count = action_count
        self.play_epoch = 0                                                             #number of times the agent has played
        self.memory = ReplayMemory(pmem_buffer_size)

        network_shape = np.array([np.prod(input_dims)] + network_shape + [action_count])  #input layer, hidden layers, output layer
        self.policy_net = DeepQNetwork(network_shape, device).to(device)                 #updated more frequently, for short term training
        self.target_net = DeepQNetwork(network_shape, device).to(device)                 #updated less frequently, for long term training and generating 'sane' branching paths for policy_net
        
        self.target_net.load_state_dict(self.policy_net.state_dict())                   # necessary to sync the rand weights
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learn_rate)
        self.loss = nn.MSELoss()

    def getAction(self, state : np.ndarray | list):
        #convert state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        #decay exploration rate
        eps_threshold = max(self.eps - self.play_epoch * self.eps_decay, self.eps_end)
        
        #decide exploration vs exploitation
        if np.random.random() < eps_threshold:
            action = np.random.choice(self.actions)
        else:
            with torch.no_grad():
                actions = self.policy_net(state.unsqueeze(0))
                action = torch.argmax(actions).item()

        return action

    #append state, action, next_state, reward, done to memory
    def remember(self, state, action, next_state, reward, done):
        self.memory.save(state, action, next_state, reward, done)

    #TODO revisit playmemory and how memories are prioritized

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        states = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_state_action_values = rewards + self.gamma * next_state_values * (1 - dones)

        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.play_epoch += 1

        if self.play_epoch % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

