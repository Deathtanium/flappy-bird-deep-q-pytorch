import random
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim

import datetime

from replay_memory import ReplayMemory, Transition
from deepq_network import DQN


TARGET_UPDATE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
GAMMA = 0.99
EPS = 0.2
EPS_DECAY = 0.000
EPS_END = 0.01
BATCH_SIZE = 32
PLAY_MEMORY = 1000000
TAU = 0.005
LAYERS_SIZES = [256, 256]


class DQNAgent_pytorch:
    def __init__(self, 
                 device,
                 act_space: int,
                 obs_space: int,
                 training_batch_size,
                 learn_rate,
                 gamma,
                 eps_start,                                                               #rate of exploration
                 eps_decay_rate,                                                   
                 eps_floor,          
                 tau,                       #rate of target network update                                        
                 network_shape,
                 pmem_buffer_size
                ):
        self.device = device
        self.actions = [i for i in range(int(act_space))]                                 #list of all possible actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_floor
        self.eps_decay = eps_decay_rate
        self.batch_size = training_batch_size
        self.action_count = act_space
        self.play_epoch = 0                                                             #number of times the agent has played
        self.tau = tau
        self.memory = ReplayMemory(pmem_buffer_size)

        network_shape = np.array([obs_space] + network_shape + [act_space])  #input layer, hidden layers, output layer
        self.policy_net = DQN(network_shape, device).to(device)                 #updated more frequently, for short term training
        self.target_net = DQN(network_shape, device).to(device)                 #updated less frequently, for long term training and generating 'sane' branching paths for policy_net
        
        self.target_net.load_state_dict(self.policy_net.state_dict())                   # necessary to sync the rand weights
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learn_rate, amsgrad=True)
        #self.f_loss = nn.SmoothL1Loss()
        #self.f_loss = nn.MSELoss()
        self.f_loss = nn.SmoothL1Loss

    def get_action(self, state, action_space):
        self.play_epoch += 1

        #decay exploration rate
        #eps_threshold = max(self.eps_start - self.play_epoch * self.eps_decay, self.eps_end)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.play_epoch / self.eps_decay)
        
        #decide exploration vs exploitation
        if np.random.random() < eps_threshold:
            action = torch.tensor([[action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(state).max(1).indices.view(1, 1)

        return action

    #append state, action, next_state, reward, done to memory
    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    #TODO revisit playmemory and how memories are prioritized

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        done_batch = torch.cat(batch.done)

        non_final_mask = ~done_batch
        non_final_next_states = torch.cat([s for s, done in zip(batch.next_state, batch.done) if not done])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values:torch.Tensor = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()#self.f_loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

