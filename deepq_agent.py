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
                 act_space: int,
                 obs_space: int,
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
        self.actions = [i for i in range(int(act_space))]                                 #list of all possible actions
        self.gamma = gamma
        self.eps_start = eps
        self.eps_end = eps_floor
        self.eps_decay = eps_decay_rate
        self.batch_size = training_batch_size
        self.action_count = act_space
        self.play_epoch = 0                                                             #number of times the agent has played
        self.memory = ReplayMemory(pmem_buffer_size)

        network_shape = np.array([obs_space] + network_shape + [act_space])  #input layer, hidden layers, output layer
        self.policy_net = DeepQNetwork(network_shape, device).to(device)                 #updated more frequently, for short term training
        self.target_net = DeepQNetwork(network_shape, device).to(device)                 #updated less frequently, for long term training and generating 'sane' branching paths for policy_net
        
        self.target_net.load_state_dict(self.policy_net.state_dict())                   # necessary to sync the rand weights
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learn_rate, amsgrad=True)
        self.f_loss = nn.SmoothL1Loss()

    def get_action(self, state):
        #convert state to tensor
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        #decay exploration rate
        eps_threshold = max(self.eps_start - self.play_epoch * self.eps_decay, self.eps_end)
        
        #decide exploration vs exploitation
        if np.random.random() < eps_threshold:
            action = torch.tensor([[random.randrange(self.action_count)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)

        return action

    #append state, action, next_state, reward, done to memory
    def remember(self, state, action, next_state, reward):
        self.memory.save(state, action, next_state, reward)

    #TODO revisit playmemory and how memories are prioritized

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        non_final_mask = torch.tensor(tuple(
                map(
                    lambda s: s is not None, 
                    batch.next_state
                )
            ), device=self.device, dtype=torch.bool) #this is a tensor of bools that acts as a filter that only keeps non-final states
        
        non_final_next_states = \
            torch.cat([s for s in batch.next_state if s is not None]) #and the states that follow those non-final states

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.f_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.play_epoch += 1

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        

        self.play_epoch += 1

        if self.play_epoch % TARGET_UPDATE == 0 and self.play_epoch > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.save("model.pt")

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

