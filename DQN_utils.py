import random
import math

from collections import deque,namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam

import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([],maxlen=capacity)

  def save(self, *args):
    """('state', 'action', 'next_state', 'reward', 'done')"""
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

class DQNAgent_tensorflow:
  def __init__(self, 
        device,
        action_count,        #nr de actiuni posibile
        input_dims,       #tuplu de dimensiuni, folosit pentru layer-ul de input
        training_batch_size,       #dimensiunea batch-ului folosit la antrenare
        learn_rate=0.001, #learn_rate, altfel spus, alfa
        gamma=0.99,       #discount-ul pentru Q-learning
        eps=0.2,          #parametrul epsilon, pentru balansarea explorare vs exploatare
        eps_decay=0.000,  #parametrul epsilon o sa scada la fiecare iteratie cu acest step, crescand exploatarea pe masura ce reteaua converge la o solutie
        eps_end=0.01,      #minimul pentru epsilon; ii lasam totusi o mica sansa de explorare
        layers_sizes=[256, 256], #dimensiunile layerelor; l-am facut dinamic pentru ca poate voi folosi agentul asta si pentru alte jocuri ;) -Serban
        play_memory=1000000
      ):
    self.device:torch.device = device
    self.actions = [i for i in range(action_count)]
    self.gamma = gamma
    self.eps = eps
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.batch_size = training_batch_size
    self.action_count = action_count
    self.playCounter = 0
    self.memory = ReplayMemory(play_memory)
    layers_sizes = [math.prod(input_dims)] + layers_sizes + [action_count]

    self.nnet = tf.keras.Sequential()
    self.nnet.add(tf.keras.layers.Dense(layers_sizes[1], activation='relu', input_dim=(layers_sizes[0])))
    for i in range(2,len(layers_sizes)-1):
      self.nnet.add(tf.keras.layers.Dense(layers_sizes[i], activation='relu'))
    self.nnet.add(tf.keras.layers.Dense(layers_sizes[-1], activation=None))
    self.nnet.compile(optimizer=Adam(learning_rate=learn_rate), loss='mean_squared_error')
  
  def getAction(self, state):
    #eps_threshold = self.eps_end + (self.eps - self.eps_end) * math.exp(-1. * self.playCounter / self.eps_decay)
    eps_threshold = max(self.eps-self.playCounter*self.eps_decay, self.eps_end)
    if np.random.random() < eps_threshold:
      action = np.random.choice(self.actions)
    else:
      state = np.array([state])
      actions = self.nnet.predict(state)
      action = np.argmax(actions)
    return action

  def remember(self, state, action, next_state, reward, done):
    self.memory.save(state, action, next_state, reward, done)

  def train(self):
    if len(self.memory) < self.batch_size:
      return

    batch = self.memory.sample(self.batch_size)
    batch = Transition(*zip(*batch))
    states = np.array(batch.state)
    actions = np.array(batch.action)
    rewards = np.array(batch.reward)
    next_states = np.array(batch.next_state)
    dones = np.array(batch.done)
    
    q_eval = self.nnet.predict(states)
    q_next = self.nnet.predict(next_states)
    q_target = np.copy(q_eval)
    batch_indexes = np.arange(self.batch_size, dtype=np.int32)

    q_target[batch_indexes, actions] = rewards + self.gamma * np.max(q_next, axis=1) * (1-dones)

    self.nnet.train_on_batch(states,q_target)

  def save(self, path):
    self.nnet.save(path)
    
  def load(self, path):
    self.brain = tf.keras.models.load_model(path)


class DeepQNet_Torch(nn.Module):
  def __init__(self, layer_sizes, device):
    super(DeepQNet_Torch, self).__init__()
    self.device = device
    self.seq = nn.Sequential()
    self.seq.add_module('fc1', nn.Linear(layer_sizes[0], layer_sizes[1],device=device))
    for i in range(2,len(layer_sizes)):
      #add a Relu
      self.seq.add_module('relu'+str(i), nn.ReLU(device=device))
      self.seq.add_module('fc'+str(i), nn.Linear(layer_sizes[i-1], layer_sizes[i],device=device))

  def forward(self, x:torch.Tensor):
    return self.seq(x)

class DQNAgent_torch:
  def __init__(self, 
        device,
        action_count,        #nr de actiuni posibile
        input_dims,       #tuplu de dimensiuni, folosit pentru layer-ul de input
        training_batch_size,       #dimensiunea batch-ului folosit la antrenare
        learn_rate=0.001, #learn_rate, altfel spus, alfa
        gamma=0.99,       #discount-ul pentru Q-learning
        eps=0.2,          #parametrul epsilon, pentru balansarea explorare vs exploatare
        eps_decay=0.000,  #parametrul epsilon o sa scada la fiecare iteratie cu acest step, crescand exploatarea pe masura ce reteaua converge la o solutie
        eps_end=0.01,      #minimul pentru epsilon; ii lasam totusi o mica sansa de explorare
        layers_sizes=[256, 256], #dimensiunile layerelor; l-am facut dinamic pentru ca poate voi folosi agentul asta si pentru alte jocuri ;) -Serban
        play_memory=1000000
      ):
    self.device:torch.device = device
    self.actions = [i for i in range(action_count)]
    self.gamma = gamma
    self.eps = eps
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.batch_size = training_batch_size
    self.action_count = action_count
    self.memory = ReplayMemory(play_memory)
    self.policy_net = DeepQNet_Torch([math.prod(input_dims)] + layers_sizes + [action_count], device)
    self.target_net = DeepQNet_Torch([math.prod(input_dims)] + layers_sizes + [action_count], device) 
    self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learn_rate, amsgrad=True)
    self.loss = nn.MSELoss()
    self.playCounter = 0

  def getAction(self, state:np.ndarray):
    state = torch.tensor(state, dtype=torch.float32, device=self.device)
    #eps_threshold = self.eps_end + (self.eps - self.eps_end) * math.exp(-1. * self.playCounter / self.eps_decay)
    eps_threshold = max(self.eps-self.playCounter*self.eps_decay, self.eps_end)
    if random.random() < eps_threshold:
      with torch.no_grad():
        print(state)
        exit()
        #TODO: remove this exit and change the return accoringly
        return self.policy_net(state).argmax().item()
    else:
      return random.choice(self.actions)

  def remember(self, state, action, next_state, reward, done):
    self.memory.save(
      torch.tensor(state,dtype=torch.float32,device=self.device), 
      torch.tensor(action,dtype=torch.int64,device=self.device), 
      torch.tensor(next_state,dtype=torch.float32,device=self.device), 
      torch.tensor(reward,dtype=torch.float32,device=self.device),
      torch.tensor(done,dtype=torch.int64,device=self.device))

  def train(self):
    if len(self.memory) < self.batch_size:
      return
    batch = self.memory.sample(self.batch_size)
    batch = Transition(*zip(*batch))
    states = torch.stack(batch.state)
    actions = torch.stack(list(batch.action))
    next_states = torch.stack(batch.next_state)
    rewards = torch.stack(list(batch.reward))
    dones = torch.stack(list(batch.done))

    non_final_mask = torch.tensor(tuple(map(lambda s: s==1, batch.done)), device=self.device, dtype=torch.bool)

    non_final_next_states = torch.stack([s for s, done in zip(batch.next_state, batch.done) if done == 0])

    state_action_values = self.policy_net(states)
    next_state_action_values = self.policy_net(next_states)
    q_target = state_action_values.clone()
    batch_indexes = np.arange(self.batch_size, dtype=np.int32)
    q_target[batch_indexes, actions] = rewards + self.gamma * torch.max(next_state_action_values, 1)[0] * (1-dones)

    loss = self.loss(state_action_values, q_target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def save(self, path):
    torch.save(self.policy_net.state_dict(), path)

  def load(self, path):
    self.policy_net.load_state_dict(torch.load(path))



