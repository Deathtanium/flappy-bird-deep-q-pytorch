import random
import math

from collections import deque,namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([],maxlen=capacity)

  def save(self, *args):
    """('state', 'action', 'next_state', 'reward')"""
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

class DeepQNet(nn.Module):
  def __init__(self, layer_sizes, device):
    super(DeepQNet, self).__init__()
    self.device = device
    self.fc_arr = nn.ParameterList().to(device)
    for i in range(1,len(layer_sizes)):
      self.fc_arr.append(nn.Linear(layer_sizes[i-1], layer_sizes[i],device=device))

  def forward(self, x:torch.Tensor):
    for i in range(len(self.fc_arr)-1):
      x = F.relu(self.fc_arr[i](x))   #NOTE: Could use something other than relu for the last layer
    x= self.fc_arr[-1](x)
    return x

class DQNAgent:
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
    self.q_net = DeepQNet([math.prod(input_dims)] + layers_sizes + [action_count], device)
    #self.target_net = DeepQNet([input_dims] + layers_sizes + [action_count])
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learn_rate, weight_decay=0.0001)
    self.loss = nn.MSELoss()
    self.playCounter = 0

  def getAction(self, state:torch.Tensor):
    eps_threshold = self.eps_end + (self.eps - self.eps_end) * math.exp(-1. * self.playCounter / self.eps_decay)
    if random.random() > eps_threshold:
      with torch.no_grad():
        return self.q_net(state).argmax().item()
    else:
      return random.choice(self.actions)

  def train(self):
    if len(self.memory) < self.batch_size:
      return
    batch = self.memory.sample(self.batch_size)
    batch = Transition(*zip(*batch))
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(list(batch.action))
    next_state_batch = torch.stack(batch.next_state)
    reward_batch = torch.stack(list(batch.reward))

    q_eval = self.q_net(state_batch)
    q_next = self.q_net(next_state_batch)
    q_target = q_eval.clone()
    q_target[range(self.batch_size), action_batch] = reward_batch + self.gamma * torch.max(q_next, 1)[0] 

    loss:nn.modules.loss = self.loss(q_eval, q_target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def save(self, path):
    torch.save(self.q_net.state_dict(), path)

  def load(self, path):
    self.q_net.load_state_dict(torch.load(path))







