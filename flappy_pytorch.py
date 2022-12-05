import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from ple.games.flappybird import FlappyBird
from ple import PLE


#have everything run on the GPU if possible
# obiect pentru stocat datele de antrenare
class ReplayMemory():
  def __init__(self, mem_size, input_dims):
    
    self.mem_size = mem_size
    self.counter = 0
    # initializez empty arrays of size capacity
    # fiecare state va contine cele 8 valori obtinute de la GetGameState()
    self.state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)     #unpack input_dims
    self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
    self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
    self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
    self.finisher_mem = np.zeros(self.mem_size, dtype=np.int32)

  def add(self, state, action, reward, new_state, done):
    # dupa ce e full, rescriem de la capat
    index = self.counter % self.mem_size
    
    self.state_mem[index] = state
    self.action_mem[index] = action
    self.reward_mem[index] = reward
    self.new_state_mem[index] = new_state
    self.finisher_mem[index] = 1-int(done)
    self.counter += 1

  def sample(self, sample_size):
    #face sample random din ce avem pana acuma
    max_mem = min(self.counter, self.mem_size)
    #^^^ daca counter a trecut de batch_size, putem folosi intreg batch-ul, inclusiv datele ramase
    if sample_size > self.counter:
      batch_indices = np.random.choice(self.counter, sample_size, replace=True) #batch_indices este un array de indici
    else:
      batch_indices = np.random.choice(max_mem, sample_size, replace=False)
    states = self.state_mem[batch_indices]
    new_states = self.new_state_mem[batch_indices]
    actions = self.action_mem[batch_indices]
    rewards = self.reward_mem[batch_indices]
    finishers = self.finisher_mem[batch_indices]
    return states, actions, rewards, new_states, finishers

class FlappyNet(nn.Module):
  def __init__(self, input_size, output_size, layer_sizes):
    self.device = torch.device("cpu")
    super(FlappyNet, self).__init__()
    self.fc_arr = nn.ParameterList()
    self.fc_arr.append(nn.Linear(input_size, layer_sizes[0]))
    for i in range(1,len(layer_sizes)):
      self.fc_arr.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
    self.fc_arr.append(nn.Linear(layer_sizes[-1], output_size))
  
  def forward(self, x:torch.Tensor):
    for i in range(len(self.fc_arr)):
      x = F.relu(self.fc_arr[i](x))
    #x = F.softmax(self.fc_arr[-1](x), dim=1)
    return x

class FlappyAgent:
  def __init__(self, 
        action_count,        #nr de actiuni posibile
        input_dims,       #tuplu de dimensiuni, folosit pentru layer-ul de input
        training_batch_size,       #dimensiunea batch-ului folosit la antrenare
        learn_rate=0.001, #learn_rate, altfel spus, alfa
        gamma=0.99,       #discount-ul pentru Q-learning
        eps=0.2,          #parametrul epsilon, pentru balansarea explorare vs exploatare
        eps_step=0.000,  #parametrul epsilon o sa scada la fiecare iteratie cu acest step, crescand exploatarea pe masura ce reteaua converge la o solutie
        eps_end=0.01,      #minimul pentru epsilon; ii lasam totusi o mica sansa de explorare
        layers_sizes=[256, 256], #dimensiunile layerelor; l-am facut dinamic pentru ca poate voi folosi agentul asta si pentru alte jocuri ;) -Serban
        play_memory=1000000
      ):
    self.actions = [i for i in range(action_count)]
    self.gamma = gamma
    self.eps = eps
    self.eps_min = eps_end
    self.eps_step = eps_step
    self.batch_size = training_batch_size
    self.action_count = action_count
    self.memory = ReplayMemory(play_memory, input_dims)
    layers = []
    layers.append(nn.Flatten())
    layers.append(nn.Linear(input_dims[0], layers_sizes[0]))
    for i in range(1,len(layers_sizes)):
      layers.append(nn.ReLU())
      layers.append(nn.Linear(layers_sizes[i-1], layers_sizes[i]))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(layers_sizes[-1], action_count))
    self.nnet = nn.Sequential(*layers)
    self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=learn_rate)
    self.loss = nn.MSELoss()
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.nnet.to(self.device)
    self.loss.to(self.device)
    self.nnet.eval()

  def getAction(self, state): #state is an array or np array
    if np.random.random() < self.eps:
      action= np.random.choice(self.actions)
    else:
      state = torch.tensor(state, dtype=torch.float32,device=self.device)
      state = state.unsqueeze(0)
      tmp = self.nnet(state)
      action = torch.argmax(tmp).item()
    return action

  def learn(self):
    if self.memory.counter < self.batch_size:
      return

    self.nnet.train()
    self.optimizer.zero_grad()
    
    states, actions, rewards, new_states, finishers = self.memory.sample(self.batch_size)
    
    states = torch.tensor(states, dtype=torch.float32,device=self.device)
    new_states = torch.tensor(new_states, dtype=torch.float32,device=self.device)
    actions = torch.tensor(actions, dtype=torch.int64,device=self.device)
    rewards = torch.tensor(rewards, dtype=torch.float32,device=self.device)
    finishers = torch.tensor(finishers, dtype=torch.float32,device=self.device)
    
    q_pred = self.nnet(states)
    q_pred = torch.gather(q_pred, 1, actions.unsqueeze(1)).squeeze(1)
    q_next = self.nnet(new_states).max(dim=1)[0]
    q_target = rewards + self.gamma*q_next*finishers
    loss = self.loss(q_pred, q_target)
    loss.backward()
    self.optimizer.step()
    self.nnet.eval()

    self.eps = self.eps - self.eps_step if self.eps > self.eps_min else self.eps_min

  def saveModel(self):
    torch.save(self.nnet.state_dict(), 'flappy_model.pth')

  def loadModel(self):
    self.nnet.load_state_dict(torch.load('flappy_model.pth'))

def stateConv(state):
  return np.array(state).flatten()
  #return np.array([state[2]/320.0,state[3]/512.0,state[4]/512.0])

if __name__ == "__main__":
  reward = 0.0
  epochs = 1000000

  game = FlappyBird()
  p = PLE(game, fps=30, display_screen=True)
  p.init()
  #note: the actionset array is [119,None]

  input_dims=(len(stateConv(list(p.getGameState().values()))),)
  agent = FlappyAgent(len(p.getActionSet()), input_dims, 100, 
    layers_sizes=[1024,1024],
    learn_rate=0.00001,
    eps=0.8,
    eps_step=0.0001,
    eps_end=0.001,
    gamma=0.99,
    play_memory=1000000
  ) #go to init() to change inner layer sizes

  if len(sys.argv) > 1:
    agent.loadModel()

  #fill the first batch with random data
  for i in range(agent.batch_size):
    p.init()
    reward = 0
    while not p.game_over():
      state = stateConv(np.array(list(p.getGameState().values())))
      act_ind = np.random.randint(0,agent.action_count)
      action = p.getActionSet()[act_ind]
      p.act(action)
      newState = stateConv(np.array(list(p.getGameState().values())))
      # compute the reward
      if not p.game_over():
        reward = 1
      else:
        reward = -1000
      reward += reward
      agent.memory.add(state, act_ind, reward, newState, p.game_over())

  # the actual training process
  for i in range(epochs):
    total_reward = 0
    p.init()
    while not p.game_over():
      state = stateConv(np.array(list(p.getGameState().values())))
      action = agent.getAction(state)
      p.act(p.getActionSet()[action])
      newState = stateConv(np.array(list(p.getGameState().values())))
      if not p.game_over():
        reward = 1
      else:
        reward = -1000
      total_reward += reward
      agent.memory.add(state, action, reward, newState, p.game_over())
      agent.learn()

    print("Iteration: ", i, "Total reward: ", total_reward)
    agent.saveModel()