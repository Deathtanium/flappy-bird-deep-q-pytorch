
import sys
import numpy as np

from ple.games.flappybird import FlappyBird
from ple import PLE

import torch

from DQN_utils import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stateConv(state):
  return np.array([state[0]/256.0-1,state[1]/16,state[2]/256-1,state[3]/256-1,state[4]/256-1])
  #return np.array([state[2]/320.0,state[3]/512.0,state[4]/512.0])

if __name__ == "__main__":
  
  #Game init
  reward = 0.0
  epochs = 1000000
  game = FlappyBird()
  p = PLE(game, fps=30, display_screen=True)
  p.init()
  input_dims=(len(stateConv(list(p.getGameState().values()))),)
  #note: the actionset array is [119,None]
  
  #Agent init
  agent = DQNAgent(
    device,
    len(p.getActionSet()), 
    input_dims,
    training_batch_size=64, 
    layers_sizes=[10,3],
    learn_rate=0.0001,
    eps=1.0,
    eps_decay=200,
    eps_end=0.001,
    gamma=0.99,
    play_memory=1000000
  )
  #load model
  if len(sys.argv) > 1:
    agent.load(sys.argv[1])


  #fill Q with random data
  for i in range(agent.batch_size):
    p.init()
    reward = 0
    while not p.game_over():
      state = stateConv(list(p.getGameState().values()))
      action_ind_random = np.random.randint(0,agent.action_count)
      p.act(p.getActionSet()[action_ind_random])
      newState = stateConv(list(p.getGameState().values()))
      # compute the reward
      if not p.game_over():
        reward = 1
      else:
        reward = -1000
      reward += reward
      agent.memory.save(torch.tensor(state,dtype=torch.float32,device=device), torch.tensor(action_ind_random,dtype=torch.int64,device=device), torch.tensor(newState,dtype=torch.float32,device=device), torch.tensor(reward,dtype=torch.float32,device=device))


  # the actual training process
  for i in range(epochs):
    total_reward = 0
    p.init()
    while not p.game_over():
      state = stateConv(list(p.getGameState().values()))
      action_ind = agent.getAction(torch.tensor(state, dtype = torch.float32, device=device))
      p.act(p.getActionSet()[action_ind])
      agent.playCounter+=1
      newState = stateConv(list(p.getGameState().values()))
      if not p.game_over():
        reward = 1
      else:
        reward = -1000
      total_reward += reward
      agent.memory.save(
        torch.tensor(state,dtype=torch.float32,device=device), 
        torch.tensor(action_ind,dtype=torch.int64,device=device), 
        torch.tensor(newState,dtype=torch.float32,device=device), 
        torch.tensor(reward,dtype=torch.float32,device=device))
      agent.train()

    print("Iteration: ", i, "Total reward: ", total_reward)
    agent.save("flappy_pytorch.pth")