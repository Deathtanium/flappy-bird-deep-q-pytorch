
import sys
import math
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
    layers_sizes=[64,64],
    learn_rate=0.001,
    eps=1.0,
    eps_decay=200,
    eps_end=0,
    gamma=0.99,
    play_memory=1000000
  )
  #load model
  if len(sys.argv) > 1:
    agent.load(sys.argv[1])

  # the actual training process
  for i in range(epochs):
    total_reward = 0
    p.init()
    print("eps: ",agent.eps_end + (agent.eps - agent.eps_end) * math.exp(-1. * agent.playCounter / agent.eps_decay))
    agent.playCounter+=1
    framesSinceLastJump = 10000
    while not p.game_over():
      framesSinceLastJump += 1
      state = stateConv(list(p.getGameState().values()))
      action_ind = agent.getAction(torch.tensor(state, dtype = torch.float32, device=device))
      """if action_ind == 0:
        if framesSinceLastJump > 10:
          framesSinceLastJump = 0
        else:
          action_ind = 1"""
      p.act(p.getActionSet()[action_ind])
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