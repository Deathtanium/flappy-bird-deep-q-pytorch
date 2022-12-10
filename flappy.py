
import sys
import math
import numpy as np
import random

from ple.games.flappybird import FlappyBird
from ple import PLE

import torch

from DQN_utils import *

import tensorflow as tf

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
tf.compat.v1.disable_eager_execution()


def stateConv(state):
  return state
  #return np.array([state[1]/16,state[2]/256-1,abs((state[3]+state[4])/2-state[0])/256])
  #return np.array([state[0]/256.0-1,state[1]/16,state[2]/256-1,state[3]/256-1,state[4]/256-1,state[5]/256-1,state[6]/256-1,state[7]/256-1])
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
  agent = DQNAgent_tensorflow(
    device,
    len(p.getActionSet()), 
    input_dims,
    training_batch_size=100, 
    layers_sizes=[32,32],
    learn_rate=0.001,
    eps=1.0,
    eps_decay=0.0001, #0.001 for regular decrease, 200 for log decrease
    eps_end=0.1,
    gamma=0.99,
    play_memory=1000000
  )
  #load model
  if len(sys.argv) > 1:
    agent.eps = 0.3
    agent.eps_end = 0.1
    agent.load(sys.argv[1])

  # the actual training process
  for i in range(epochs):
    total_reward = 0
    p.init()
    #print("eps: ",agent.eps_end + (agent.eps - agent.eps_end) * math.exp(-1. * agent.playCounter / agent.eps_decay))
    framesSinceLastJump = 10000
    while not p.game_over():
      agent.playCounter+=1
      framesSinceLastJump += 1
      state = stateConv(list(p.getGameState().values()))
      action_ind = agent.getAction(state)
      #action_ind = agent.getAction(torch.tensor(state, dtype = torch.float32, device=device))
      if action_ind == 0:
        """if framesSinceLastJump < 3:
          action_ind = 1
        else:"""
        framesSinceLastJump = 0
      p.act(p.getActionSet()[action_ind])
      if not p.game_over():
        reward = 1
      else:
        reward = -1000
      newState = stateConv(list(p.getGameState().values()))
      agent.remember(state,action_ind,newState,reward,int(p.game_over()))
      agent.train()
      total_reward += reward

    print("Iteration: ", i, "Total reward: ", total_reward)
    #agent.save("model.h5")