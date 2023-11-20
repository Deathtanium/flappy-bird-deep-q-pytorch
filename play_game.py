import torch 
import numpy as np

import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human")

from deepq_agent import DQNAgent_pytorch


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

#get size of observation space
obs_space = np.prod(env.observation_space.shape)
act_space = np.prod(env.action_space.shape)



agent = DQNAgent_pytorch(DEVICE, env.observation_space, env.action_space.shape, BATCH_SIZE, LR, GAMMA, EPS, EPS_DECAY, EPS_END, LAYERS_SIZES, PLAY_MEMORY)


while True:
    # Next action:
    action = agent.get_action()
    
    # Processing:
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()