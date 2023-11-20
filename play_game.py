import torch 

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
obs_size = env.observation_space.shape

agent = DQNAgent_pytorch(DEVICE, 2, env.action_space.shape, BATCH_SIZE, LR, GAMMA, EPS, EPS_DECAY, EPS_END, LAYERS_SIZES, PLAY_MEMORY)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()