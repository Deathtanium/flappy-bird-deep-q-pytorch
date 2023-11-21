# %%
import torch 
import numpy as np

import flappy_bird_gymnasium
import gymnasium

from deepq_agent import DQNAgent_pytorch

from gymnasium.wrappers import FlattenObservation

# %%
env = gymnasium.make("CartPole-v1") #CartPole-v1 FlappyBird-v0
state,_ = env.reset()

# %%
#get size of observation space
obs_space = len(state) #overriden
act_space = env.action_space.n

obs_space, act_space

# %%
TARGET_UPDATE = 100
DEVICE = 'cuda' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
GAMMA = 0.99
EPS = 0.9
EPS_DECAY = 1000 
EPS_END = 0.05
BATCH_SIZE = 128
PLAY_MEMORY = 10000
LAYERS_SIZES = [512, 512]
EPOCHS = 1000000
TAU = 0.005

# %%
agent = DQNAgent_pytorch(
        device=DEVICE,
        act_space=act_space,
        obs_space=obs_space,
        training_batch_size=BATCH_SIZE,
        learn_rate=LR,
        gamma=GAMMA,
        eps_start=EPS,                                                               #rate of exploration
        eps_decay_rate=EPS_DECAY,                                                   
        eps_floor=EPS_END,                                                       
        network_shape=LAYERS_SIZES,
        tau=TAU,
        pmem_buffer_size=PLAY_MEMORY
    )

# %%
#agent.load("model.pt")
def state_filter(state:np.ndarray):
    state = state#[:-1]
    return state

# %%
#env = gymnasium.make("FlappyBird-v0")
 
#env = FlattenObservation(env)

from itertools import count
import random

frames_since_jump = 0
for epoch in range(EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state_filter(state), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0
    for t in count():
        action = agent.get_action(state, env.action_space)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=DEVICE)
        done = terminated or truncated
        next_state = torch.tensor(state_filter(next_state), dtype=torch.float32, device=DEVICE).unsqueeze(0)

        total_reward += reward.item()


        done = torch.tensor([done], device=DEVICE)

        agent.memory.push(state, action, next_state, reward, done)

        state = next_state
        
        agent.train()

        policy_dict = agent.policy_net.state_dict()
        target_dict = agent.target_net.state_dict()
        for name in policy_dict:
            target_dict[name] = agent.tau * policy_dict[name] + (1.0 - agent.tau) * target_dict[name]
        agent.target_net.load_state_dict(target_dict)

        if done.item():
            break

    print("Iteration: ", epoch, "Total reward: ", t)

