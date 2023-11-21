import random
from collections import namedtuple, deque
#store using tensors
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """
    A cyclic buffer of bounded size that holds the transitions observed recently.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """('state', 'action', 'next_state', 'reward', 'done')"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from memory.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
