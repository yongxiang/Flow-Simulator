import random
from collections import namedtuple

import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


    def reg_rewards(self, model, scale=1):
        batch = self.sample()
        state = torch.Tensor(batch.state).to(device)
        action = torch.Tensor(batch.action).to(device)
        next_state = torch.Tensor(batch.next_state).to(device)
        sas = torch.cat((state, action, next_state), 1)

        reward = -torch.sigmoid(model(sas)).view(-1)
        reward = (reward - reward.mean()).detach().cpu()
        return reward * scale
