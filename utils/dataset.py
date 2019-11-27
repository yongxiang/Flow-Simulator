import torch
import numpy as np
from utils import device

class Dataset(object):
    def __init__(self, size=1000000):
        self.size = size
        self.sa_queue = None
        self.diff_queue = None
        self.sas_queue = None

    def reset(self):
        self.sa_queue = None
        self.diff_queue = None
        self.sas_queue = None

    def batch_push(self, batch):
        states = torch.Tensor(batch.state).to(device)
        actions = torch.Tensor(batch.action).to(device)
        next_states = torch.Tensor(batch.next_state).to(device)

        sa = torch.cat((states, actions), 1)
        sd = next_states - states
        self.push(sa, sd)

    def push(self, sa, diff, sas=None):
        if self.sa_queue is None:
            self.sa_queue = sa.clone()
            self.diff_queue = diff.clone()
            if sas is not None:
                self.sas_queue = sas.clone()
        else:
            self.sa_queue = torch.cat((self.sa_queue, sa), 0)
            self.diff_queue = torch.cat((self.diff_queue, diff), 0)
            if sas is not None:
                self.sas_queue = torch.cat((self.sas_queue, sas), 0)

        self.sa_queue = self.sa_queue[-self.size:, :]
        self.diff_queue = self.diff_queue[-self.size:, :]
        if self.sas_queue is not None:
            self.sas_queue = self.sas_queue[-self.size:, :]

    def sample(self, size=2048, idx=None):
        if idx is None:
            idx = np.random.choice(self.sa_queue.shape[0], size)
        if self.sas_queue is not None:
            return self.sa_queue[idx, :], self.diff_queue[idx, :], self.sas_queue[idx, :]
        else:
            return self.sa_queue[idx, :], self.diff_queue[idx, :]

    def all(self):
        if self.sas_queue is not None:
            return self.sa_queue, self.diff_queue, self.sas_queue
        else:
            return self.sa_queue, self.diff_queue

    def reset(self):
        self.sa_queue = None
        self.diff_queue = None
        self.sas_queue = None
