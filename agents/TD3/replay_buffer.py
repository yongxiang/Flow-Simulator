import math, os
import numpy as np
import torch
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer(object):
    def __init__(self, max_size=5e4, save_reward=True):
        self.save_reward = save_reward
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def batch_add(self, data_itr):
        for data in data_itr:
            self.add(data)

    def sample(self, batch_size, discriminator=None):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        # x: state
        # y: next state
        # u: action
        # r: reward
        # d: done
        x, y, u, r, d = [], [], [], [], []

        if self.save_reward:
            for i in ind: 
                X, Y, U, R, D = self.storage[i]
                x.append(np.array(X, copy=False))
                y.append(np.array(Y, copy=False))
                u.append(np.array(U, copy=False))
                r.append(np.array(R, copy=False))
                d.append(np.array(D, copy=False))
            x = np.array(x)
            y = np.array(y)
            u = np.array(u)
            r = np.array(r).reshape(-1, 1)
            d = np.array(d).reshape(-1, 1)

        else:
            for i in ind:
                X, Y, U, D = self.storage[i]
                x.append(np.array(X, copy=False))
                y.append(np.array(Y, copy=False))
                u.append(np.array(U, copy=False))
                d.append(np.array(D, copy=False))
            x = np.array(x)
            y = np.array(y)
            u = np.array(u)
            d = np.array(d).reshape(-1, 1)
            r = discriminator(self.cat(x, u)).cpu().detach().numpy()
        return x, y, u, r, d

    def cat(self, x, u):
        sas = np.concatenate((x, u), 1)
        sas = torch.Tensor(sas).to(device)
        return sas
