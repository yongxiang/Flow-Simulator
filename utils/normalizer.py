from copy import deepcopy, copy
import torch
import torch.nn as nn
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Normalizer(nn.Module):
    def __init__(self, obs_dim, eps=1e-8, norm=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mean = torch.zeros(obs_dim).to(device)
        self.std = torch.ones(obs_dim).to(device)
        self.n = 0
        self.eps = eps
        self.norm = norm

    def forward(self, x, inverse=False):
        if not self.norm:
            return x
        if not x.is_cuda:
            std = self.std.clone().cpu()
            mean = self.mean.clone().cpu()
        else:
            std = self.std.clone()
            mean = self.mean.clone()

        ### prevent zero std
        std[std < 1e-3] = 1.0

        #return x

        if inverse:
            return x * (std) + mean
        return (x - mean) / (std)

    def update(self, samples: np.ndarray):
        if not self.norm:
            return
        old_mean, old_std, old_n = self.mean.cpu().numpy(), self.std.cpu().numpy(), copy(self.n)
        samples = samples - old_mean

        m = samples.shape[0]
        delta = samples.mean(0)
        new_n = old_n + m
        new_mean = old_mean + delta * m / new_n
        new_std = np.sqrt((old_std**2 * old_n + samples.var(axis=0) * m + delta**2 * old_n * m / new_n) / new_n)

        self.mean = torch.from_numpy(new_mean).to(device)
        self.std = torch.from_numpy(new_std).to(device)
        self.n = new_n

        print(self.mean, self.std, self.n)
