import math
import torch
from torch.distributions.bernoulli import Bernoulli
from utils import device

def log_density(actions, prob):
    m = Bernoulli(prob)
    return m.log_prob(actions).sum(1, keepdim=True)


def gae(batch, value_net, gamma, tau):
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    actions = torch.Tensor(batch.action).to(device)
    states = torch.Tensor(batch.state).to(device)
    values = value_net(states.to(device))
    
    returns = torch.Tensor(actions.size(0), 1).to(device)
    deltas = torch.Tensor(returns.shape).to(device)
    advantages = torch.Tensor(returns.shape).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages
