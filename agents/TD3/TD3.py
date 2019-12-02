import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from models.agent import Policy, TD3Value
from agents.TD3.replay_buffer import ReplayBuffer
from utils import device

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3(object):
    def __init__(self, state_dim, action_dim, iters, normalizer=None, lr=1e-3):
        self.actor = Policy(state_dim, action_dim, hidden_dim=400, normalizer=normalizer).to(device)
        self.actor_target = Policy(state_dim, action_dim, hidden_dim=400, normalizer=normalizer).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TD3Value(state_dim + action_dim, hidden_dim=400, normalizer=normalizer).to(device)
        self.critic_target = TD3Value(state_dim + action_dim, hidden_dim=400, normalizer=normalizer).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.iters = iters

        self.type = 'TD3'

    def new_replaybuffer(self):
        self.replay_buffer = ReplayBuffer()

    def buffer_add(self, batch):
        states = batch.state.tolist()
        actions = batch.action.tolist()
        next_states = batch.next_state.tolist()
        rewards = batch.reward.tolist()
        dones = (1. - batch.mask).tolist()

        self.replay_buffer.batch_add(zip(states, next_states, actions, rewards, dones))

    def get_actor(self):
        return self.actor

    def select_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state).cpu().data.numpy() > 0.5


    def train(self, batch_size=100, discount=0.99, tau=0.005, policy_freq=2):

        for it in range(self.iters):

            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.DoubleTensor(x).to(device)
            action = torch.DoubleTensor(u).to(device)
            next_state = torch.DoubleTensor(y).to(device)
            done = torch.DoubleTensor(1 - d).to(device)
            reward = torch.DoubleTensor(r).to(device)

            # Select action according to policy and add clipped noise 
            m = Bernoulli(torch.ones(u.shape) * 0.2)
            noise = m.sample().to(device)
            next_action = ((noise - self.actor_target(next_state)) != 0).double()


            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    """
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
    """
    def to_train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def to_eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cpu(self):
        self.actor.cpu()
        self.actor_target.cpu()
        self.critic.cpu()
        self.critic_target.cpu()

    def to(self, device):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
