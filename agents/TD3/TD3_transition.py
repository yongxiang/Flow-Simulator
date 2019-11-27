import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition import Dynamic, Transition_TD3Value
from models.transition import Dynamic
from agents.TD3.replay_buffer import ReplayBuffer
from agents.TD3.TD3 import TD3
from utils.loss import L2Loss, MSELoss
from utils import device

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3_transition(TD3):
    def __init__(self, state_dim, action_dim, max_action, iters, normalizer=None, lr=1e-3, behavior_cloning=0.):
        super(TD3_transition, self).__init__(state_dim, action_dim, max_action, iters, normalizer, lr)
        self.actor = Dynamic(state_dim + action_dim, state_dim, 500, normalizer).to(device)
        self.actor_target = Dynamic(state_dim + action_dim, state_dim, 500, normalizer).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)        
       
        self.critic = Transition_TD3Value(2 * state_dim + action_dim, hidden_dim=500, normalizer=normalizer).to(device)
        self.critic_target = Transition_TD3Value(2 * state_dim + action_dim, hidden_dim=500, normalizer=normalizer).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(save_reward=False)
        self.behavior_cloning = behavior_cloning
        self.criterion = L2Loss()

    def new_replaybuffer(self):
        self.replay_buffer = ReplayBuffer(save_reward=False)

    def buffer_add(self, batch):
        states = batch.state.tolist()
        actions = batch.action.tolist()
        next_states = batch.next_state.tolist()
        dones = (1. - batch.mask).tolist()

        self.replay_buffer.batch_add(zip(states, next_states, actions, dones))

    def train(self, dataset, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, discriminator=None):

        for it in range(self.iters):

            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size, discriminator)
            state = torch.DoubleTensor(x).to(device)
            action = torch.DoubleTensor(u).to(device)
            next_state = torch.DoubleTensor(y).to(device)
            done = torch.DoubleTensor(1 - d).to(device)
            reward = torch.DoubleTensor(r).to(device)

            # Select action according to policy and add clipped noise 
            noise = torch.DoubleTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

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
                sa, sd = dataset.sample()
                pred_sd = self.actor(sa)
                bc_loss = self.criterion(pred_sd, sd)

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                (actor_loss + self.behavior_cloning * bc_loss).backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
