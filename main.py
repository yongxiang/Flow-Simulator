from env.flow_lib import flow_env
from utils import parser
from agents.TD3 import TD3
from agents.PPO import PPO
from agents.TRPO import TRPO

import numpy as np
import gym
import gym.spaces

import torch
import torch.optim as optim
import torch.nn as nn

torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

args = parser.parser()
print('agent type: {}'.format(args.pg_type))
env, env_name = flow_env(render=args.render, use_inflows=True)

### seeding ###
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
###############

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape

for i in range(10):
    state = env.reset()
    #print(state)
    for j in range(100000):
        action = np.ones(act_dim) * 10
        state, reward, done, _ = env.step(action)
        print(i, j, reward, action)
        if done:
            break

