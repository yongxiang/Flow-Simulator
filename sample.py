from env.flow_lib import flow_env
from flow.core.experiment import Experiment

import numpy as np

env, env_name = flow_env(render=True, use_inflows=True)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape

for i in range(10):
    state = env.reset()
    #print(state)
    for j in range(100000):
        action = np.zeros(act_dim)
        next_state, reward, done, _ = env.step(action)
        print(i, j, reward, action)
        state = next_state
        if done:
            break

