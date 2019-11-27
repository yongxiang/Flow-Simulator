from env.flow_lib import flow_env

import numpy as np

env, env_name = flow_env(render=True, use_inflows=True)
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

