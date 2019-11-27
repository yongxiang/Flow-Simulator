from env.flow_lib import flow_env

env, env_name = flow_env(render=True)
print("simulated task: {}".format(env_name))

for i in range(10):
    state = env.reset()
    #print(state)
    for j in range(100000):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(i, j)
        if done:
            break

