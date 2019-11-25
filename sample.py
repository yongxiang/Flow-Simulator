from env.flow_lib import flow_env

env, env_name = flow_env()
print("simulated task: {}".format(env_name))

for _ in range(10):
    state = env.reset()
    print(state)
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            break

