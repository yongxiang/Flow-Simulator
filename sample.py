from flow_lib import flow_env

env, env_name = flow_env()
print(env_name, env.action_space)
