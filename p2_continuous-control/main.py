from unityagents import UnityEnvironment
import numpy as np
from ddpg import ddpg
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = UnityEnvironment(file_name='env/Reacher_Linux/Reacher_Linux/Reacher.x86');

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of actions and states
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

scores = ddpg(env,brain_name,agent,n_episodes=300, max_t=1000, print_every=100)
print(scores)


