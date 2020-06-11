import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent import Agent
from dqn import dqn

# sys.path.insert(1, '/python/unityagents')
# from environment import UnityEnvironment

env = UnityEnvironment(file_name="banana_env/Banana_Linux/Banana.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]


# number of actions and states
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size=state_size, action_size=action_size, seed=0, dqn_type="double", dueling = True)
scores = dqn(agent, env, brain_name)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()
