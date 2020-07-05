from unityagents import UnityEnvironment
import numpy as np
from multi_ddpg import ddpg
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = UnityEnvironment(file_name='env/Tennis_Linux/Tennis.x86_64');

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of actions and states
action_size = brain.vector_action_space_size
state = env_info.vector_observations
state_size = state.shape[1]

agent1 = Agent(state_size=state_size*2, action_size=action_size, random_seed=2)
agent2 = Agent(state_size=state_size*2, action_size=action_size, random_seed=2)

agents = [agent1,agent2]

scores = ddpg(env,brain_name,agents,n_episodes=5000, max_t=1000, print_every=100)
print(scores)

import pdb; pdb.set_trace()
fig, ax = plt.subplots()
ax.plot(np.arange(1, len(scores) + 1), scores)
ax.set_ylabel('Scores')
ax.set_xlabel('Episode #')
fig.savefig("score_x_episodes.png")
plt.show()

w = 10
mean_score = [np.mean(scores[i - w:i]) for i in range(w, len(scores))]
fig, ax = plt.subplots()
ax.plot(np.arange(1, len(mean_score) + 1), mean_score)
ax.set_ylabel('Scores')
ax.set_xlabel('Episode #')
fig.savefig("score_x_episodes_smorthed.png")
plt.show()

#https://github.com/jrandson/Tennis-udacity/blob/master/score_x_episodes.png