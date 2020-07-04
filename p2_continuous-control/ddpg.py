import numpy as np
import torch
from collections import deque

from ddpg_agent import Agent

def ddpg(env,brain_name,agent, n_episodes=300, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    printed = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]

            # next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done,t)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_deque)
            ),
            end="",
        )
        torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
        torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if np.mean(scores_deque)>30 and not printed:
            print('env solved in {} episodes'.format(i_episode))
            printed = True

    return scores