import numpy as np
import torch
from collections import deque

from ddpg_agent import Agent

def ddpg(env,brain_name,agents, n_episodes=300, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    printed = False
    agent1, agent2 = agents

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations
        state = np.reshape(state, (1, -1))
        episode_score = np.zeros(2)
        for t in range(max_t):
            action1 = agent1.act(state)
            action2 = agent2.act(state)

            action = [action1, action2]

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            next_state = np.reshape(next_state, (1, -1))
            reward1, reward2 = env_info.rewards  # get the reward
            done1,done2 = env_info.local_done

            agent1.step(state, action1, reward1, next_state, done1,t)
            agent2.step(state, action2, reward2, next_state, done2,t)

            episode_score += [reward1,reward2]

            state = next_state


            if np.any([done1,done2]):
                break

        scores_deque.append(np.max(episode_score))
        scores.append(np.max(episode_score))
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_deque)
            ),
            end="",
        )
        torch.save(agent1.actor_local.state_dict(), "checkpoint_actor_1.pth")
        torch.save(agent1.critic_local.state_dict(), "checkpoint_critic_1.pth")

        torch.save(agent2.actor_local.state_dict(), "checkpoint_actor_2.pth")
        torch.save(agent2.critic_local.state_dict(), "checkpoint_critic_2.pth")
        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if np.mean(scores_deque)>=0.5 and not printed:
            print('Environment is solved in {} episodes only !'.format(i_episode))
            printed = True
            break

    return scores