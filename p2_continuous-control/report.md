# Deep Reinforcement Learning

### Project 2: Continuous Control in Reacher environment

##### Overview:

The aim of this project is to train an agent, which is adouble-jointed arm, to learn how to reach his moving target. The report describes my implementation of a Deep Deterministic Policy Gradient. My solution implements a DDPG agent based on the Continuous Control With Deep Reinforcement Learning paper (https://arxiv.org/pdf/1509.02971.pdf).

#### Architecture:

Deep Deterministic Policy Gradient (DDPG) is a reinforcement Leraning algorithm that can be seen as a deep Q-learning for continuous action spaces. The idea behind it, is that it tries to learns a Q-function and a policy independently, using two models called Critic and Actor. Hence its belonging to the Actor-Critic methods. The policy and the value function are learned respectively py the actor and the critic models. The actor gives the best believed action given the current state of the environment. To ensure a better exploration for our agent, we can add a stochastic noise model at each training step like an Ornstein-Uhlenbeck process. Then the critic will use this output with the current state to estimate Q(s,a). Both models, actor and critic are learning from the critic form this he last operation. As described in the paper, this method is off-policy. In fact it's using four neural networks: two local actor and critic networks, but also a target network for each one of them, which is time-delayed copies of their original networks. This makes the learning more stable.
In order to achieve the required Reward average, I had to make some more improvements. First I had to integrate a decayed noise because at the beginning of learning, the agent needs to explore more, but over time, and as he begins to learn how to act well, he will need less exploration and thus we can reduce the noise on each action, leading the agent to a more elegant behaviour. The second technique I used is to ensure a smooth and less agressive updates. instead of updating the actor and critic networks every timestep, I made a 10-times update every 20 timestep.

#### Code description:

- model.py : definition of the model architechture of the actor and critic networks.
- ddpg.py : implementation of DDPG algorithm
- agent.py : implementation of agent using DDPG and some improvements.
- Continuous_Control.ipynb : start environment, test it, train agent and plot scores.

#### Hyperparameters:

- BUFFER_SIZE : int(1e6) # replay buffer size
- BATCH_SIZE : 256 # minibatch size
- GAMMA : 0.99 # discount factor
- TAU : 1e-3 # for soft update of target parameters
- LR_ACTOR : 1e-3 # learning rate of the actor
- LR_CRITIC : 1e-3 # learning rate of the critic
- WEIGHT_DECAY : 0 # L2 weight decay
- UPDATE_EVERY : 20 # timesteps between updates
- EPSILON : 1.0 # epsilon for the noise process added to the actions
- EPSILON_DECAY : 1e-6 # decay for epsilon above

#### Results:

![image](results/Figure1.png)

#### Future Work:
- Trying out different Actor Critic methods.
- Testing the multi agents project.
