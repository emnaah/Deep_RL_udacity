# Deep Reinforcement Learning

### Project 2: Continuous Control in Reacher environment

##### Overview:

The aim of this project is to train an agent, which is adouble-jointed arm, to learn how to reach his moving target. The report describes my implementation of a Deep Deterministic Policy Gradient. My solution implements a DDPG agent based on the Continuous Control With Deep Reinforcement Learning paper (https://arxiv.org/pdf/1509.02971.pdf).

#### Architecture:

Deep Deterministic Policy Gradient (DDPG) is a reinforcement Leraninf algorithm that can be seen as a deep Q-learning for continuous action spaces.
The idea behinf it, is that it tries to learns a Q-function and a policy at once, using two

can be seen as
It is basically Q-learning for continuous action spaces. 



Actor gives best believed action
Critic takes that action, and state and predicts Q Value.


DDPG uses four neural networks: a Q network, a deterministic policy network, a target Q network, and a target policy network
The target networks are time-delayed copies of their original networks that slowly track the learned networks. 
- enprovements
  Parameter noise lets us teach agents tasks much more rapidly than with other approaches

Parameter noise helps algorithms explore their environments more effectively, leading to higher scores and more elegant behaviors. We think this is because adding noise in a deliberate manner to the parameters of the policy makes an agent’s exploration consistent across different timesteps, whereas adding noise to the action space leads to more unpredictable exploration which isn’t correlated to anything unique to the agent’s parameters.

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

My solution is working with one agent only. The multi agents project is something interesting to test !
