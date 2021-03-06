# Deep Reinforcement Learning

### Project 1: Navigation in banana envirornement

##### Overview:

The aim of this project is to train an agent to learn how to navigate in the banana evironmenet by collecting yellow bananas and avoid blue ones. The report describes my implementation of a deep Q-network (DQN) agent. My solution implements a deep Q-network (DQN) agent using the following techniques:

- Double DQN: https://arxiv.org/pdf/1509.06461.pdf
- Dueling DQN: https://arxiv.org/pdf/1511.06581.pdf

#### Architecture:

- ###### DQN:

  In Q-learning, we aim to learn the Q table. In fact, Q (s, a) values Q-values represents the maximum discounted future reward for all possible combinations of s and a, and gives the performance of your agent when performing an action in state s. However, Once we have a large number of states and actions, this table Q is no longer effective because it is difficult to represent. This is where the idea for DEEP Q-Learning came from. The representation of Q will be done using a non linear function or a neural network which takes input actions. The DQN has the weights parameter θ and tries to learn to choose an optimal action. To make this modelisation even more stable, one good technique is to use exprerience replay. It consists on storing experience tuples (s, a, r, s') in replay memory D, and during training we use andom mini-batches from the replay memory and this has shown a great improvement in the agent behavior.

- ###### Double DQN:

  With this technique we seek more stability for the learning. for this, we use two similar neural networks 'local' and 'target'. The first one learns during the experience replay and the second one is a copy of the last episode of the first model and is used to calculate the target Q value. In other terms, we use the local model to get the indexes that give the best Q value and. Then we use them with the target network to get the best action.
  ![image](images/double_dqn.png)

- ###### Dueling DQN:

  The idea behind duel DQN is to change the model architecture. We will compute the value state function Q(s,a) as the sum of the Value function V(s) and an Advantage function A(s,a). Here, the value function tells how good it is to be in a state. The Q function measures how good is to choose a particular action when in this state.Thus, the advantage function subtracts the value function from the Q function to obtain how important is each action.
  ![image](images/duel_dqn.png)

#### Code description:

- model.py : definition of the model architechture for Simple and Dueling networks.
- dqn.py : implementation of DQN algorithm
- agent.py : implementation of agent using DQN and Double DQN methods.
- Navigation.ipynb : start environment, test it, train agent and plot scores.

#### Hyperparameters:

- BUFFER_SIZE = int(1e5)          # replay buffer size
- BATCH_SIZE = 64                 # minibatch size
- GAMMA = 0.99                    # discount factor
- TAU = 1e-3                      # for soft update of target parameters
- LR = 1e-4                       # learning rate
- UPDATE_EVERY = 4                # how often to update the network

#### Results:

![image](results/double_dueling_dqn.png)

#### Future Work:

In my solution, experience transitions are sampled randomly from the replay memory. To improve the agent, we can add Prioritized Experience Replay, to replay important transitions more frequently, and therefore learn more efficiently. (https://arxiv.org/pdf/1511.05952.pdf)
