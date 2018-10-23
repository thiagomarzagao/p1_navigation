import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unityagents import UnityEnvironment
from collections import namedtuple, deque

# set hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    '''
    architecture and activation functions of the neural network
    '''

    def __init__(self, state_size, action_size, seed, hidden_size = 24):
        '''
        state_size -> dimension of state
        action_size -> dimension of action
        seed -> fix seed to allow replication
        hidden_size -> number of neurons in each hidden layer
        '''
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        '''
        forward pass of the computation
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    '''
    agent that learns to get yellow bananas and avoid blue bananas
    '''

    def __init__(self, state_size, action_size, seed):
        '''
        state_size -> dimension of state
        action_size -> dimension of action
        seed -> fix seed to allow replication
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnet = QNetwork(state_size, action_size, seed).to(device)
        self.qnet_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr = LR)

        # replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            # and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps = 0.):
        '''
        state -> current state
        eps -> epsilon (for epsilon-greedy action selection)
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet.eval()
        with torch.no_grad():
            action_values = self.qnet(state)
        self.qnet.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''
        experiences -> (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma -> discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        # get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)

        # compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # get expected Q values from local model
        Q_expected = self.qnet(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft-update target network
        self.soft_update(self.qnet, self.qnet_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''
        soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        local_model -> PyTorch model
        target_model -> PyTorch model
        tau -> interpolation parameter
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer(object):
    '''
    fixed-size buffer to store experience tuples
    '''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''
        action_size -> dimension of action
        buffer_size -> max size of replay buffer
        batch_size -> size of training batch
        seed -> fix seed to allow replication
        '''

        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names = ['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
        add a new experience to memory
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''
        randomly sample a batch of experiences from memory
        '''
        experiences = random.sample(self.memory, k = self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        return the current size of internal memory
        '''
        return len(self.memory)

env = UnityEnvironment(file_name = 'Banana.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode = True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)

n_episodes = 2000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
max_t = 1000

agent = Agent(state_size = state_size, action_size = action_size, seed = 42)

scores = []                          # list containing scores from each episode
scores_std = []                      # List containing the std dev of the last 100 episodes
scores_avg = []                      # List containing the mean of the last 100 episodes
scores_window = deque(maxlen = 100)  # last 100 scores
eps = eps_start                      # initialize epsilon

for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode = True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    for t in range(max_t):
        action = agent.act(state, eps)                 # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.step(state, action, reward, next_state, done)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    scores_window.append(score)                        # save most recent score
    scores.append(score)                               # save most recent score
    scores_std.append(np.std(scores_window))           # save most recent std dev
    scores_avg.append(np.mean(scores_window))          # save most recent std dev
    eps = max(eps_end, eps_decay * eps)                # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end = '')
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window) >= 13.0:
        s_msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
        print(s_msg.format(i_episode, np.mean(scores_window)))
#        torch.save(agent.qnet.state_dict(), 'checkpoint.pth')
        break

# plot scores over episodes
df = pd.DataFrame()
df['episode'] = range(len(scores))
df['score'] = scores
df.plot.scatter(x = 'episode', y = 'score')
plt.show()

env.close()
