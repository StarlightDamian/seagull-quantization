# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:24:58 2023

@author: awei
reinforcement_learning_ddpg
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import pandas as pd

from __init__ import path

# Define the actor and critic neural networks using PyTorch
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 64)
        self.fc_action = nn.Linear(action_dim, 64)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):
        x_state = torch.relu(self.fc_state(state))
        x_action = torch.relu(self.fc_action(action))
        
        # Ensure the dimensions are compatible for concatenation
        if len(x_state.shape) == 1:
            x_state = x_state.unsqueeze(0)
        if len(x_action.shape) == 1:
            x_action = x_action.unsqueeze(0)

        x = torch.cat([x_state, x_action], dim=1)
        x = torch.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = DDPGActor(state_dim, action_dim)
        self.critic = DDPGCritic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)

        # Critic loss
        q_value = self.critic(state, action)
        next_action = self.actor(next_state)
        next_q_value = self.critic(next_state, next_action.detach())
        target_q = reward + 0.99 * next_q_value * (1 - done)
        critic_loss = nn.MSELoss()(q_value, target_q)

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
    def simulate_trading(self, data):
        capital = 1000000  # Initial capital
        position = 0  # Initial position
        state = data.iloc[0].values
        total_reward = 0

        for i in range(1, len(data)):
            action = self.get_action(state)
            close_price = data['close'].iloc[i]
            reward = (close_price - data['close'].iloc[i - 1]) * position  # Profit or loss from holding position
            total_reward += reward

            # Buy/Sell decision based on DDPG agent's action
            if action > 0 and capital >= close_price:  # Buy
                position += capital // close_price  # Buy as many shares as possible
                capital -= position * close_price
            elif action < 0 and position > 0:  # Sell
                capital += position * close_price
                position = 0

            next_state = data.iloc[i].values
            self.train(state, action, reward, next_state, False)  # Training on each step
            state = next_state

        return total_reward, capital + position * data['close'].iloc[-1]

# Simulated stock trading environment
class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.total_steps = len(data)
        self.state_dim = len(data.columns)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step == self.total_steps - 1
        reward = self.data['Close'].pct_change().iloc[self.current_step] * action[0]
        next_state = self.data.iloc[self.current_step].values
        return next_state, reward, done, {}

# Simulated data (replace this with your actual stock data)
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
data = pd.DataFrame(index=dates)
data['close'] = np.random.randn(len(dates)).cumsum() + 100

# Initialize agent and environment
state_dim = len(data.columns)
action_dim = 1
agent = DDPGAgent(state_dim, action_dim)
env = StockTradingEnv(data)

# Simulate trading for 100 episodes
for episode in range(100):
    total_reward, final_balance = agent.simulate_trading(data)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Final Balance: {final_balance}")
    
# =============================================================================
# # Training loop (simulated trading)
# for episode in range(30):
#     state = env.reset()
#     total_reward = 0
# 
#     while True:
#         action = agent.get_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.train(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward
# 
#         if done:
#             print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
#             break
# =============================================================================

# Save the trained models
torch.save(agent.actor.state_dict(), f'{path}/checkpoint/actor_model.pth')
torch.save(agent.critic.state_dict(), f'{path}/checkpoint/critic_model.pth')

# Load the saved models
loaded_actor = DDPGActor(state_dim, action_dim)
loaded_critic = DDPGCritic(state_dim, action_dim)
loaded_actor.load_state_dict(torch.load(f'{path}/checkpoint/actor_model.pth'))
loaded_critic.load_state_dict(torch.load(f'{path}/checkpoint/critic_model.pth'))

# Use the loaded models for prediction on real data
real_data = data.head(100)  # pd.read_csv('real_stock_data.csv')  # Replace 'real_stock_data.csv' with your actual stock data file
real_env = StockTradingEnv(real_data)

# Predictions on real data
state = real_env.reset()
predictions = []

while True:
    action = loaded_actor(torch.FloatTensor(state)).detach().numpy()
    next_state, _, done, _ = real_env.step(action)
    predictions.append((state, action[0]))
    state = next_state

    if done:
        break

# Display the predictions
for i, (state, action) in enumerate(predictions):
    print(f"Step {i + 1}: State = {state}, Predicted Action = {action}")

