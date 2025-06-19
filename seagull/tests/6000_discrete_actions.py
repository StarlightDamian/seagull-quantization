# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:11:34 2023

@author: awei
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
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

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = SACActor(state_dim, action_dim)
        self.critic = SACCritic(state_dim, action_dim)
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
        # Your simulation logic here
        pass

# =============================================================================
# import torch.nn.functional as F
# 
# class SACActor(nn.Module):
#     def __init__(self, state_dim, continuous_action_dim, discrete_action_dim):
#         super(SACActor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc_continuous = nn.Linear(32, continuous_action_dim)
#         self.fc_discrete = nn.Linear(32, discrete_action_dim)
# 
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         
#         # Continuous actions
#         action_continuous = torch.tanh(self.fc_continuous(x))
# 
#         # Discrete actions (categorical distribution)
#         action_discrete_logits = self.fc_discrete(x)
#         action_discrete_probs = F.softmax(action_discrete_logits, dim=-1)
# 
#         return action_continuous, action_discrete_probs
# 
# =============================================================================
