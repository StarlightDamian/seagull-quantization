# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 14:17:40 2024

@author: awei
sac_recycle
"""

# =============================================================================
#     def get_action(self, state, exploration_noise=0.1):
#         with torch.no_grad():
#             state = state.to(device)
# 
# 
# 
#             discrete_action_probs = self.discrete_actor(state)
#             continuous_mean, continuous_log_std = self.continuous_actor(state)
# 
#             # Sample discrete action
#             discrete_action = torch.multinomial(discrete_action_probs, 1)
#             
#             # Sample continuous action using reparameterization trick
#             continuous_action = continuous_mean + torch.exp(continuous_log_std) * torch.randn_like(continuous_mean)
# 
#             # Combine discrete and continuous actions
#             action = torch.cat([discrete_action, continuous_action], dim=-1)
#             print('discrete_action',discrete_action)
#             print('continuous_action',continuous_action)
#             print('action',action)
#             # Add exploration noise
#             action += exploration_noise * torch.randn_like(action)
#             #action = torch.clamp(action, 0.0, 1.0)  # Clip continuous part to [0, 1]
# 
#             return action.cpu().detach().numpy()
# =============================================================================


# =============================================================================
#     def simulate_trading(self, state, total_reward, next_state, done):
#         action = self.get_action(state)
#         state = torch.FloatTensor(state.to_numpy()).to(device)
#         self.train(state, action, total_reward, next_state, done)
# =============================================================================
# =============================================================================
# import argparse
# from datetime import datetime
# import math
# import os
# 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import gym
# import pandas as pd
# 
# from __init__ import path
# from trade import trade_eval
# from base import base_connect_database, base_utils
# 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# 
# TRADE_ORDER_TABLE_NAME = 'trade_order_details'
# TRADE_MODEL_TABLE_NAME = 'trade_model'
# #TRADE_SHARE_REGISTER_TABLE_NAME = 'trade_share_register'
# 
# 
# # Define the actor and critic neural networks using PyTorch
# class SACActor(nn.Module):
#     def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):#action_dim
#         super(SACActor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         #self.fc3 = nn.Linear(32, action_dim)
#         
#         # Discrete action layer
#         self.fc_discrete = nn.Linear(32, discrete_action_dim)
#         
#         # Continuous action layers
#         self.fc_mean = nn.Linear(32, continuous_action_dim)
#         self.fc_log_std = nn.Linear(32, continuous_action_dim)
#         
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         
#         #action_mean = torch.sigmoid(self.fc3(x))  # [0, 1]
#         #log_std = torch.zeros_like(action_mean)  # Fixed log_std for simplicity
#         
#         # Discrete action
#         discrete_action_logits = self.fc_discrete(x)
#         discrete_action_probs = F.softmax(discrete_action_logits, dim=-1)
#         
#         # Continuous action
#         action_mean = self.fc_mean(x)
#         log_std = self.fc_log_std(x)
#         log_std = torch.clamp(log_std, -20, 2)  # Clipping log_std for numerical stability
#         
#         return discrete_action_probs, action_mean, log_std
# 
# 
#     
# class SACCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(SACCritic, self).__init__()
#         self.fc_state = nn.Linear(state_dim, 64)
#         self.fc_action = nn.Linear(action_dim, 64)
#         self.fc1 = nn.Linear(128, 128)
#         self.fc2 = nn.Linear(128, 1)
# 
#     def forward(self, state, action):
#         x_state = torch.relu(self.fc_state(state))
#         x_action = torch.relu(self.fc_action(action))
#         
#         # Ensure the dimensions are compatible for concatenation
#         if len(x_state.shape) == 1:
#             x_state = x_state.unsqueeze(0)
#         if len(x_action.shape) == 1:
#             x_action = x_action.unsqueeze(0)
# 
#         x = torch.cat([x_state, x_action], dim=1)
#         x = torch.relu(self.fc1(x))
#         q_value = self.fc2(x)
#         return q_value
# 
# class SACAgent:
#     def __init__(self, state_dim, action_dim):
#         self.lr_actor = 1e-4
#         self.lr_critic = 1e-4
#         self.actor = SACActor(state_dim, action_dim).to(device)
#         self.critic = SACCritic(state_dim, action_dim).to(device)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
# 
#     def get_action(self, state, exploration_noise=0.1):
#         with torch.no_grad():
#             state = torch.FloatTensor(state.values).to(device)  # Convert state to PyTorch tensor
#             action_mean, log_std = self.actor(state)
#             std = torch.exp(log_std)
#             raw_action = action_mean + std * torch.randn_like(std).to(device)
#             #action = torch.tanh(raw_action)  # Scale to [-1, 1]
#             action = torch.sigmoid(raw_action)
#             
#             # Add exploration noise
#             action += exploration_noise * torch.randn_like(action)
#             action = torch.clamp(action, 0.0, 1.0)  # Clip to [0, 1]
#             return action.cpu().detach().numpy()
#         
#     def train(self, state, action, reward, next_state, done):
#         state = torch.FloatTensor(state.values).to(device)
#         action = torch.FloatTensor(action).to(device)
#         reward = torch.FloatTensor([reward]).to(device)
#         next_state = torch.FloatTensor(next_state.values).to(device)  #.values state is a dataframe
# 
#         # Critic loss
#         q_value = self.critic(state, action)
#         next_action, _ = self.actor(next_state)
#         next_q_value = self.critic(next_state, next_action.detach())
#         target_q = reward + 0.99 * next_q_value * (1 - done)
#         critic_loss = nn.MSELoss()(q_value, target_q)
# 
#         # Actor loss
#         action_mean, _ = self.actor(state)
#         actor_loss = -self.critic(state, action_mean).mean()
# 
#         # Update networks
#         self.actor_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()
#         actor_loss.backward()
#         critic_loss.backward()
#         self.actor_optimizer.step()
#         self.critic_optimizer.step()
#         
#     def save_model(self, primary_key):
#         #if not os.path.exists('models'):
#         #    os.makedirs('models')
# 
#         actor_path = f'{path}/checkpoint/sac_actor/{primary_key}.pth'
#         critic_path = f'{path}/checkpoint/sac_critic/{primary_key}.pth'
# 
#         torch.save(self.actor.state_dict(), actor_path)
#         torch.save(self.critic.state_dict(), critic_path)
#         
#     def load_model(self, primary_key):
#         actor_path = f'{path}/checkpoint/sac_actor/{primary_key}.pth'
#         critic_path = f'{path}/checkpoint/sac_critic/{primary_key}.pth'
#     
#         self.actor.load_state_dict(torch.load(actor_path))
#         self.critic.load_state_dict(torch.load(critic_path))
# 
#     def simulate_trading(self, state):
#         action = self.get_action(state)
#         
#         
#         discrete_probs, mean, log_std = actor(state)
#         
#         # Sample discrete action
#         discrete_action = torch.multinomial(discrete_probs, 1)
#         
#         # Sample continuous action using reparameterization trick
#         continuous_action = mean + torch.exp(log_std) * torch.randn_like(mean)
#         
#         # Print results
#         print("Discrete Action:", discrete_action.numpy())
#         print("Continuous Action Mean:", mean.detach().numpy())
#         print("Continuous Action Log Std:", log_std.detach().numpy())
#         print("Continuous Action Sample:", continuous_action.detach().numpy())
#         
#         self.train(state, action, total_reward, next_state, False)
# 
# if __name__ == '__main__':
#     # Example usage
#     state_dim = 10
#     discrete_action_dim = 5000
#     continuous_action_dim = 9
#     
#     # Create SACActor
#     #actor = SACActor(state_dim, discrete_action_dim, continuous_action_dim)
#     agent = SACAgent(state_dim, discrete_action_dim, continuous_action_dim)
#     # Forward pass to get action probabilities, mean, and log_std
#     state = torch.randn(5, state_dim)
# 
# =============================================================================

        
# =============================================================================
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# 
# =============================================================================
# =============================================================================
# class SACActor(nn.Module):
#     def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
#         super(SACActor, self).__init__()
#         
#         # Shared layers
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         
#         # Discrete action layer
#         self.fc_discrete = nn.Linear(32, discrete_action_dim)
#         
#         # Continuous action layers
#         self.fc_mean = nn.Linear(32, continuous_action_dim)
#         self.fc_log_std = nn.Linear(32, continuous_action_dim)
# 
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         
#         # Discrete action
#         discrete_action_logits = self.fc_discrete(x)
#         discrete_action_probs = F.softmax(discrete_action_logits, dim=-1)
#         
#         # Continuous action
#         mean = self.fc_mean(x)
#         log_std = self.fc_log_std(x)
#         log_std = torch.clamp(log_std, -20, 2)  # Clipping log_std for numerical stability
#         
#         return discrete_action_probs, mean, log_std
# =============================================================================
