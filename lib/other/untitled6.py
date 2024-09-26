# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:44:10 2023

@author: awei
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SACActor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        super(SACActor, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Discrete action layer
        self.fc_discrete = nn.Linear(32, discrete_action_dim)
        
        # Continuous action layers
        self.fc_mean = nn.Linear(32, continuous_action_dim)
        self.fc_log_std = nn.Linear(32, continuous_action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Discrete action
        discrete_action_logits = self.fc_discrete(x)
        discrete_action_probs = F.softmax(discrete_action_logits, dim=-1)
        
        # Continuous action
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clipping log_std for numerical stability
        
        return discrete_action_probs, mean, log_std

# Example usage
state_dim = 10
discrete_action_dim = 5000
continuous_action_dim = 9

# Create SACActor
actor = SACActor(state_dim, discrete_action_dim, continuous_action_dim)

# Forward pass to get action probabilities, mean, and log_std
state = torch.randn(1, state_dim)
discrete_probs, mean, log_std = actor(state)

# Sample discrete action
discrete_action = torch.multinomial(discrete_probs, 1)

# Sample continuous action using reparameterization trick
continuous_action = mean + torch.exp(log_std) * torch.randn_like(mean)

# Print results
print("Discrete Action:", discrete_action.item())
print("Continuous Action Mean:", mean.detach().numpy())
print("Continuous Action Log Std:", log_std.detach().numpy())
print("Continuous Action Sample:", continuous_action.detach().numpy())
