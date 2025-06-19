# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:59:48 2023

@author: awei
"""
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Assuming the __init__.py file is in the same directory
from seagull.settings import PATH
from trade import trade_eval
from base import base_connect_database, base_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRADE_ORDER_TABLE_NAME = 'trade_order_details'
TRADE_MODEL_TABLE_NAME = 'trade_model'

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_mean = torch.sigmoid(self.fc3(x))  # [0, 1]
        log_std = torch.zeros_like(action_mean)  # Fixed log_std for simplicity

        return action_mean, log_std

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
        self.lr_actor = 1e-3
        self.lr_critic = 1e-3
        self.actor = SACActor(state_dim, action_dim).to(device)
        self.critic = SACCritic(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def get_action(self, state, exploration_noise=0.1, dataset_type='train'):
        """
        在动作均值上添加随机噪声：
SAC 算法中的策略网络（Actor）会根据当前状态预测出动作的均值 action_mean，然后通过一个学习得到的标准差 std 来确定动作的分布。因此，通过在 action_mean 上加上服从正态分布的随机噪声，可以使得输出的动作具有一定的随机性，有助于探索更多的动作空间。

在动作上添加探索噪声：
另一方面，SAC 算法也会在生成的动作上添加额外的探索噪声，这通常通过在已经确定的动作上加上服从正态分布的随机噪声实现。这个探索噪声可以帮助算法在训练过程中更好地探索环境，从而学习到更加鲁棒和泛化的策略。

因此，综合来看，这两种方式在 SAC 算法中都具有重要作用：

在动作均值上添加随机噪声有助于使策略网络的输出更加多样化，避免过度依赖某些确定性动作。
在动作上添加探索噪声则能够促进智能体在训练过程中进行有效的探索，从而学习到更优的策略。

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        exploration_noise : TYPE, optional
            DESCRIPTION. The default is 0.1.
        dataset_type : TYPE, optional
            DESCRIPTION. The default is 'train'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)  # Convert state to PyTorch tensor
            action_mean, log_std = self.actor(state)
            std = torch.exp(log_std)
            if dataset_type=='train':
                raw_action = action_mean + std * torch.randn_like(std).to(device)
            else:
                raw_action = action_mean
            #action = torch.tanh(raw_action)  # Scale to [-1, 1]
            action = torch.sigmoid(raw_action)
            
            # Add exploration noise
            if dataset_type=='train':
                action += exploration_noise * torch.randn_like(action)
            action = torch.clamp(action, 0.0, 1.0)  # Clip to [0, 1]
            #print('dataset_type',dataset_type)
            return action.cpu().detach().numpy()
        
    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        next_state = torch.FloatTensor(next_state).to(device)  #.values state is a dataframe

        # Critic loss
        q_value = self.critic(state, action)
        next_action, _ = self.actor(next_state)
        next_q_value = self.critic(next_state, next_action.detach())
        target_q = reward + 0.99 * next_q_value * (1 - done)
        critic_loss = nn.MSELoss()(q_value, target_q)

        # Actor loss
        action_mean, _ = self.actor(state)
        actor_loss = -self.critic(state, action_mean).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
    def save_model(self, primary_key):
        #if not os.path.exists('models'):
        #    os.makedirs('models')

        actor_path = f'{PATH}/checkpoint/sac_actor/{primary_key}.pth'
        critic_path = f'{PATH}/checkpoint/sac_critic/{primary_key}.pth'

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
    def load_model(self, primary_key):
        actor_path = f'{PATH}/checkpoint/sac_actor/{primary_key}.pth'
        critic_path = f'{PATH}/checkpoint/sac_critic/{primary_key}.pth'
    
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
            