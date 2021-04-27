import os
import os.path as path
import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from ConvNetwork import *
from LinearNetwork import *

"""
Implement:
 - [X] Double Q Learning
 - [] Prioritised Experience Replay
 - [X] Dueling networks
 - [] Multi-step learning
 - [] Distributed reinforcement learning
 - [] Noisy linear layers
"""

class DQNWrapper:
    def __init__(self, model_config, optimizer_config):
        super(DQNWrapper, self).__init__()
        print("help")
        #DEVICE and other basics
        self.device = model_config['device']
        self.save_name = model_config['save_name']
        self.target_update = optimizer_config['targetNetUpdate']
        self.train_n_batch = optimizer_config['trainBatchSize']
        self.epsilon = optimizer_config['epsilon']
        self.memory_max = optimizer_config['replayMemorySize']
        self.memory = ReplayMemory(self.memory_max)
        #OPTIONS
        self.is_dueling = model_config['is_dueling']
        self.is_doubleL = model_config['is_doubleL']
        #MODELS
        self.epoch = 0
        if (self.is_dueling):
            self.policy_net = DuelingConvNet().to(self.device)
            self.target_net = DuelingConvNet().to(self.device)
        else:
            # self.policy_net = StandardConvNet().to(self.device)
            self.policy_net = StandardConvNet().to(self.device)
            self.target_net = StandardConvNet().to(self.device)
            # self.target_net = StandardConvNet().to(self.device)
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=optimizer_config['lr'])
        # load stuff
        self.load_weights() #load weights for policy net
        self.target_net.load_state_dict(self.policy_net.state_dict()) #load target net from policy net

    #state is np array
    def get_action(self, state):
        if len(self.memory) < self.memory_max or random.random() < self.epsilon: #exploit
            return np.random.randint(3)
        else:
            pred_v = self.policy_net(state.to(self.device))
            return torch.argmax(pred_v, dim=1).item()

    def is_mem_full(self):
        return len(self.memory) < self.memory_max

    def train_model(self, state, action, next_state, reward, end_game):
        self.target_net.eval()
        self.policy_net.train()
        self.memory.push(state.to(self.device), torch.tensor([action], device=self.device),
            next_state.to(self.device), torch.tensor([reward], device=self.device), torch.tensor([end_game], device=self.device))
        if len(self.memory) < self.train_n_batch:
            return -1
        transitions = self.memory.sample(self.train_n_batch)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = ~(torch.tensor(batch.is_final)) #flip is_final tensor
        non_final_next_states = [s for s, is_final in zip(batch.next_state, batch.is_final)
            if is_final == False]
        non_final_next_states = torch.cat(non_final_next_states) if len(non_final_next_states) != 0 else None
        # pass state through policy net to get predicted v
        v = self.policy_net(state_batch)
        pred_v = torch.gather(v, dim=1, index=action_batch.unsqueeze(dim=0))
        # calculate actual v
        if (self.is_doubleL):
            # use policy_net to determine next action
            next_action = self.policy_net(non_final_next_states).max(1)[0].detach()
            # use target_net to predict next_next_state value
            non_final_next_v_ = self.target_net(non_final_next_states).gather(1, next_action).squeeze()
        else:
            non_final_next_v_ = self.target_net(non_final_next_states).max(1)[0].detach()
        next_v_ = torch.zeros(self.train_n_batch, device=self.device)
        if non_final_next_states is not None:
            next_v_[non_final_mask] = non_final_next_v_.detach()
        actual_v = (next_v_ * (0.95 ** 1)) + reward_batch
        # ALSO NEED TO UPDATE REWARD TO BE N_STEPS
        # Compute Huber loss
        loss = nn.MSELoss()
        loss = loss(pred_v, actual_v.unsqueeze(0))
        # loss = F.smooth_l1_loss(actual_v.unsqueeze(0), pred_v)
        # optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_param, self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.optimiser.step()
        # update target network if needed
        if end_game:
            self.epoch += 1
            if self.epoch % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss

    def load_weights(self):
        fname = path.join('saved_models', self.save_name)
        if os.path.exists(fname):
            checkpoint = torch.load(fname)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            print('Loaded with', self.epoch, 'epochs.')
        else:
            print('weights not found for', self.save_name)

    def save_weights(self):
        _filename = path.join('saved_models', self.save_name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }, _filename)
        print('Model saved.')
