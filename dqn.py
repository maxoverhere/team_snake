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

from Networks.tron_net import TronNet

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_final'))

class ReplayMem(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, n_batch):
        return random.sample(self.memory, n_batch)

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self, model_name='default'):
        super(TronPlayerDQN, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("running on", self.device)
        self.epoch = 0
        self.model_name = model_name
        self.target_update = 10
        # models
        self.policy_net = NeuralLink().to(self.device)
        # optimisers
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.load_weights()
        self.target_net = NeuralLink().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMem(10000)

    def preprocess(self, raw_state):
        board, _location = raw_state
        board = torch.from_numpy(board).unsqueeze(dim=0).float().to(self.device)
        state = torch.zeros_like(board).repeat(2, 1, 1)
        state[0, _location[0][0], _location[0][1]] = 1
        state[1, _location[1][0], _location[1][1]] = 1
        return torch.cat((board, state), dim=0)

    def get_action(self, raw_state):
        state = self.preprocess(raw_state).unsqueeze(dim=0)
        pred_v = self.policy_net(state)
        a = torch.argmax(pred_v, dim=1)
        # v = torch.gather(pred_v, dim=1, a)
        return a

    def train_model(self, n_batch, raw_state, action, next_raw_state, reward, end_game):
        self.memory.push(self.preprocess(raw_state), action,
            self.preprocess(next_raw_state).unsqueeze(dim=0), torch.tensor([reward], device=self.device), torch.tensor([end_game], device=self.device))
        if end_game:
            self.epoch += 1
            just_updated = True
        else:
            just_updated = False
        if len(self.memory) < n_batch:
            return
        transitions = self.memory.sample(n_batch)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = ~(torch.tensor(batch.is_final)) #flip is_final tensor
        non_final_next_states = [s for s, is_final in zip(batch.next_state, batch.is_final)
            if is_final == False]
        non_final_next_states = torch.cat(non_final_next_states) if len(non_final_next_states) != 0 else None
        # pass through network
        v = self.policy_net(state_batch)
        # print(v[0:10])
        # print(action_batch[0:10])
        pred_v = torch.gather(v, dim=1, index=action_batch.unsqueeze(dim=0))
        # print(pred_v.shape, pred_v[0:10])
        # calculate actual
        next_v_ = torch.zeros(n_batch, device=self.device)
        if non_final_next_states is not None:
            next_v_[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        actual_v = (next_v_ * 0.95) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(pred_v, actual_v.unsqueeze(0))
        # optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_param, self.max_g, norm_type=2) # to prevent gradient expansion, set max
        self.optimiser.step()
        # update target network if needed
        if just_updated and self.epoch % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_weights(self):
        fname = path.join('models', self.model_name)
        if os.path.exists(fname):
            checkpoint = torch.load(fname)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            print('Loaded with', self.epoch, 'epochs.')
        else:
            print('weights not found for', self.model_name)

    def save_weights(self):
        _filename = path.join('models', self.model_name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }, _filename)
        print('Model saved.')