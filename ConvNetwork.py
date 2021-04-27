import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Code taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "is_final"))


class ReplayMemory:
    __slots__ = ["capacity", "memory", "position"]

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        size = len(self.memory)
        size = batch_size if size >= batch_size else size
        sample = random.sample(self.memory, size)
        return list(filter(None, sample))

    def set_capacity(self, capacity):
        self.capacity = capacity
        if len(self.memory) > self.capacity:
            self.memory = self.memory[: self.capacity]
            self.position = 0

    def random_clean_memory(self, size):
        if size <= len(self.memory):
            self.memory = random.sample(self.memory, size)
            self.position = size

    def __len__(self):
        return len(self.memory)

class StandardConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        # fc layers
        self.fc1 = nn.Linear(128, 256) #5by5 = 32, 10by10 = 1152, 6by6 = 128
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        print(x.shape)
        val = F.relu(self.fc1(x))
        val = self.fc2(val)
        return val

class DuelingConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        # adv layers
        self.fc_adv1 = nn.Linear(1152, 256)
        self.fc_adv2 = nn.Linear(256, 4)
        # value layers
        self.fc_val1 = nn.Linear(1152, 256)
        self.fc_val2 = nn.Linear(256, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        val = F.relu(self.fc_val1(x))
        val = self.fc_val2(val)

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        return val + adv - adv.mean()
