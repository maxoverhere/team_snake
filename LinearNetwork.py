import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(75, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, board):
        board = self.fc1(board.view(-1, 75))
        board = F.relu(board)
        board = self.fc2(board)
        return board
