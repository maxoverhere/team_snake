import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(25, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)

    def forward(self, board):
        board = self.fc1(board.squeeze(dim=1).view(-1, 25))
        board = F.relu(board)
        board = self.fc2(board)
        board = F.relu(board)
        board = self.fc3(board)
        board = F.relu(board)
        board = self.fc4(board)
        return board
