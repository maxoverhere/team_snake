import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralLink(nn.Module):
    def __init__(self):
        super(NeuralLink, self).__init__()

        ## For Board
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 2, 3, padding=1)

        ## For combined Board
        self.fc1 = nn.Linear(16 * 16 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, board):
        board = F.relu(self.conv1(board))
        board = F.relu(self.conv2(board))
        board = F.interpolate(board, size=(16, 16),
                              mode='bicubic',
                              align_corners=False)
        board = board.view(-1, 2 * 16 * 16)
        combined = F.relu(self.fc1(board))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        return self.fc4(combined)
