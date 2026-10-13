import torch
import torch.nn as nn


class WheelEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 32)
        )

    def forward(self, x):
        return self.encoder(x)
