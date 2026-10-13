import torch
import torch.nn as nn


class IMUEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(10, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 32)
        )

    def forward(self, x):

        return self.encoder(x)
